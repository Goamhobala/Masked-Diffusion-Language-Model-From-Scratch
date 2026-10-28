"""
Toy MDLM (MD4, Shi et al. 2024) on Tiny Shakespeare, character level.
This is my attempt at replicating a minimum version of it from scratch.
"""

import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *




if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(0)

# data loading
def load_text(path=DATA_PATH):
    if not os.path.exists(path):
        print(f"downloading Tiny Shakespeare -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


text = load_text()
chars = sorted(set(text))

VOCAB = len(chars)   # m: number of real tokens (65 for Tiny Shakespeare)
MASK = VOCAB         # the mask token gets the extra index m
MASK_ONE_HOT = F.one_hot(torch.tensor([MASK], device=device), VOCAB + 1)


char_to_int = {c: i for i, c in enumerate(chars)} # char to int (actual char to the integerId)
int_to_char = {i: c for i, c in enumerate(chars)} # the other way around


# def encode(s):
#     """str -> (len,) long tensor of token ids"""
#     return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def encode (s):
    """
    Str -> torch.tensor Long 
    convert the sequence into a tensor of token ids
    """
    return torch.tensor([char_to_int[c] for c in s], dtype=torch.long)

def decode(ids, mask_char="_"):
    """
    torch.tensor Long -> Char -> Str
    takes in a tensor of ids, and converts it to a list, then convert it back to a sequence
    """
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return "".join(mask_char if i == MASK else int_to_char[i] for i in ids)


data = encode(text)
n_train = int(0.9 * len(data)) # train val split
train_data, val_data = data[:n_train], data[n_train:]


def get_batch(split="train", batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
    """
    Split String -> Int -> Int -> [String]
    Get a batch of sequences
    """
    d = train_data if split == "train" else val_data

    ix = torch.randint(len(d) - seq_len, (batch_size,)) # from 0 to len(d) - seq_len, and returns a random batch_size of ints
    x0 = torch.stack([d[i : i + seq_len] for i in ix]) # for each start length, we slice out a seq_len portion from the data, and returns it to device
    return x0.to(device) # (B, SEQ_LEN)



# denoiser. Remember we're using the neural net to predict x0.

class Denoiser(nn.Module):
    """
    Bidirectional transformer encoder (no causal mask), so every position
    attends to every other position. The encoder is followed by a linear predictor (to actually unmask the token
    This predicts the x0
    """

    def __init__(self, vocab_size=VOCAB, seq_len=SEQ_LEN, d_model=128,
                 n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size + 1, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, xt):
        B, N = xt.shape
        pos = torch.arange(N, device=xt.device)
        h = self.tok_emb(xt) + self.pos_emb(pos)   # (B, N, d_model)
        h = self.encoder(h)                         # (B, N, d_model)
        return self.head(self.norm(h))              # (B, N, VOCAB) raw logits



def alpha(t):
    """t: (B,) float in [0, 1]  ->  alpha_t: (B,)
    here we just use a linear scheduler
    """
    
    return 1 - t


def weight(t):
    """t: (B,)  ->  alpha'_t / (1 - alpha_t): (B,)   (watch t -> 0)
    alpha'_t is the derivative of alpha_t with respect to t
    It's technically just -1 in this case because we're using a linear function. 
    But to keep the code modular and for practice's sake, I'm using autograd
    
    """
    with torch.enable_grad():
        t= t.detach().requires_grad_(True) # detach creates a copy with no history
        alpha_t = alpha(t)
        alpha_prime = torch.autograd.grad(alpha_t, t, grad_outputs=torch.ones_like(alpha_t))[0]

    return (alpha_prime) / (1-alpha_t)
     



def add_noise(x0, t):
    """
    x0: (B, N) long, t: (B,) float  ->  xt: (B, N) long
    Each token independently becomes MASK with probability 1 - alpha_t.
    Forward process produces the training data
    """
    batch_size, seq_len= x0.shape
    mask_rates = (1 - alpha(t)).unsqueeze(-1).expand(-1, seq_len) # need to match with the dimension of thresholds
    thresholds = torch.rand(batch_size, seq_len, device=x0.device)
    x1 = torch.where(thresholds > mask_rates, x0, MASK)

    return x1



def mdlm_loss(logits, x0, xt, t):
    """
    logits: (B, N, VOCAB), x0: (B, N), xt: (B, N), t: (B,)  ->  scalar
    """
    masked_pos = xt == MASK                                   # (B, N) bool, no need for torch.where
    ce = F.cross_entropy(logits[masked_pos], x0[masked_pos], reduction="none")  # none means no aveaging/sum
    w = weight(t)[:, None].expand_as(xt)[masked_pos]         # (M,)
    return (-w * ce).sum()
     
@torch.no_grad()
def sample(model, n_samples=4, steps=256, seq_len=SEQ_LEN):
    """
    Start from all-MASK, returns all steps and run `steps` reverse steps
    from t=1 down to t=0. Returns (n_samples, seq_len, steps) long. All steps
    are returned for analysis
    
    This follows the section F.2 from Shi et al.
    """
    discrete_steps = torch.linspace(0, 1, steps + 1)
    step = steps
    xT = torch.full((n_samples, seq_len), MASK, device=device)
    x_store = torch.zeros(n_samples, seq_len, steps+1, dtype=torch.long, device=device) # to visualise trajectories
    x_store[:,:, steps] = xT
    while step > 0:
        t = discrete_steps[step]
        s = discrete_steps[step-1]
        xt= x_store[:, :, step]
        masked_pos = xt == MASK
        x0_logits = model(xt) # (n_samples, seqlen, VOCAB)
        # x0_logits[:, :, MASK] = -torch.inf

        mu0 = F.softmax(x0_logits, dim=-1)
        mu0 = F.pad(mu0, (0, 1)) # make last column zero
        
        probs = (alpha(s) - alpha(t))/(1-alpha(t)) * mu0 + (1-alpha(s))/(1-alpha(t)) * MASK_ONE_HOT
        x0_predicted = torch.distributions.Categorical(probs=probs).sample()


        xs = torch.where(masked_pos, x0_predicted, xt)         
        step -= 1
        x_store[:, : ,step] = xs


    return x_store



def count_params(model):
    return sum(p.numel() for p in model.parameters())


def smoke_test():
    """Checks the scaffolding only (data + model shapes)."""
    x0 = get_batch("train", batch_size=2)
    print("device:", device, "| vocab:", VOCAB, "| MASK id:", MASK)
    print("x0 shape:", tuple(x0.shape))
    print("sample text:", repr(decode(x0[0])[:80]))

    model = Denoiser().to(device)
    xt = x0.clone()
    xt[:, ::3] = MASK                     # fake masking, just for the shape check
    logits = model(xt)
    print("masked view:", repr(decode(xt[0])[:80]))
    print("logits shape:", tuple(logits.shape), "(expect (2, %d, %d))" % (SEQ_LEN, VOCAB))
    print("params: %.2fM" % (count_params(model) / 1e6))



# training loop



def estimate_val_loss(model, n_batches=VAL_BATCHES):
    """Same procedure as a training step (random t, add_noise, forward, loss),
    on held-out data, averaged over a few batches. Returns loss per token."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x0 = get_batch("val")
            t = 1 - torch.rand(x0.shape[0], device=x0.device)
            xt = add_noise(x0, t)
            total += mdlm_loss(model(xt), x0, xt, t).item()
    model.train()
    return total / (n_batches * BATCH_SIZE * SEQ_LEN)


if __name__ == "__main__":
    model = Denoiser().to(device)
    print("params: %.2fM on %s" % (count_params(model) / 1e6, device))
    steps = 32
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
    step_losses =[] # for tracking
    for step in range(N_STEPS):
        model.train()
        
        training_batch = get_batch() #(B, SEQ_LEN)
        # 1 - is to turn the rand to (0, 1] instead
        time_steps = 1 - torch.rand(BATCH_SIZE, device=training_batch.device) #(B,). Also have to make sure they are on the same device

        masked_batch = add_noise(training_batch, time_steps) 
        logits_batch = model(masked_batch)

        loss = mdlm_loss(logits_batch, x0=training_batch, xt=masked_batch, t = time_steps)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        step_losses.append(loss.item())
        


        if step % VALID_LOSS_EVERY == 0:
            val_loss = estimate_val_loss(model)
            print("step %d | train loss/token %.4f | val loss/token %.4f"
                  % (step, loss.item() / (BATCH_SIZE * SEQ_LEN), val_loss))

        if step % SAMPLE_EVERY == 0:
            model.eval()
            generated = sample(model, n_samples=1)[0, :, 0]   # t=0 end of the first sample's trajectory
            model.train()
            print("sample:", repr(decode(generated)))

