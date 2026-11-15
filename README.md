This is my own implementation of MDLM from scratch without using AI.  The implementation is loosely based on Shi et al.'s MD4, but much simplified.

The dataset uses tiny-shakespeare, and tokenisation happens at the character level. This is just a toy example to make sure I've understood the paper thoroughly.

It trained for 10 hours on 2 T4 provided by Kaggle. 
The result went from:
LH,oHoogIvjoaJOpJVM?oEh XNhyBR\nv, $Nnu\n;&dfbtw&P.dTMA\n'h&dt!pO?hSpHFtTF'3$bQ$FW&q,'zlkW!xvXEJOn;l$vQyZsmJbGdhXm:m$QP:IUAOvVfhgtB

to this:

. and brothes,

Mine too laws envoian which affieal music, igainst

Thou will confuse your that is surong,

And furom duke this sat

which although contains many spelling mistake and is ultimately still non-sense, the improvement is evident. Furthermore, the mistakes are expected of a similarly sized AR model, such as Andrej Karpathy's nanoGPT. 
