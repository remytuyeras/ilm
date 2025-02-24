# ilm
**Intuitionistic Language Models** are language models whose tokenization is defined relative to the structure of the language itself, rather than relying on a fixed alphabet. Their representative units are derived through a precise algorithmic construction, reflecting the intrinsic patterns of the language.

There are approximately $N=250000$ words in english language. This means that every word can be encoded as a sequence of $\lceil \log_2(N) \rceil = 18$.

Since $18 = 3 \cdot 6$, we can encode these words are sequence of $3$ syllables, which would be made out of $2^6 = 64$ possible letters.

For instance $\texttt{the} = [0,1,4]$. Each of the syllables could have a hidden intuitionistic meaning based on the language itself.

Assume that we want to infrer $\texttt{word} = w_0w_1w_2$: 

- $w_0$ would be inferred by computing as follows:
    - compute the statistical positioning of $\texttt{word}$ in a sentence;
    - partition the set of words ordered as such in 8 bins;
    - attribute the index associated with $\texttt{word}$ to $w_0$.

- $w_{i+1}$ relies on $w_{0}\dots w_{i}$ as follows:
    - compute the relative statistical positioning of $\texttt{word}$ among all words starting with $w_{0}\dots w_{i}$ for any available sentence;
    $$[w_0,a,b]_{=0} \quad\dots\quad [w_0,c,d]_{=1} \quad\dots\quad [w_0,e,f]_{=2}$$
    - partition the set of words ordered as such in 8 bins;
    - attribute the index associated with $\texttt{word}$ to $w_{i+1}$
