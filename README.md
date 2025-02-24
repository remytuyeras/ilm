# ilm
**Intuitionistic Language Models** are language models whose tokenization is defined relative to the structure of the language itself, rather than relying on a fixed alphabet. Their representative units are derived through a precise algorithmic construction, reflecting the intrinsic patterns of the language.

There are approximately $N=250000$ words in english language. This means that every word can be encoded as a sequence of $\lceil \log_2(N) \rceil = 18$.

Since $18 = 3 \cdot 6$, we can encode these words are sequence of $6$ syllables, which would belong to a set of $2^3 = 8$ values.

For instance $\texttt{the} = 001200$. Each of the syllables could have a hidden intuitionistic meaning based on the language itself.

Assume that we want to infrer $\texttt{word} = w_0w_1w_2w_3w_4w_5$: 

- $w_0$ would be inferred by computing as follows:
    - compute the statistical positioning of $\texttt{word}$ in a sentence;
    - partition the set of words ordered as such in 8 bins;
    - attribute the index associated with $\texttt{word}$ to $w_0$;

- [NEED TO FIX] $w_{1}$ relies on $w_{0}$ as follows:
    -   for any $w \geq w_0$ for which there exists a sentence of the form
        $$w\square\square\square\square\square \quad \dots \quad \texttt{word}$$
        compute the statistical positioning of $\texttt{word}$.

- [NEED TO FIX] $w_{2}$ relies on $w_{1}$ as follows:
    -   for any $w \geq w_1$ for which there exists a sentence of the form
        $$\square w\square\square\square\square \quad \dots \quad \texttt{word}$$
        compute the statistical positioning of $\texttt{word}$.

For example, if we have:
 - $\texttt{The }=000000$
 - $\texttt{scientist }=000000$
 - $\texttt{carefully }=000000$
 - $\texttt{analyzed }=100000$ 
 - $\texttt{the }=100000$
 - $\texttt{data }=200000$
 - $\texttt{, }=400000$
 - $\texttt{searching }=500000$
 - $\texttt{for }=600000$
 - $\texttt{patterns }=500000$
 - $\texttt{in }=200000$
 - $\texttt{the }=100000$ 
 - $\texttt{results.}=700000$