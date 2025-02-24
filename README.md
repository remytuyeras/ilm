# ilm
**Intuitionistic Language Models** are language models whose tokenization is defined relative to the structure of the language itself, rather than relying on a fixed alphabet. Their representative units are derived through a precise algorithmic construction, reflecting the intrinsic patterns of the language.

There are approximately $N=250000$ words in english language. This means that every word can be encoded as a sequence of $\lceil \log_2(N) \rceil = 18$.

Since $18 = 3 \cdot 6$, we can encode these words are sequence of $6$ syllables, which would belong to a set of $2^3 = 8$ values.

For instance $\mathtt{the} = 001200$. Each of the syllables could have a hidden intuitionistic meaning based on the language itself.

Assume that we want to infrer $\mathtt{word} = w_0w_1w_2w_3w_4w_5$: 

- $w_0$ would be inferred by computing as follows:
    - compute the statistical positioning of $\mathtt{word}$ in a sentence;
    - partition the set of words ordered as such in 8 bins;
    - attribute the index associated with $\mathtt{word}$ to $w_0$;

- $w_{1}$ relies on $w_{0}$ as follows:
    -   for any $w \geq w_0$ for which there exists a sentence of the form
        $$w\square\square\square\square\square \quad \dots \quad \mathtt{word}$$
        compute the statistical positioning of $\mathtt{word}$.

- $w_{2}$ relies on $w_{1}$ as follows:
    -   for any $w \geq w_1$ for which there exists a sentence of the form
        $$\square w\square\square\square\square \quad \dots \quad \mathtt{word}$$
        compute the statistical positioning of $\mathtt{word}$.

For example, if we have:
 - $\mathtt{The\_}=000000$
 - $\mathtt{scientist\_}=100000$
 - $\mathtt{carefully\_}=200000$
 - $\mathtt{analyzed\_}=300000$ 
 - $\mathtt{the\_}=400000$
 - $\mathtt{data\_}=500000$
 - $\mathtt{,\_}=600000$
 - $\mathtt{searching\_}=700000$
 - $\mathtt{for\_}=010000$
 - $\mathtt{patterns\_}=020000$
 - $\mathtt{in\_}=030000$
 - $\mathtt{the\_}=400000$ 
 - $\mathtt{results.}=050000$