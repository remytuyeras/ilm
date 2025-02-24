# ilm
**Intuitionistic Language Models** are language models whose tokenization is defined relative to the structure of the language itself, rather than relying on a fixed alphabet. Their representative units are derived through a precise algorithmic construction, reflecting the intrinsic patterns of the language.

There are approximately $N=250000$ words in english language. This means that every word can be encoded as a sequence of $\lceil \log_2(N) \rceil = 18$.

Since $18 = 3 \cdot 6$, we can encode these words are sequence of $3$ syllables, which would be made out of $2^6 = 64$ possible letters.

Since our alphabet is made of $2^6$, we can have a NN with 6 outputs that returns the binary decomposition of the letter. Since the NN gives the actual letter (and not the probability associated with the 64 letters), we can direct use the output as the next token in the sentence.

<div align="center">
  <img src="img/nn.png" alt="Neural Network Diagram" width="400">
</div>


For instance $\texttt{the} = [0,1,4]$. Each of the syllables could have a hidden intuitionistic meaning based on the language itself.

Assume that we want to infrer $\texttt{word} = w_0w_1w_2$: 

- $w_0$ would be inferred by computing as follows:
    - compute the statistical positioning of $\texttt{word}$ in a sentence;
    - partition the set of words ordered as such in 8 bins;
    - attribute the index associated with $\texttt{word}$ to $w_0$.

- $w_{i+1}$ relies on $w_{0}\dots w_{i}$ as follows:
    - compute the relative statistical positioning of $\texttt{word}$ among all words starting with $w_{0}\dots w_{i}$ for any available sentence;
      
    $$[w_0,a,b] (\to 0) \quad\dots\quad [w_0,c,d] (\to 1) \quad\dots\quad [w_0,e,f] (\to 2)$$
  
    - partition the set of words ordered as such in 8 bins;
    - attribute the index associated with $\texttt{word}$ to $w_{i+1}$
