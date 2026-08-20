# Model Architecture

ILM models a lexical entry as a fixed sequence of coordinate events. For a
base-$K$ code of depth $S$,

$$
\phi(w_n) = (c_0(w_n), \ldots, c_{S-1}(w_n)),
\qquad c_p(w_n) \in \{0, \ldots, K - 1\}.
$$

The tokenizer flattens lexical codes into one causal stream. With lexical index
$n$ and coordinate role $p$,

$$
x_{Sn+p} = c_p(w_n).
$$

For example, a depth-three tokenizer may encode `"The king"` as
`(2, 54, 30, 50, 54, 10)`. The usual next-event shift therefore learns both
within-entry predictions and transitions to the next lexical entry.

## Flat ILM

Flat ILM applies an ordinary decoder-only Transformer to this flattened stream.
It uses a coordinate vocabulary of size $K$, learned token embeddings, learned
time-position embeddings, causal self-attention, feed-forward blocks, and one
shared $K$-way output head. The attention computation is the standard causal
Transformer computation. The coordinate representation changes the event stream
presented to the model, not the attention kernel.

The standard teacher-forced loss is coordinate-level cross-entropy:

$$
\mathcal{L}_{\mathrm{coord}}
=
-\frac{1}{BT}
\sum_{b=1}^{B}\sum_{t=0}^{T-1}
\log P_\theta(y_{b,t}\mid x_{b,<t}).
$$

## Full ILM

Full ILM retains the same causal Transformer stack and coordinate stream. It
enables three independent options:

| CLI option | Effect |
| --- | --- |
| `--ilm-input-embeddings` | Learns a separate input embedding for each pair of coordinate value and coordinate role. |
| `--ilm-output-heads` | Learns one $K$-way output head for each predicted coordinate role. |
| `--ilm-objective` | Masks next-event losses whose context lies in a partial lexical code at the left edge of a sampled training window. |

The first two options change checkpoint tensor shapes. The objective changes
only training loss selection. It does not modify attention and it is not used
during generation.

### Coordinate-role input embeddings

Let $r_{b,t} = (r_b+t) \bmod S$ be the role of input event $x_{b,t}$. Full ILM
uses an embedding table $E_{\mathrm{tok}} \in \mathbb{R}^{SK \times C}$ indexed
by both the role and coordinate value:

$$
e_{b,t} = E_{\mathrm{tok}}[r_{b,t}K + x_{b,t}].
$$

This gives the same coordinate value a different learned vector when it occurs
at different positions within a lexical code.

### Coordinate-role output heads

The target at context position $t$ has role

$$
\rho_{b,t} = (r_b+t+1) \bmod S.
$$

When `--ilm-output-heads` is enabled, the hidden state $h_{b,t}$ is scored by
the matching role-specific head:

$$
z_{b,t} = W_{\mathrm{lm},\rho_{b,t}}h_{b,t} + b_{\mathrm{lm},\rho_{b,t}}.
$$

## Word-prefix objective

A sampled window can start at role one or two of a depth-three lexical code.
The beginning of that window then contains a suffix without the earlier
coordinates needed to construct the lexical entry from its start. Full ILM can
exclude those losses.

If $r_b$ is the role of the first context event in batch item $b$, the number
of masked initial positions is

$$
\delta_b = (-r_b) \bmod S.
$$

The binary prefix mask is

$$
m^{\mathrm{prefix}}_{b,t}
=
\begin{cases}
0, & 0 \leq t < \delta_b,\\
1, & \delta_b \leq t < T.
\end{cases}
$$

For a depth-three stream beginning at role one, such as
`x = (54, 30, 50, 54, 10, ...)`, the mask begins `(0, 0, 1, 1, ...)`.
The first retained prediction has a context that includes the first coordinate
of the next lexical entry. The implementation constructs this selection through
a binary alignment tensor, but attention and generation remain ordinary
coordinate-time computations.

See [training.md](training.md) for compatible window geometry and
[decoding.md](decoding.md) for generation.
