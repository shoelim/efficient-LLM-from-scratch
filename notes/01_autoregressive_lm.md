# 01 — From probability to next-token training

## What is a language model?

Write the corpus as tokens `x_0, ..., x_{T-1}` in a finite alphabet of size V.
The probability chain rule is an identity, not a modeling assumption:

\[
p(x_0,\ldots,x_{T-1})=p(x_0)\prod_{t=1}^{T-1}p(x_t\mid x_0,\ldots,x_{t-1}).
\]

A decoder-only Transformer parametrizes each conditional distribution by a
function of the preceding tokens. At time t, logits `z_t∈R^V` become token
probabilities by softmax:

\[
p_\theta(x_{t+1}=j\mid x_{\le t})
=\frac{e^{z_{t,j}}}{\sum_{k=0}^{V-1}e^{z_{t,k}}}.
\]

The teacher-forced maximum-likelihood loss of a packed window is

\[
L(\theta)=-\frac{1}{T-1}\sum_{t=0}^{T-2}
\log p_\theta(x_{t+1}\mid x_0,\ldots,x_t).
\]

**Important distinction:** the model sees the *true prefix* during training;
at generation time, its own previous sampled tokens become the prefix. Our
`get_batch` constructs `input = ids[i:i+block_size]` and
`target = ids[i+1:i+block_size+1]`; thus row t of the output predicts the
next token. The training loop doesn't require a separate inference target.

## Small worked example

Suppose we see two tokens `a`, `b`, with a smoothed bigram vocabulary
`{a,b}`. The training stream `a b a a` contains pairs
`(a,b), (b,a), (a,a)`. With Laplace smoothing α=1:

\[
p(b\mid a)=(1+1)/(2+2)=1/2, \quad
p(a\mid b)=(1+1)/(1+2)=2/3.
\]

The baseline is a table of conditional probabilities, not a neural network.
It can be fit in one pass over adjacent tokens. Look at
`src/llmfs/evaluation/baselines.py` and check that `BigramLM.fit([0,1,0,0])`
returns these exact probabilities.

A network outputting all-zero logits over V tokens yields probability `1/V`
for each token, so its cross-entropy is `log(V)`. This is a sanity check, not
an expected final model loss.

## Loss, perplexity, and evaluation

If average next-token negative log likelihood is `L` measured in nats, then
`perplexity = exp(L)`. It is the exponential of a **token-level** quantity.
A byte-level BPE tokenizer and a character tokenizer generally have different
numbers of tokens per byte; their raw perplexities are not apples-to-apples.
For matched corpus coverage, use total NLL divided by raw UTF-8 bytes (bits
per byte) rather than comparing raw token perplexity across tokenizers.

**Limit of current implementation:** our `evaluate` command computes loss on a
fixed-seed random sample of packed windows. It is not an exact total corpus
likelihood or a bits-per-byte implementation.

## Student checkpoint

1. Prove the chain-rule identity for three tokens.
2. Compute the bigram probabilities above with α=0.5 and α=1.
3. Derive `∂(-log softmax(z)[y])/∂z_j = softmax(z)_j - 1[j=y]`.
4. Open `tests/test_model.py` and identify the line testing `log V`.
