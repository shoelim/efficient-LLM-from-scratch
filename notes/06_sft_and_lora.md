# 06 — Supervised fine-tuning and low-rank adapters

## Response-only objective

Each SFT example is a `prompt` and an expected `response`. We concatenate
`prompt_ids + response_ids + [EOS]` and create a causal shifted `(x,y)` pair.
But targets that belong to the prompt are set to -100, the PyTorch ignore
index, so only response tokens (including its final EOS) contribute to loss.
The **first response token must remain supervised**: its predictive input
position is the last token of the prompt. Off-by-one masking mistakes often
silently train the wrong objective. See `test_sft_masks_prompt_but_not_first_response`.

## LoRA equation

For a frozen linear transformation `y=Wx+b`, choose low-rank matrices
`A∈R^{r×din}` and `B∈R^{dout×r}`. Replace the linear transformation with

\[
y = Wx+b + (\alpha/r)BAx.
\]

Initialize `A` randomly and `B=0`; at initialization, the adapted model
behaves exactly like the original model. The original `W,b` remain frozen,
while autograd learns only `A,B`. At inference time, `BA` may be added to
`W` (merged) for the same linear function up to floating-point rounding.
The adapter includes only the low-rank matrices, not a standalone model; it
requires the base checkpoint.

Our default targeting is attention query and value projections only. This is
an educational LoRA example, not an optimized adapter training recipe.

## Validation and limits

The SFT CLI reserves whole JSONL examples for validation; it does not sample
parts of an example for both sides. It deliberately errors on overlong
examples rather than truncating away parts of a response. Small toy SFT data
is not an alignment or instruction-following benchmark. `sft` trains
separate weights or adapters but cannot make a barely pretrained language
model suddenly high-quality.

## Exercises

1. Trace label masking for prompt `[1,2]`, response `[3,4]`, EOS `5`.
2. Why is the LoRA output unchanged at initialization?
3. Count trainable adapter parameters for a 32×32 linear layer at rank 4.
4. Prove that the merge is exact in real arithmetic, and explain sources of
   small numerical differences in floating-point computations.
