# 04 — Training loop, AdamW, and checkpoints

For token targets `y`, the per-position cross-entropy can be written in stable
log-sum-exp form:

\[
\ell(z,y)=\log\sum_{j=1}^V e^{z_j}-z_y.
\]

PyTorch's `F.cross_entropy` combines stable log-softmax and a negative log
likelihood. We do not compute `log(softmax(z))` directly, as this can
underflow. `GPT.forward` receives a token input batch and same-shaped target
IDs, then averages the loss over positions (except targets with label -100).

## AdamW with decoupled weight decay

For gradient `g_t=∇L(θ_t)`, maintain first and second moments:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

Bias-correct them: `m̂_t = m_t/(1-β₁ᵗ)`,
`v̂_t = v_t/(1-β₂ᵗ)`. Then compute

\[
\theta_{t+1}=(1-\eta_t\lambda)\theta_t-
\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
\]

Weight decay is decoupled from the gradient moments; it is **not** the same as
adding `λθ` to `g_t` in the adaptive optimizer. Our `ScratchAdamW` is compared
to PyTorch AdamW on deterministic double-precision tensors across many steps.
The default optimizer groups decay matrix parameters and do not decay vectors
(norm scales and biases), which is a common but not unique training convention.

## Gradient accumulation

For `K` equal-size microbatches, sum `loss_k/K` *before* calling backward.
This gives the same mean gradient as a full batch with dropout disabled (up
to floating-point order). Scale then clip gradients only once, before the
optimizer step. The effective tokens per update are:

`batch_size × grad_accum_steps × block_size`.

## Learning-rate schedule

We use linear warmup over the first W optimizer steps then cosine decay over
the remaining steps to `min_lr`. The schedule is a pure function of the
zero-based optimizer step, facilitating exact resume. Changing the total
number of steps changes the schedule even at earlier steps; an exact-resume
run therefore rejects a changed config.

## What must a checkpoint remember?

Saving weights alone permits inference but not faithful training resumption.
Our `last.pt` contains optimizer moments, global Python/NumPy/PyTorch RNG
states, a separate training-window generator state, optional CUDA AMP scaler,
step count, both configs, and a dataset/tokenizer fingerprint. At fixed
software, hardware, data and numerical settings, this lets a run resume
without replaying earlier stochastic microbatches.

**Security:** load only trusted local `.pt` files; arbitrary torch/pickle
checkpoint objects may execute code on deserialization.

## Exercises

1. Starting from `m_0=v_0=0`, calculate the first AdamW step in 1D.
2. Explain why zeroing the gradient after each microbatch is wrong.
3. Show that doubling the number of tokens per step is not equivalent to
   doubling the learning rate.
4. Change only one seed and identify which training quantities move.
