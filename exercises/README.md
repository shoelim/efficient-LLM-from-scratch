# Exercises: intentionally incomplete; production code is complete

This directory is the place to practice independently. **TODOs are only here**.
The runnable library under `src/llmfs/` has no educational placeholders.

Do the first exercise before reading its solution. For later exercises, use
the cited production module only to check your independent reasoning.

| Exercise | Skill | Reference after attempting |
|---|---|---|
| `00_python_basics.py` | Dictionaries, windows and lists | `solutions/00_python_basics.py` |
| `01_bigram.py` | Conditional probabilities, NLL | `evaluation/baselines.py` |
| `02_autograd.py` | Shapes, softmax gradients | `notes/01_autoregressive_lm.md` |
| `05_attention.py` | Causal masking, einsum | `model/attention.py` |
| `07_adamw.py` | Moments and bias correction | `training/optim.py` |
| `10_sampling.py` | Temperature, top-k, top-p | `inference/sample.py` |

Suggested process: implement, write a failing test, pass it, compare with
reference, then commit. Import modules directly from the `exercises` path if
you want to test your partial solutions. They are never imported by the main
training CLI.
