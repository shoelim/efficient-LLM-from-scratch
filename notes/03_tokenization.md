# 03 — Data and byte-level byte-pair encoding (BPE)

A tokenizer maps a Unicode string to discrete IDs and back. A character
vocabulary is readable but treats unfamiliar characters as an error. UTF-8
bytes give a base vocabulary of size 256 that can represent arbitrary strings.
We reserve ID 256 for `<eos>` and start merged byte sequences at ID 257.

## BPE construction

1. Convert each training document to a sequence of UTF-8 byte IDs.
2. Count every adjacent ID pair **within documents only**.
3. Select a most frequent pair (deterministic tie-break: smallest pair).
4. Add a token whose byte string is the concatenation of the pair's strings.
5. Replace every non-overlapping occurrence of the pair in every training
   document, then count again, until the vocabulary budget is reached or no
   pair reaches the frequency threshold.

A toy example: on `abab, abab`, pair `(a,b)` occurs four times, exceeding
`(b,a)` twice. It is merged to ID 257; `abab` becomes `[257,257]`.

## Encoding new text

Start from its UTF-8 bytes. Apply the learned merge operations **in exactly
training order**, not by merging whichever adjacent pair appears most
frequent in the new text. `decode` concatenates the bytes of tokens and
converts UTF-8 bytes back to Unicode. The text roundtrip is exact on complete
UTF-8 strings, including previously unseen characters.

**Caveat:** sampling may stop halfway through a multi-byte character.
`ByteBPETokenizer.decode(..., errors='replace')` is appropriate for displaying
partial generations, whereas `errors='strict'` is better for tokenizer unit
tests because it exposes truncation.

## Train / validation separation

`prepare` deduplicates complete input documents, shuffles with an explicit
seed, sets aside validation documents, then fits the tokenizer on training
text ONLY. It then encodes both splits, appending EOS after each document and
saving the token streams separately. EOS boundary packing is deliberate; this
is *not* a no-cross-document-attention training objective.

The current implementation is simple but computationally costly: every merge
requires recounting pairs across the sequences. This is pedagogical. Scaling
to million-document corpora calls for caching pair counts, pretokenization,
parallel processing, or an established tokenizer implementation—but doing so
changes the performance engineering exercise, not the semantics tested here.

## Exercises

1. Why does byte BPE need no unknown token for valid Unicode text?
2. What goes wrong when two BPE training runs choose different tied pairs?
3. Trace merges for a two-document corpus with five characters by hand.
4. Explain why fitting the tokenizer on held-out text leaks information.
