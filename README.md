# Self-Improving Tiny LLM

A from-scratch transformer that improves itself — **zero third-party dependencies**,
pure Python, no numpy/torch. Built for teaching how a language model is trained
and how it can improve its own responses.

## What it does

1. **Base training** — a tiny decoder-only transformer (`MiniLM`, ~2k to ~107k params
   depending on flags) is trained on a hardcoded Q&A corpus with plain cross-entropy.
2. **Self-improvement loop** — the trained model improves *its own* answers with two
   methods, alternated round by round:
   - **Self-distillation**: generate candidate answers per question, score them with a
     keyword/length reward, and fine-tune on its own best-scoring completions.
   - **REINFORCE**: roll out sampled answers, score them, and nudge the parameters by
     the policy-gradient update `w = (seq_len - 1) * (reward - baseline)`.
3. **Eval** — prints the average reward before/after and a per-round curve, plus sample
   generations.

The reward function scores an answer by keyword hits, a full-hit bonus, length
closeness to the reference, starting with a letter / ending with punctuation, minus a
repetition penalty.

## Files

- `llm.py` — the transformer: `MiniLM`, Adam optimizer, tokenizer, causal multi-head
  attention with LayerNorm/GELU, full forward **and** backward pass, sampling.
- `improve.py` — corpus, reward, base training, distill + REINFORCE loops, CLI, chat.

## Run it

Requires Python 3.10+ (nothing else — no pip installs).

```bash
# quick smoke test (~30 s)
python improve.py --tiny

# full run with defaults (d_model=12, 1 layer, 8 rounds, 150 base epochs)
python improve.py

# bigger model (if you have patience — pure-Python training is slow)
python improve.py --d-model 32 --n-layers 3 --d-ff 64 --epochs 40 --rounds 8

# chat with the improved model
python improve.py --chat

# train in the cloud (GitHub Actions), then chat locally with the saved weights
python improve.py --load model.json --epochs 0 --rounds 0 --chat
```

## Cloud training (GitHub Actions)

The workflow `.github/workflows/train.yml` runs the heavy run (~100k params) on a
GitHub-hosted runner so your own machine is free. It uploads the full training log and
the trained weights (`model.json`) as an artifact:

- Push to `main`, or
- Actions → **Train** → *Run workflow* to choose epochs/rounds.

The training log prints every 5 epochs, then the before/after reward curve and example
generations. Grab `model.json` from the artifact and chat with it locally with
`python improve.py --load model.json --epochs 0 --rounds 0 --chat`.

## How to read the output

The self-improvement curve shows the average reward after base training (`0`) and after
each round. A positive `gain` means the model genuinely improved its own answers —
that's the two loops working.

This is deliberately tiny: the corpus is 30 Q&A pairs and the model is small enough to
memorize, so the interesting signal is whether self-distillation and REINFORCE can push
the reward curve upward on their own.
