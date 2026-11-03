# Context: what we read and what it shows

The study started from one post and one reply to it. Everything below comes from the post, its published logs, or checks saved in `checks/`.

## The post
Takehi, Teoh, Zhang and Khattab (MIT CSAIL), [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), 2 Nov 2026. [Announcement](https://x.com/rikiyatakehi/status/2117313742891327994), [logs](https://github.com/RikiyaT/lm_understand-logs).

They claim frontier LLMs can natively read and write neural network weights. "Astra" is OpenAI's `gpt-6-astra`, called through `codex exec`.

| Exp. | Weights | Task | Tools | Ran the model? | Result |
|---|---|---|---|---|---|
| 1 | 17k | Name the game a policy was trained on | None | No | 4 of 4 |
| 2 | 24k | Name the shape a diffusion model draws | None | No | 3 of 3 |
| 3 | 7.16M | Name a small LM's language, find copying heads | Python | No | All 3, both attempts |
| 4 | 172k | Fix a failing car policy | None | No | 100% success from 0% |
| 5 | 20k | Write MinAtar policies from zeros | None | No | Beat PPO at 100M steps |
| 6 | 1.48M | Write a Go network from zeros | Python | 0 to 50 passes | Beat equal-sized PPO at 50 passes |
| 7 | 268M | Find the sycophantic Gemma | Python | No | Failed, 4 attempts |
| 7 | 268M | Same, with attention patterns | Python | Yes | All 3 pairs |
| 8 | 752M | Make Qwen3.5-0.8B better at math | Python | Yes, 50 passes | MATH-500 7.9% to 48.1% |

MinAtar is a set of miniature Atari games on a 10x10 grid (Young & Tian, 2019). It is not part of our study.

Experiment 3 calls its targets copying heads (induction heads) and measures them by running each network on a random string repeated twice, the same kind of truth as our reading test.

## The objection
[Adam Karvonen](https://x.com/adam_karvonen/status/2117337487458938949): "It looks like Astra didn't natively read Qwen's weights?" He says it used code for standard interpretability, found a looping direction and removed it from the weights.

The logs support him (`exp8_math/runs/p50c/`):
- The first forward pass asked for activations.
- `final_report.md` takes the direction from matched productive and repetitive contexts at the residual outputs.
- The edit is `W = W0 - 6*outer(u, mask64*(u@W0))` on 64 columns of two MLP matrices. By our arithmetic, the factor of 6 reverses the direction, leaving -5 times the original component on those neurons.
- The prompt names "repeats itself and never finishes" as an example failure.

The improvement itself holds up: on 1,152 held-out practice problems, accuracy went from 10.6% to 60.4%, with 579 rescues and 5 regressions.

## What else the logs show
**The Qwen gains track finishing.** Every benchmark's gain comes with fewer answers hitting the token limit. On MMLU-Pro, the model loops least and cutoffs nearly vanish, yet it gains only 5.2 points. The post also reports no real gain on Gemma 3 270M, which rarely loops.

| Benchmark | Accuracy before, after | Cut at token limit before, after |
|---|---|---|
| GSM8K | 13.8%, 67.1% | 87.5%, 7.2% |
| MATH-500 | 7.9%, 48.1% | 93.0%, 33.1% |
| IFEval | 21.0%, 62.3% | 74.9%, 2.2% |
| GPQA Diamond | 22.7%, 36.6% | 59.5%, 2.4% |
| MMLU-Pro | 40.3%, 45.5% | 28.9%, 0.5% |

**At LLM scale, reading worked only once Astra could see the model running.** Gemma failed with Python and no running. This may be size or a subtle task: the two Gemma models differ by about 340 fine-tuning steps with a frozen embedding. Their data cannot separate the two.

**Nobody has written an LLM from scratch here.** Their no-tool writes are small MLPs. Their LLM result is an edit that used activations.

**Single runs and selection.** Each budget is one run, and the logs leave out trial runs of earlier prompt versions.

**Typing cost, from their MinAtar runs.** Breakout used 20,567 output tokens, of which 16,828 were reasoning. It set 815 nonzero weights of 20,486, about 4.4 tokens per value written. The zero-pass Go run used about 32k output tokens in 22 minutes.

**Token cost of reading.** About 4.2 input tokens per weight at 17k and 24k, and 2.1 at 172k. We did not check why the last is lower. With their 872k context setting, whole-network reading runs out at roughly 200k to 400k weights.

## Architectures checked
Computed from configs or listed by Hugging Face on 2 Nov 2026. All run with zero weights and give zero logits, tested on cut-down configs (`checks/`).

| Model | Weights | Layers, width | Listed downloads | Likes |
|---|---|---|---|---|
| DistilGPT-2 | 82M | 6, 768 | 1.5M | 666 |
| GPT-2 | 124M | 12, 768 | 29.3M | 3,912 |
| GPT-2 medium | 355M | 24, 1024 | 150k | 208 |
| GPT-2 large | 774M | 36, 1280 | 841k | 372 |
| GPT-2 XL | 1.56B | 48, 1600 | 79k | 397 |
| Qwen3.5-0.8B | 752M (text part) | 24, 1024 | 3.7M | 794 |
| Qwen3.8-27B | 26.9B (text part) | 64, 5120 | 7.9M | 19,008 |

Qwen3.8-27B and Qwen3.5-0.8B load as the same class, `Qwen3_5ForCausalLM`.

**Tokenizers differ on numbers.** `12+34=` is 4 tokens on GPT-2, which has a token for every number from 10 to 99, and 6 on Qwen, which works digit by digit. This is why addition was dropped. Spaced letters, `The capital of France is` and the news labels tokenize the same on both.

**A one-value token code.** A token can be set with one nonzero value, so width 768 gives 768 distinct tokens for 768 typed values. Normalization and the tied input and output table make this harder in practice. Bigger models may therefore be harder because of depth, not the token table. This is a hypothesis.

## Related work seen so far
From two X posts and a GitHub name search on 3 Nov 2026, so this is not a full review.

| Source | Writer | Setting | Varies | Shows |
|---|---|---|---|---|
| [Singh report](https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html) | Claude as a coding agent | Python, many attempts with score feedback, may change the architecture | Task, then shrinks the network | All 10 algorithmic tasks at 100% within 3 attempts. Its language results have an evaluation flaw, which the report says itself |
| [Krentsel chart](https://x.com/alexkrentsel/status/2117376683724660898) | GPT and Claude generations | Not stated in the post | Writer generation | A sharp jump in the latest models, about 18x to 22x PPO-500M. The task is not stated, and we read it as probably Breakout |
| [nihilistau/weight-space-claude](https://github.com/nihilistau/weight-space-claude) | Claude Opus 5.5 | Their prompts, no tools | Writer | Breakout 307, against Astra's 517 |

A wider search on 3 Nov 2026 (web, GitHub, Hugging Face, Hacker News) also found no size sweep. It could not cover X search, replies under the original post, Reddit or Bluesky. A [Hacker News comment](https://news.ycombinator.com/item?id=50342350) names relative model sizes as interesting future work.

Also found:
- [Lucas Beyer](https://x.com/giffmana/status/2117346161866514524) says the Qwen prompt hints heavily at looping, which matches what we read in the logs.
- [Sean Goedecke](https://seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/) argues a good human researcher could do these tasks, and shows the game-reading signal as a heatmap in three lines of Python.
- [ML-Intern-lab](https://huggingface.co/ML-Intern-lab/Qwen3.5-0.8B-weight-space-programming) re-evaluated the released Qwen edit: MATH-500 51.0% against the post's 48.1%, with answers cut at the limit 31.6% against 33.1%. This checks their numbers, not how the edit was found.
- [yikerman/llm-rw-weight-extended-test](https://github.com/yikerman/llm-rw-weight-extended-test) repeats Experiments 1 and 2 across several LLMs, so it tests reading only.

None of these holds a real architecture fixed and varies its size. Three points bear on our design:
- The Singh report says most programs replace LayerNorm with the identity. Ours has to keep it.
- That agent moved from writing mechanisms to fitting them once it could read data. Our no-tools rule and hidden test sets close this off.
- Its first word-reversal program was already correct, on a small custom network with Python. Reverse on GPT-2 with no tools tests whether that survives.

## Budget
A $100 Codex plan over two weeks. List prices for `gpt-6-astra` are $10 per million input tokens and $50 per million output. Our cost figures use list prices, and how they map to plan usage is unknown.
