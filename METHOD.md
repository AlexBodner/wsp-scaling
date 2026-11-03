# Writing LLM weights: method

Question: with no tools, can Astra write GPT-2 weights that do a specified task, and can it predict what those weights will do?

"Optimal" is not tested, because nobody knows the optimal weights. Scores are compared with a simple baseline and with trained GPT-2.

## Decided
- Operation: write from all-zero weights, not edit a trained model.
- Model: GPT-2 small (124M weights).
- Writer: `gpt-6-astra` through sealed `codex exec`, one reply per run, one fresh call per run.
- No Python, calculator, forward passes, training data, feedback or retries.
- Prompt describes the task only, with no strategy hints. Frozen before the first run. Every run is reported.
- Format: typed statements. Explicit values and ranges allowed. No `*=`, `+=` or `-=`.

## Experiments
1. Positive control (1 run): the post's Breakout prompt through our setup. If it fails, fix the harness before reading anything else.
2. Write (3 runs):
   - Capitals: can it program a lookup? Must beat chance.
   - Reverse a string: can it program attention? Must beat chance.
   - Next-token prediction on TinyStories: can it make a language model? Must beat token frequency counts.
3. Predict (3 runs): a fresh call gets a written network, or a small hand-built one if a write produced nothing useful, plus fixed prompts. It predicts the top next token for each, and we compare against running the network.

Prompts get only the tokens each task needs, in ID order with no frequencies.

## Guards against recalled GPT-2 knowledge
- Token IDs are shuffled with a fixed random order, and the new IDs are listed in the prompt. Real GPT-2 is renumbered the same way for scoring.
- The prompt gives the architecture and tensor shapes, without naming GPT-2.
- After each run, the typed values are compared with real GPT-2 at the same positions, with the shuffle undone.

General knowledge from published GPT-2 interpretability work cannot be ruled out. Qwen3.5-0.8B is the check for that after a positive result.

## Pass rule
A task works only if the 95% confidence interval sits fully above its trivial reference. Trained GPT-2's score is reported for scale, and Astra does not have to match it.

## Task specs
Shared: raw text with no chat template, always the top token, effort `xhigh`. Only the last edit block counts, an unreadable statement is skipped and reported, and values are capped at 1e6. Prompts, test sets and this rule are frozen before the first call.

- Capitals. About 190 countries, leaving out disputed or multiple capitals. Score: the first token of the capital is the top prediction. Must beat always giving the most common first token.
  - Visible phrasings, which decide pass or fail: `The capital of {country} is` and `{country}'s capital is`.
  - Hidden phrasings, reported separately: `In {country}, the capital city is` and `What is the capital of {country}? The answer is`.
  - Reported separately for capitals that are one GPT-2 token and capitals that are several.
  - The prompt names no country as an example.
- Reverse. `rev: a b c ->`. 1,000 random strings of 3 to 6 letters, fixed seed. Score: whole string exact, generating as many tokens as the input has letters. Must beat echoing the input or a constant answer.
- Next-token prediction. Held-out TinyStories validation text. Score: bits per token. Must beat token frequency counts from the training split. Beating the uniform floor of 15.6 bits but not the counts is reported as partial.
- Predict. 50 fixed prompts, half from the task and half unrelated. Score: matching the network's actual top token. Must beat always guessing the network's most common output. Hand-built fallback networks are frozen before any results.
- Positive control. Scored over 100 episodes. Passes if it clearly beats their best trained PPO score of 25.3. More episodes only if we need a close comparison with their 517.4, which used 1,000.

## Special handling
| What | Choice | Why |
|---|---|---|
| Capitals test set | 179 of 193 UN members. Left out: South Africa, Bolivia, Israel, Eswatini, Benin, Netherlands, Ivory Coast, Tanzania, Sri Lanka, Malaysia, Nauru, Switzerland, Burundi, Indonesia | Disputed, multiple or moving capitals |
| Names | ASCII only, such as `Asuncion` and `Bogota` | Keeps tokenization predictable |
| TinyStories | The `TinyStoriesV2-GPT4` files | Chosen on memory of it being the cleaner version, not checked |
| Token frequency counts | First 100 MB of the training split, add-one smoothing | The Mac has little free disk, and counts settle well before the full 2.2 GB |
| Language test | First 1,000 validation stories, cut to 512 tokens | Keeps scoring on the Mac practical |
| Language token list | About 16,500 tokens seen in that 100 MB, which is about 100k input tokens | The full table would double that |
| Failed calls | One attempt per run, no retry | Matches "no retries". A timeout is reported, not rerun silently |

## Dry run, no Astra calls
| Check | Result |
|---|---|
| Parser on the post's Breakout reply | Reproduces their published network exactly, 92 statements |
| Prompt equations against GPT-2 code, random weights | Match |
| Their Breakout reply through our scorer, 1,000 episodes | 502.2 ± 7.8, against their 517.4 ± 7.4 |
| All-zero network | 0% on capitals and reverse, 15.62 bits on language, fails all three |
| Hand-built capitals network, typed through the parser | 63.7% visible (56.7 to 70.4), 62.6% hidden, passes |
| Copy check on that network | Correlation -0.055, inside the random-position range (-0.064 to -0.047), no value within 0.001 |
| Predict, perfect and empty answers | 100% passes, 0% fails, against a reference of 40% |
| Offline seal check, codex-cli 0.153.4 | 11 of 11 pass |

Trained GPT-2, for scale:

| Task | Trained GPT-2 | Trivial reference | Beats it? |
|---|---|---|---|
| Capitals, visible | 1.1% | 2.8% | No |
| Capitals, hidden | 0.0% | 2.8% | No |
| Reverse | 0.8% | 1.4% | No |
| Language | 3.53 bits | 8.45 bits | Yes |

Trained GPT-2's top token after `The capital of {country} is` is ` the` for 169 of 179 countries. For France, ` Paris` comes 5th. The score is real: the task asks for the top token, and this model usually continues differently. So passing capitals or reverse does not mean matching a trained model's general ability, and beating trained GPT-2 on them is a low bar.

The hand-built network shows typed weights can do the capitals task in this architecture. It used code to generate about 49k tokens of statements and our own design, so it says nothing about Astra.

## Results
Positive control, one sealed Astra call at `xhigh`, their Breakout prompt word for word: **535.1 ± 24.4 over 100 episodes, which passes** against 25.3. Their run scored 517.4 ± 7.4 over 1,000. 139 of 139 statements applied, 2,921 nonzero weights of 20,486, 7,845 input and 24,556 output tokens, 14 minutes, seal intact. This says the setup works. It is not evidence about LLM weights.

## Open
- The six GPT-2 runs: three writes, then three Predict.

## After a positive result
- Repeat each working task 3 times.
- Qwen3.5-0.8B (the post's model) on the same tasks.
- Size sweep across the GPT-2 family, then modern models.

## Later studies
- Python tools: the same writing test, with Astra allowed to run code.
- Reading from weights: their weights-only reads worked at 7.16M weights (language) and failed at 268M (Gemma sycophancy). Hold one task fixed across sizes to find out whether size or the task explains that.
- Other LLMs, if Astra works.

## Dropped from the first stage
- Two-digit addition: GPT-2 and Qwen tokenize numbers differently, so it doesn't compare across models.
- News topic (AG News): adds little beyond the other three tasks.
