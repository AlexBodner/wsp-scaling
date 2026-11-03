# Reading and writing GPT-2's weights: method

Question: with no tools, can Astra write GPT-2 weights that do a specified task, predict what those weights will do, and read a real GPT-2's weights? Writing comes first. Reading starts at "Reading test".

"Optimal" is not tested, because nobody knows the optimal weights. Scores are compared with a simple baseline and with trained GPT-2.

## Decided
- Operation: write from all-zero weights, not edit a trained model.
- Model: GPT-2 small (124M weights).
- Writer: `gpt-6-astra` through sealed `codex exec`, one reply per run, one fresh call per run. The reading test differs, and its section says how.
- No Python, calculator, forward passes, training data, feedback or retries.
- Prompt describes the task only, with no strategy hints. Frozen before the first run. Every run is reported.
- Format: typed statements. Explicit values and ranges allowed. No `*=`, `+=` or `-=`.

## Experiments
1. Positive control (1 run): the post's Breakout prompt through our setup. If it fails, fix the harness before reading anything else.
2. Write (3 runs):
   - Capitals: can it program a lookup? Must beat chance.
   - Reverse a string: can it program attention? Must beat chance.
   - Next-token prediction on TinyStories: can it make a language model? Must beat token frequency counts.
3. Predict (3 runs, not run live, dry checks only): a fresh call gets a written network, or a small hand-built one if a write produced nothing useful, plus fixed prompts. It predicts the top next token for each, and we compare against running the network.

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

GPT-2 writes, one sealed Astra call each at `xhigh`:

| Task | Result | Reference | Trained GPT-2 | Output tokens | Nonzero weights |
|---|---|---|---|---|---|
| Capitals, visible | 98.0% (96.1 to 99.4) | 2.8% | 1.1% | 36,529 | 104,020 |
| Capitals, hidden | 53.6% (51.7 to 55.9) | 2.8% | 0.0% | | |
| Reverse | 100% | 1.4% | 0.8% | 5,642 | 2,326 |
| Language | Aborted by hand before any reply, no result | | | | |

Neither network looks copied. The capitals correlation of -0.078 sits outside the random range, which we attribute without testing to its fills of plus and minus 10,000.

**These results do not show that Astra understands LLM weights.** Both networks are small circuits inside an otherwise empty GPT-2 (`checks/container_test.py`):

| Reply | 12 layers (124M weights) | 2 layers (54M) | 1 layer (46M) |
|---|---|---|---|
| Capitals, visible | 98.04% | 98.04% | 98.04% |
| Reverse | 100% | 100% | Cannot run, it writes to layer 1 |

- Capitals uses 0.08% of the weights and never reads the country's position or the question. `The president of France is` gives ` Paris`, and `The population of Japan is` gives ` Tokyo`.
- Hidden phrasings split: `In {country}, the capital city is` scores 98.9% and `What is the capital of {country}? The answer is` scores 8.4%. A critic attributes the failure to the second `is` raising the gate's threshold, which we have not tested.
- Reverse scores 100% at lengths 1 to 6 and 0% from length 7, and it ignores the `rev:` command.
- What this shows: one-shot compilation of a small program into a real architecture, with no tools. Tracr compiles programs into transformer weights without any LLM understanding.
- A GPT-2 family size sweep would very likely be flat, because extra layers can stay empty, so it would not measure anything about size.

Known issues from the critic reviews:
- The prompt states exact GELU, while the model uses a close approximation, at most 0.0005 apart. Rescoring both replies with exact GELU changes neither score.
- Predict would leak the writer's comments, and on sparse networks it is close to uninformative.
- On language, about 14.0 bits is free, because the token list alone gives a uniform over listed tokens. 30 of 193,029 test tokens are not listed.
- A reply cut off before its closing fence would score as an all-zero network.
- One run per task. The intervals cover test-set sampling only, not variation between replies.

## Reading test, GPT-2 small only
Question: name the 5 heads, of 144, with the highest copying score, by reading the weights. A head scores high when, on seeing a token again, it attends to the token after the earlier copy. These are induction heads, described in [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/) (Anthropic, 2021). Our top five, 5.5, 6.9, 5.1, 7.10 and 7.2 in GPT-2's numbering, are the ones commonly listed for GPT-2 small, so a reader may know of them from papers. The disguise hides which head is which, and layer order stays.

- **Model:** GPT-2 small, disguised: heads reordered within each layer and re-expressed in random internal coordinates, and the key bias replaced by noise. Behaviour is unchanged, within 1e-8 on logits in the tests.
- **Reader:** Astra through a viewer that only looks up values, rounded to 3 significant digits. 20,000 values, at most 6 turns, no Python, no forward passes. Each turn is a fresh sealed call replaying the transcript, so private reasoning is not kept.
- **Truth:** measured by running real GPT-2 on 50 random tokens repeated twice, 96 sequences. A head counts if it is in the top 10.
- **Pass rule:** at least 6 of 10 hits over two reads on different disguises. At the memory ceiling of 25%, that happens by luck about 2% of the time.
- **Arms:** read twice, blind once. If it fails, one hinted read naming the statistic, then stop. A pass does not start a size sweep without a new decision.
- **Disclosed choices:** top 10 and skipping the first third of layers in the solver were chosen after looking at GPT-2 small, and they are frozen now.

Free checks so far:

| Reader, naming 5 heads | Precision |
|---|---|
| Chance | 7% |
| Best possible from memory of which layers | 25% |
| Query bias . key bias, the shortcut, before removing it | 80% |
| The same statistic after replacing the key bias with noise | 17% |
| Best other simple norm | 28% |
| Scripted reader through the viewer, 20,000 values, 30 disguises | 67% ± 15% |
| Same statistic on all rows, with no budget, layers 4 to 11 | 80% |
| Same statistic on all rows, all 12 layers | 0% |

20,000 values is about 93k input tokens.

The scripted reader's 67% depends on skipping layers 0 to 3, a rule chosen after looking at GPT-2 small. Without it the statistic scores 0%, because those layers dominate it. So 67% is what the arithmetic achieves given that prior, and the weights alone do not supply it. Within layers 4 to 11, chance is about 10%, since all 10 true heads sit there. The read arm does not get this rule. The hinted arm does.

Harness rules:
- Prompts over 600 KB make the next turn the last, because macOS caps arguments at about 1 MB. Over 900 KB the run stops before calling.
- The reader is told when a turn is its last. No answer block by then scores zero.
- A call that fails writes `ABORTED.md`, and no `result.json`, so it cannot be mistaken for a scored run.
- Rerun policy, set before any call: a run that crashed before an answer may be rerun once on the same seed, and this is reported. A scored run is never rerun, however bad.

What a pass or fail would mean:
- A pass shows Astra can locate one specific kind of head by hand from raw weights, not that it understands an LLM in general.
- A fail with a working hinted read shows it could do the arithmetic but did not find what to compute. A fail on both shows no hand reading at this budget, where a short script succeeds.

Reading test results, sealed Astra calls at `xhigh`:

| Arm | Hits of 5 | Budget used | Turns | List price |
|---|---|---|---|---|
| Blind | 1 | 0 | 1 | $0.05 |
| Read 1, seed 1001 | 0 | 19,998 of 20,000 | 6 | $5.70 |
| Read 2, seed 1002 | 0 | 20,000 of 20,000 | 6 | $5.50 |
| Hinted, seed 1003 | **4** | 19,968 of 20,000 | 6 | $6.80 |

All 18 read and hinted calls were sealed, with no tool items. Hits were recomputed from the saved answers and match (`checks/read_audit.py`).

Astra failed the pass rule, at 0 of 10, and the hinted read worked. So it could do the arithmetic once pointed there. We cannot tell which part of the hint mattered, so "did not find what to compute" goes further than these runs.

- **0 of 10 is not strong evidence by itself:** guessing gives it 48% of the time over all heads and 32% within layers 4 to 11.
- **What the score measures:** attention to the token after the earlier copy, not whether a head writes that token. So a head counts here by where it looks. Value-output alignment is the usual proxy for what a head writes, from general knowledge of the induction-head literature. We did not measure what the picked heads write.
- **Why both reads missed:** both also read some query and key weights (5,888 and 9,344 of their values, far more thinly than the hinted read's 19,968), but built their shortlists from value-output alignment, which finds 0 of 5 with every row on both disguises. The query-key statistic with every row, in layers 4 to 11, finds 4 of 5 on both. Across all 12 layers it finds 0 of 5, so the layer rule matters as much as the statistic.
- **Its unhinted picks:** the closest across both reads ranks 18th of 144. On read 2 the best is 57th and scores 0.011, against 0.46 for the 10th-best head.
- **The hinted picks:** 5.10, 7.9, 5.6 and 6.7 are hits, ranking 3, 5, 9 and 2 of 144. The miss, 8.2, ranks 13. These labels are on disguise 1003. In GPT-2's numbering they are 5.1, 7.2, 5.0, 6.9 and the miss 8.1, so the hits include 3 of the top five. The scores drop from 0.83 to 0.51 after fifth, so we also rescore on those five, after the fact: 0 of 10 alone, 3 of 5 hinted, and a mean of 2.6 for the script (`checks/top_five.py`). On these five, knowing the layers gets 17%, and guessing gets 0 of 10 about 58 to 70% of the time. The blind call, shown no weights, named layers 5, 5, 6, 7, 7, exactly where they sit, so Astra may know those layers from papers. Its one blind hit is luck, because the scramble hides which head is which. The other five in the top 10 are probably not induction heads, from memory of the IOI work and not checked here. Guessing gives at least 4 of 5 about 3e-4 of the time within layers 4 to 11.
- **Its picks follow from its numbers:** the same statistic on the same rows, in code, gives the same five heads (`checks/hinted_arithmetic.py`).
- **Its typed arithmetic:** across its 96 stated dot products on row 137, correlation with code is 0.998 and the median error is 0.003, which is near the rounding floor of the two decimals it typed. 86 of 96 match code to two decimals, 99% have the right sign, and the largest error is 0.103. Its three-row means for the top five match code to 0.01.
- **Reported token use:** the 48 dot products of layers 4 to 7 are 3,072 products, in one uninterrupted request reporting 13,352 reasoning tokens, or 4.3 per product.

Limits:
- **The hint gives three things:** the statistic, the fact that it predicts copying, and the rule to skip layers 0 to 3. We wrote it after looking at GPT-2 small, and this run cannot separate them.
- **Reasoning is hidden:** we see token counts and final tables, not the working, and "by hand" goes beyond what we can observe. No tool items appear in our logs, but we cannot observe the provider's side. Astra itself labels its tables approximate.
- **Two calls were interrupted:** hinted turn 3 and read 2 turn 4 share a gap of about 69 minutes from 15:00, so the cause is probably on our machine, likely the Mac sleeping, though we did not confirm that. Their rollouts span 81.9 and 74.0 minutes, while `latency_s` shows 18.1 and 10.2, and those turns report far more cached input (66,688 and 77,696, against 3,968 on most others). We infer that usage covers only the request after the gap, so those two token counts are too low. The prices above are lower bounds, and the replies and seals are unaffected.
- **The hinted read is one run, on one model, on one kind of head.** The scripted reader's 67% ± 15% gives the comparison for 4 of 5. It says nothing about understanding an LLM in general.
- **Deviation from plan:** read 2 was optional once the pass rule became unreachable, and both ran at once rather than in order. Neither result was seen before the other was started.

## Open
- What to run next on writing. A size sweep needs a design where size matters.
- Constraint: use bigger, more relevant ready-made models, and do not train networks ourselves. Astra sets or reads the weights.
- Size axis, proposed for later: the same budget on every model, so the visible share falls as models grow.

## After a positive result
Not done: no repeats, Qwen or size sweep has run.

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
