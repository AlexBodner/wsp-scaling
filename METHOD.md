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
- Positive control. Passes if it clearly beats their best trained PPO score of 25.3.

## Open
- Dry run with an all-zero and a hand-built network, no Astra calls.

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
