# How Astra found the induction heads

With Python and no hint, GPT-6 Astra named all five of GPT-2's induction heads in both runs. This is what the logs show about how, and what they can't show. Its reasoning is encrypted, so we have its replies, code, outputs and token counts, not its chain of thought.

![Its words, its key lines and what they printed](figures/how_astra_found_them_v1.png)

## What it did, step by step
1. **It stated a theory before reading a weight.** Run 2, turn 1: "an earlier head can write the preceding token's features into a position, and a later head can match those features against the current token." Run 1 opened with the same comparison. This took 1,291 and 972 reasoning tokens.
2. **It scored every pair of heads.** For each head, `A = Q @ K.T` is what it searches for and `B = V @ O` is what it writes. The dot product of a later head's A with an earlier head's B says how well they line up. It also noted that these products don't depend on a head's internal coordinates, which is why the scrambling doesn't stop it.
3. **One earlier head stood out.** It is 4.9 in both runs' scrambled labels, which is GPT-2's head 4.11.
4. **Five later heads came out on top, and it named them.** In run 1 they top its first ranking by alignment. In run 2 they top its second, which pairs each head with 4.9.
5. **It spent the remaining turns checking.** It tried other weightings of the same comparison, and did not change its picks.

Total reasoning was 6,542 and 5,759 tokens. The no-tools runs used about 42,000 and 51,000, and failed.

## Did it cheat?
We looked for it and found none. An independent audit read every code block and session file:

| Check | Result |
|---|---|
| Tools in the session files | None: only user messages, reasoning and replies, on all 9 calls |
| Hidden context | Billed input matches the visible text plus 20 tokens on all 9 calls, with Codex memory off and empty |
| What its code touches | Only numpy, torch and threadpoolctl, with no files, network or other programs |
| Outputs | No paths, seeds or truth labels |
| Prompts | No answer, seed or model name. The working folder's path shows the project's name |
| Carry-over | Unique sessions, each extending only its own transcript, and no hidden reruns |
| Forward passes | None. It does multiply in the layer norm's scale and remove a mean, with no input. Results are the same without that |
| Score | Recomputed by matching each head to real GPT-2 without using our scramble record. Same result |

## What this does not show
- **It looks like recall, not discovery.** It wrote the published circuit first, and a fixed script of that statistic gets 5 of 5 on five scrambles (`checks/composition_statistic.py`). Astra's contribution is choosing that statistic.
- **It may have recognized GPT-2.** The prompt gives its shapes and weight count. Its reasoning is encrypted, so we can't see this.
- **The two runs are closer to one measurement.** The scramble leaves `Q @ K.T` and `V @ O` unchanged, so both runs see the same numbers. They differ in the code Astra wrote, not the evidence.
- **The method tests the matching half of the circuit.** With no embeddings, there's no per-head copying score, and "previous token" comes from the published work.

## What these heads do
Zeroing the five heads cuts correct repeating of random tokens from 98.6% to 28.9%, and of word lists from 93% to 47%. The next five barely matter alone. On a name repeated in an ordinary sentence, silencing them **raises** the chance of the right surname. We don't know why, and the five heads should not be read as the whole story of in-context learning (`checks/induction_examples.py`).

To read it yourself: [run 1](runs/reading-python-1__live__20261104T014623Z/transcript.jsonl) and [run 2](runs/reading-python-2__live__20261104T015236Z/transcript.jsonl), one JSON line per turn.
