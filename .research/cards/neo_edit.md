# Can Astra edit a model it can't have memorised the heads of?

Question: Alex: "do the runs that would settle that it is not direct memorization". Replaces arms B and C of not_memory.md: the critic showed distilgpt2 is GPT-2 layers 0, 2, 4, 7, 9, 11 with the same head numbers, confirmed by QK^T cosines of 0.79 to 0.93.
Model: EleutherAI/gpt-neo-125M, 12 layers, 12 heads, width 768, trained separately from GPT-2. Top induction heads 6.1, 6.6, 8.1, 8.0, 6.11, in layers 6 and 8, not 5 to 7. Fixed-script best pairing finds 4 of 5. We know of no published head list for it, which we can't prove.
Arms, sealed, same task and prompt shape as GPT-2 small's edit runs:
A. One call with no weights: name the heads. Shows what memory alone gives.
B. Edit runs 1 and 2, seeds 1501 and 1502, each capped at $6.
Prediction, fixed before any call:
1. A hits at most 1 of the top 5, and doesn't favour layers 6 and 8.
2. B names at least 3 of the top 5, and edits at least 3, in both runs.
3. B cuts repeat accuracy to 0.5 or less at a text rise of 0.05 or less, in both runs.
Reading: if 2 and 3 hold, remembered head positions don't explain the GPT-2 result. Remembering the method is still possible, as it is in any run. If 2 holds and 3 fails, it reads but its edit doesn't transfer. If 2 fails, the GPT-2 result leaned on memory of GPT-2, and the thread says so.
Stop rule: run A, then B1, read it, then B2. Stop at $20 of new spend. Nothing is rerun to get a better result.
Output: scripts/run_write_neo.py, runs/, checks/outputs/neo_baselines.txt.
