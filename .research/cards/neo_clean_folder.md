# Do the Neo edits hold when the run's folder name doesn't say Neo?

Question: a code review found Codex shows the model its working directory, so runs write-neo-1, write-neo-2 and neo-blind had "neo" in text Astra could read, though the prompt never names the model. C020 was therefore not as blind as written. Does the result depend on it?
Change: only the call folder, now a meaningless path outside the project. Same prompts, frozen hashes, seeds 1501 and 1502, same cap, same scoring.
Arms: A. one call with no weights. B. edit runs 1 and 2.
Prediction, fixed before any call:
1. A's guess stays unspecific: at most 1 of Neo's top 5 induction heads, and it doesn't favour layers 6 and 8.
2. B edits previous-token heads again, at least 3 of 4.9, 5.0, 5.6, 5.8, 7.1 in each run, because the folder name didn't drive the method. Low confidence.
3. B cuts repeat accuracy to 0.5 or less in both runs.
Reading: C020's result stays published either way, with this disclosure. This isn't a rerun for a better result: both sets are reported whatever they show. If 2 and 3 hold, the folder name doesn't explain the Neo result and the thread stands. If they fail, post 4 and the Neo section change to say so. Neither outcome settles memory of the method.
Stop rule: these three runs once. Nothing further is rerun.
Output: scripts/run_write_neo.py, runs/, checks/outputs/neo_edit_check.txt, with the old output kept as checks/outputs/neo_edit_check_named_folder.txt.
