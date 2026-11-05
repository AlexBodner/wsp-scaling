# Can Astra swap one capability for a specific other one, blind?

Question: Alex: "it would be a cooler experiment if we say swap this capability for this other specific", and "like it changes this capability but remains other ones almost untouched (the ones that do not depend on it)".
Target: on repeated random tokens, the top next token becomes the current token instead of the next token of the repeat. Scrambled GPT-2 small, Python, no forward pass, no score seen. Prompt: prompts/swap.txt, scored by scripts/run_swap.py. Seeds 1601 and 1602.
Revised after the critic, before any call. The task now also asks the model to keep its top token elsewhere, including the first copy.
Baselines (checks/outputs/swap_baselines.txt): earlier edits score swap 0.032 and 0.004, so this target is new. A logit bias toward the current token stays under 0.03 on random tokens, so it can't be gamed cheaply. Best rewires: 4 heads without 5.1, swap 0.629, first-copy echo 0.667, rise +0.131, facts 7. Top 3, swap 0.198, first-copy echo 0.333, rise +0.064, facts 7. No baseline has swap above first-copy echo by more than 0.27.
Claim rule, fixed before any call: "swap" only for a run with swap at least 0.5, first-copy echo at most 0.1, lists current word at least 0.5, text rise at most 0.10, facts at least 13, and capitals, pronoun gender and successor at least 0.95. "Partial swap" only for swap at least 0.2, swap above first-copy echo by at least 0.1, text rise at most 0.10, and facts at least 10. Anything else is reported as failed, with numbers. An echo everywhere doesn't count, however high its swap score.
Prediction, fixed before any call:
1. No run reaches "swap". Telling repeats apart is harder than removing a behaviour.
2. At least one of two runs reaches "partial swap". Low confidence.
3. Old behaviour falls below 0.5 in both runs.
Nothing is rerun to get a better result, and a failure is reported as plainly as a success.
Spend: Alex: "its fine to use any ammount of astra tokens". Each run capped at $30 of list price, not billed per call.
Output: runs/swap-*, checks/outputs/swap_baselines.txt.
