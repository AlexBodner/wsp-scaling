# Can Astra sort a real model's heads into jobs on its own?

Question: Q004. Beyond induction heads, can GPT-6 Astra classify what heads do from the scrambled blocks, with no method given?
Rests on: runs/kind-duplicate__live__20261104T035747Z scored 4 of 5, the same as trace(QK^T) with one line of code, at $5.98 and 10 code runs. Each single kind costs about $6, so the night's budget allows one open run in place of the four other kinds in C004, which ran one kind and aborted a second when the disk filled.
Mechanism: an open question lets us see what Astra looks for when no kind is named, which is what Alex wants to see: "classify the function of the heads", and "from how it does it analysis maybe something super interesting comes out".
Prediction: it forms an early group and an induction-like group and places their top heads together. It places at least 3 of induction's top 5 in one group and at least 3 of duplicate's top 5 in one group. It doesn't put at least 3 of the name mover top 5 in one group labelled for that job.
Control: the scoring script checks/astra_kinds.py was written before the run and compares groups with where heads attend on repeated tokens and sentences. Chance comparison for grouping: the same grouping sizes with heads shuffled within layers, to be run in scoring. Fresh scramble seed 1106. Prompt frozen in prompts/HEAD_KINDS_MANIFEST.json. A forward pass voids it.
If it comes out as predicted: Astra can name two kinds of head from weights and not the sentence-defined ones, consistent with what simple statistics can do.
If it comes out otherwise: a group matching name movers or S-inhibition would be new, so read how it got there and rerun that statistic. If it can't group induction heads when not asked for them, then naming the target was doing much of the work in the earlier result.
Output: scripts/run_head_kinds.py open writing runs/kind-open__live__*/, scored in checks/outputs/astra_kinds.txt
Cost: 1 sealed run, about $6 at list price, capped at $5 plus its last turn, ChatGPT plan pays.
