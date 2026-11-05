# Which other abilities are fair edit tasks for Astra?

Question: Alex: "we can try with other ones". An ability is a fair task only if a few heads carry it, or Astra fails however well it reads.
Rests on: checks/outputs/edit_controls.txt, where 5 heads carry repeating at almost no text cost when fixed at their mean output.
Mechanism: abilities with a known attention circuit should concentrate in few heads. Fact recall should sit mostly in MLPs, so no 5 heads carry it.
Method: real GPT-2 small. Rank heads by the fall from fixing each one alone at its mean output, on half the prompts. Score the top 5 together on the other half, against 8 random sets of 5 excluding them, with text loss on 64 windows of 48 tokens.
Prediction, fixed before running, as falls on held-out prompts in share of prompts where a right token beats a wrong one:
1. Indirect object ("When Mary and John went to the store, John gave a drink to"): top 5 falls at least 30 points.
2. Pronoun gender ("Mary went home because"): falls 10 to 40 points.
3. Greater-than years ("from the year 1732 to the year 17"): falls 10 to 40 points.
4. Successor (days, months, numbers): falls under 30 points.
5. Capitals ("The capital of France is"): falls under 15 points.
6. Random sets fall at most 10 points on average for every ability.
Rule, fixed now: an ability is worth an Astra run only if the top 5 fall at least 30 points on held-out prompts, random sets fall at most 10 on average, and text loss rises at most 0.03.
If it comes out as predicted: only the indirect object task qualifies. Its heads are published, so it doesn't test recall.
If it comes out otherwise: an ability with no famous head list that qualifies is the strongest next task, because memory helps less there.
Output: checks/other_abilities.py writing checks/outputs/other_abilities.txt
Cost: free, no Astra calls, about 3 GB of memory.
