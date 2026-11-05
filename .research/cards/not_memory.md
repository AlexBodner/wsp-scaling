# Is Astra reading these weights, or remembering GPT-2?

Question: Alex: "do the runs that would settle that it is not direct memorization". No run rules out general knowledge. These rule out specific kinds of memory.
Rests on: scrambled heads already rule out remembered head numbers. Shown no weights, Astra named layers 5, 5, 6, 7, 7 on GPT-2 small, which is likely recall, and a script with the known statistic also finds the heads.
Arms, all sealed, Python, no forward pass, no score seen:
A. distilgpt2, no weights, one call each on 2 seeds: guess the 5 heads. Tells what memory alone gives on this model.
B. distilgpt2, reading with Python, 2 runs, already built as arms distil-1 and distil-2 in scripts/run_family.py. Truth is the model's own top 5 by attention. A script reaches 4 of 5 by best pairing.
C. GPT-2 small with no-op layers inserted, so the shape, weight count and layer numbers no longer match GPT-2, reading with Python, 2 runs. Inserted layers are not blank: see the critic's answer before building.
Prediction, fixed before running:
1. A scores at most 1 of 5 on each seed, near chance within likely layers.
2. B finds at least 3 of 5 in both runs, and more than A.
3. C finds at least 4 of 5 in both runs, in the shifted layer numbers.
Reading of results: if B and C hold, remembered layers and shape don't explain it, and the thread can say so. If C holds and B fails, the statistic may be GPT-2 specific. If both fail, GPT-2's result leaned on memory and the thread says that. Neither rules out remembering the method. A later arm tests that, on an ability with no known recipe, chosen by checks/other_abilities.py.
Cap: $20 in all, stopping when reached, at list prices. Each reading run about $1.30 to $3.
Output: runs/, checks/not_memory.py writing checks/outputs/not_memory.txt.
