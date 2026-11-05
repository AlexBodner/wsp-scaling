# Does the statistic find induction heads on other models and sizes?

Question: Q003. Does the cosine statistic work beyond GPT-2 small, on smaller models, bigger models, and ones without a published head list?
Rests on: checks/outputs/composition_statistic.txt: 5 of 5 on GPT-2 small, but all scrambles share the same products, so that's one model.
Mechanism: if QK^T against an earlier VO reflects a general circuit that transformers learn, it should work wherever induction heads exist. If it reflects something particular to GPT-2 small, it will decay.
Prediction: at least 3 of the top 5 on GPT-2 medium (355M, 24 layers, 16 heads) and distilgpt2 (82M, 6 layers, 12 heads). At most 2 of 5 on pythia-14m, if it has clear induction heads at all. Zeroing the top 5 by score cuts repeat accuracy by at least 30 points where the model repeats well.
Control: for each model, chance overlap computed from its number of heads, and the same statistic with the earlier and later roles swapped.
If it comes out as predicted: the statistic generalizes across the GPT-2 family, so a bigger Astra test is worth running, and a less famous model can separate memory from reasoning.
If it comes out otherwise: if it fails on medium, the 124M result doesn't scale and that's the headline. If it succeeds on a tiny model too, it says little about Astra.
Output: checks/other_models.py writing checks/outputs/other_models.txt
Cost: free, about 1.8 GB of downloads, about an hour on CPU.
