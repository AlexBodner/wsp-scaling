# On medium, what actually matters for repeating: top attention, or the statistic's picks?

Question: Q014. On medium, do the statistic's picks matter more for repeating than the top attention heads, once compared with heads from the same layers?
Rests on: checks/outputs/other_models.txt: zeroing medium's top 5 by attention costs 4.7 points of repeat accuracy, against 71.0 on GPT-2 small. A verifier's check on one batch, in its own scratch file, found best-pairing picks cost 41 points and strong-writer picks 10.
Mechanism: medium has more heads doing similar jobs, so attention to the right place doesn't mean a head is needed. Best-pairing picks include 5.8 and 6.15, which score low on attention, and may be early heads whose loss hurts for other reasons.
Prediction: best-pairing picks cost at least 30 points at 5 heads, but heads from the same layers chosen at random cost almost as much, within 10 points. So the effect is about the layers and not the picks.
Control: for each pick set, 8 random sets with the same number of heads from each layer, excluding the picks. Fixed now. Also 8 uniformly random sets.
If it comes out as predicted: there's no sign that the statistic finds what matters. Those layers are just fragile, and 2 of 5 stands as a miss.
If it comes out otherwise: if picks cost far more than same-layer sets, the statistic finds functionally important heads that attention misses at size, which is a new result worth a figure. Then test on distilgpt2 and use functional cost as a second score for Astra's picks on medium.
Output: checks/zero_order.py writing checks/outputs/zero_order.txt
Cost: free, under 30 minutes, about 3 GB of memory, one job at a time. Sequences: at least 64, several batches.
