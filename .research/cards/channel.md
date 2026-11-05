# Is 4.11's output subspace the channel behind the five heads' pattern?

Question: Q008. Is 4.11's output subspace really the channel behind the five heads' pattern, or is the cosine just a correlate?
Rests on: checks/outputs/how_statistic_classifies.txt. Cosine with 4.11 finds 5 of 5. Silencing 4.11 alone moves their attention 0.870 to 0.813, against a random lowest of 0.854. Top 8 previous-token heads through keys move it to 0.524. Token self-match through raw embeddings is only 0.020.
Mechanism: 4.11 writes into directions that several previous-token heads also write into, and the five's keys read those directions. If the shared directions are the channel, removing them from the keys should collapse the pattern even though removing 4.11 alone doesn't, and matching tokens through a better estimate of the residual should improve a lot.
Prediction: projecting the top 64 right singular directions of 4.11's VO out of the five's key inputs drops their induction attention to under 0.5. Same-rank random subspaces leave it above 0.8. Token self-match through E passed through block 0's MLP rises above 0.1.
Control: 8 same-rank random subspaces, and the same projection on queries. Both are fixed now.
If it comes out as predicted: the cosine is reading a real shared channel, the mechanism is what we say, and the channel can be edited directly, which makes a read-then-write test fair.
If it comes out otherwise: if it stays near 0.81, the cosine is a correlate, and the post has to say it predicts these heads without showing how they work.
Output: checks/channel.py writing checks/outputs/channel.txt
Cost: free, under 20 minutes, about 2 GB of memory.
