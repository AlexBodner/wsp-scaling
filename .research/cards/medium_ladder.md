# Is GPT-2 medium a ladder, with early heads feeding late ones?

Question: Q017. Is medium a ladder: do the statistic's early picks feed the late high-attention heads through their queries, which would explain why zeroing attention's top 5 barely matters?
Rests on: checks/outputs/zero_order.txt: best-pairing top 5 (6.1, 7.2, 5.8, 6.15, 9.9) cost 41.2 points, attention top 5 (9.9, 18.5, 11.1, 12.1, 7.2) cost 3.9. checks/outputs/signed_ladder.txt: on GPT-2 small, late heads' attention falls from 0.54 to 0.04 when the top five are silenced only for their queries.
Mechanism: early induction heads predict the next token and write it into the residual, and late heads use that as their query. So late heads have high attention scores and are downstream of the early ones. Removing them leaves others to carry the signal. Removing the early ones starves everything after.
Prediction: silencing the 5 picks only for later heads' queries cuts the mean attention of medium's late high-scoring heads (18.5, 20.12, 13.14, 19.15, 20.7, 11.1, 12.1) by at least half. Through their keys the fall is under a quarter. Random same-layer sets through queries move it less than 0.1. Attention's top 5, silenced the same way, cut later heads' attention much less than the picks do.
Control: 8 same-layer random sets through queries, and through keys. Fixed now.
If it comes out as predicted: bigger models have a depth of copying stages, the statistic finds the early stage that matters, and a head's attention score alone can mislead. That's a finding about models and about how to read them.
If it comes out otherwise: the 41 points don't come through later heads' queries, so look at direct effects, or whether these picks matter for some other reason.
Output: checks/medium_ladder.py writing checks/outputs/medium_ladder.txt
Cost: free, under 30 minutes, under 3.5 GB, one job at a time.
