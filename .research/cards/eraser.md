# Do negative value-output weights mark heads that push down what they attend to?

Question: Q010. Is there a kind of head that erases what it attends to, found from negative VO eigenvalues?
Rests on: checks/outputs/signed_ladder.txt: 10.7 -1.24 and 11.10 -0.72 on repeats, -0.86 and -0.31 on sentences, are the strongest negative direct effects. runs/kind-duplicate turn 3 and the open run's "match and subtract" group point to negative VO as a signature, though a verifier found that group mixed.
Mechanism: a head whose VO has mostly negative eigenvalues writes minus what it reads, so it lowers the attended token's logit when the embedding roughly passes through. Copy suppression heads are known to work this way, from memory of the published work and not measured here.
Prediction: 10.7 and 11.10 are in the top 10 of 144 by negative trace(VO) share. Spearman between negative VO share and measured direct effect on the attended token is at least 0.3 in magnitude on all heads, and the top 10 by the statistic have mean direct effect below 0, while matched positive-VO heads from the same layers don't.
Control: within-layer shuffle for chance, and same-layer positive-VO heads. The direct effect is measured on the token the head attends to most, on repeats and on real text from data/tinystories/valid.txt.
If it comes out as predicted: there's a readable weight signature for negative heads that Astra's open run half found, so a kind can be added to what blocks can classify.
If it comes out otherwise: VO sign doesn't classify, and Astra's label "match and subtract" was an algebraic story with no behavior behind it.
Output: checks/eraser.py writing checks/outputs/eraser.txt
Cost: free, under 15 minutes, under 3 GB.
