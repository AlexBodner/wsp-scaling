# Does Astra's surgical edit replicate on a fresh scramble?

Question: Q012. Can Astra turn reading into writing: edit scrambled weights so repeating fails while real text survives, with no forward pass?
Rests on: runs/write-write-1__live__20261104T*/write_score.json, repeat accuracy 0.987 to 0.180 with text loss rising 0.013, from one run. The same prompt on a new scramble tests whether this was luck. Its method can't be copied, because each call is sealed and this is a fresh disguise, but the same products appear.
Mechanism: if the first run's method follows from reading the weights, Astra arrives at something like it again: cutting a source head's written directions out of later heads' keys.
Prediction: repeat accuracy under 0.5 with text loss rising under 0.1, by an edit on keys rather than deletion. Lower confidence than usual: this may vary a lot between runs.
Control: write_baselines.txt, with the same combining rule as the first card. Fresh scramble seed 1302, frozen prompt. A forward pass voids it.
If it comes out as predicted: the surgical edit is repeatable, so the headline stands on two runs with different code.
If it comes out otherwise: if it deletes heads or fails, the first result was one good run, and it gets reported as that.
Output: scripts/run_write.py writing runs/write-write-2__live__*/write_score.json
Cost: 1 sealed run, about $4 to $6 at list price, capped at $6 plus its last turn.
