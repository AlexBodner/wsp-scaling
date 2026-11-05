# Does the token-space score find negative heads on other models?

Question: Q022. Is the token-space score a reliable detector of negative heads on other models?
Rests on: .research/verification/last/verify_eraser.txt, off-card: E through block 0's MLP, V, O, the final scale and the unembedding, as a copy score, ranks 10.7 and 11.10 first and second most negative on GPT-2 small, Spearman with direct effect 0.795 on repeats and 0.434 on stories. Found after looking, so it needs a fresh test.
Mechanism: a head whose output on a token's own embedding lowers that token's logit suppresses what it attends to. Reading through block 0's MLP approximates how tokens look to later layers.
Prediction: on gpt2-medium and distilgpt2, the 3 most negative heads by this score are all among the 10 most negative by measured direct effect on the attended token on repeats, and Spearman with direct effect is at least 0.5 on repeats.
Control: within-layer shuffle for chance. The same score with E rows permuted relative to the unembedding.
If it comes out as predicted: with the token table, negative heads are readable across models, which answers part of Ravfogel's question: the boundary tables matter for this kind.
If it comes out otherwise: the GPT-2 small result was a lucky find among statistics, and it gets reported that way.
Output: checks/token_negative.py writing checks/outputs/token_negative.txt
Cost: free, under 20 minutes, under 3.5 GB, one job at a time.
