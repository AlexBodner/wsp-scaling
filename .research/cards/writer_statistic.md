# Does the writer side of the statistic find previous-token heads from blocks alone?

Question: Q006. Can the writer-side statistic, without any tables, find previous-token heads beyond 4.11, on GPT-2 and other models?
Rests on: checks/outputs/head_types.txt after a fix: best cosine as writer for a later head finds 1 of the top 5 previous-token heads, and ranks 4.11 first. checks/outputs/other_models.txt names medium's top previous-token heads 5.11, 4.13, 6.7, 8.1, 13.5 and distilgpt2's 2.11, 1.2, 1.9, 3.0, 2.3.
Mechanism: previous-token heads matter because later heads read what they write, so strong writers for many later heads should be previous-token heads even though position isn't visible in the blocks.
Prediction: summing each head's cosine over later readers does better than taking its best: at least 2 of 5 of the top previous-token heads on gpt2 and medium, with the single top previous-token head within the top 10 by this statistic on all three models.
Control: a within-layer shuffle of each head's statistic for chance, 2000 draws, and the same statistic with the reader and writer roles swapped.
If it comes out as predicted: previous-token heads have a readable signature from blocks through what later heads do with them, which a post can state, and Astra could be tested on it.
If it comes out otherwise: position-defined heads stay unreadable from blocks by this route, so the table limit stands.
Output: checks/writer_statistic.py writing checks/outputs/writer_statistic.txt
Cost: free, under 15 minutes, under 3 GB.
