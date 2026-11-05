# Is attention to the first position a "no match" state, or just a base rate?

Question: Q002. Why do the top induction heads put 0.91 to 0.99 of their attention on position 0 when nothing repeats?
Rests on: checks/outputs/head_types.txt: first position 5.1 0.99, 6.9 0.95, 7.10 0.93, 7.2 0.93, 5.5 0.91, but also 7.9 0.94, 5.7 0.94, 7.8 0.90.
Mechanism: a head whose query finds no match has to put its softmax somewhere. If these heads default to position 0 and read almost nothing from it, this is an off state, and being on or off could be what makes them useful as a signal to later heads.
Prediction: this is a base rate. At least 20 heads in layers 4 to 9 exceed 0.5 on position 0, and the five aren't special among them. On real text the five attend to position 0 less than on random tokens but still more than 0.3. Their direct effect on the logits from position 0 is under 0.1.
Control: the same measures on all 144 heads, on random tokens and on real text from data/tinystories/valid.txt, and with the first token swapped for another to check it isn't that token.
If it comes out as predicted: drop it as a finding about induction heads and note it as a general property of mid-layer heads.
If it comes out otherwise: if the five are specific, or attention to position 0 falls sharply when a match appears, then this is an on-off signal, which gives a new way to read induction heads and is worth testing through late heads' queries.
Output: checks/first_position.py writing checks/outputs/first_position.txt
Cost: free, under 20 minutes.
