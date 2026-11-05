# Is finding the heads really the hard part of the edit?

Question: thread v28 says "finding them looks like the hard part", from a simple edit on Astra's heads matching Astra. Reviewers on PR 7 pointed out nothing applies that edit to the wrong heads, or to whole layers, which needs no head finding at all. Also, 53% and 74% on repeated stories have no floor, and nobody measured where the five heads attend after the edit.
Rests on: checks/outputs/edit_demo.txt, checks/outputs/edit_limits.txt, .research/verification/write/step4_controls.txt, where Astra's own projection on random heads in layers 5 to 7 stays at 0.98 and up.
Mechanism: the five are few and specific, so an edit on others does little. Whole layers contain the five, so they lose repeating too, but also change many heads text needs.
Prediction, fixed before running, on scramble 1302:
1. Simple edit on 8 random sets of 5 anywhere, and 8 random sets of 5 in layers 5 to 7: repeat at least 0.90 on every set, text rise at most 0.03.
2. Simple edit on the next 5: repeat at least 0.90.
3. Simple edit on all 36 heads of layers 5 to 7: repeat at most 0.05, text rise between 0.03 and 0.20.
4. Simple edit on all 144 heads: repeat at most 0.05, text rise at least 0.30.
5. Stories, same targets on the first pass with no repeat: unedited 30 to 45% right. Run 2's 53% on the second copy stays at least 5 points above its own first pass.
6. After Astra's edits and the simple edit, the top five's attention to the induction target falls below 0.10, and position 0 takes most of it.
Control: unedited model, zeroed keys of the top 5, and a mean-ablated top 5 and layers 5 to 7, which use a forward pass, so they are a bar and not something Astra could do.
If it comes out as predicted: finding the heads buys a measured text saving over whole layers, and the thread can state the number.
If it comes out otherwise: if the simple edit on all of layers 5 to 7 reaches repeat at most 0.05 at rise at most 0.03, knowing the layers is enough, which is recallable from the literature, and the thread drops "finding them is the hard part". If story first-pass accuracy is within 5 points of 53%, "copying isn't gone" is reworded.
Output: checks/edit_controls.py writing checks/outputs/edit_controls.txt
Cost: free, no Astra calls, about 3 GB of memory, one job at a time.
