# On medium, where do the 41 points go?

Question: Q021. On medium, where do the other 30 of the 41 points go: direct effects of the picks on the logits, or their effect on other mid-layer heads?
Rests on: checks/outputs/medium_ladder.txt and checks/outputs/zero_order.txt. Silencing the picks everywhere costs 41.2 points, through all later queries 10.3, and each pick alone at most 0.9.
Mechanism: the picks may write the answer directly, so removing them from only the logits' view would do most of the damage. Or their output could feed later heads' values rather than queries.
Prediction: silencing the picks only in the final residual, with all later layers seeing them, costs at least 20 points. Silencing them only for later heads' values costs under 10.
Control: 8 same-layer random sets through each route. Fixed now.
If it comes out as predicted: on medium these heads mainly write the answer directly, with many heads adding up, which explains why each alone costs so little.
If it comes out otherwise: the damage runs through later MLPs or several routes combined, so report only what's measured.
Output: checks/medium_routes.py writing checks/outputs/medium_routes.txt
Cost: free, under 30 minutes, under 3.5 GB, one job at a time.
