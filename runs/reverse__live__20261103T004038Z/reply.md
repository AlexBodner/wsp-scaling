Layer 0 retrieves the arrow’s position \(A\). Layer 1 selects position \(2A-j-1\), the next letter to emit, and amplifies its encoding above the current token’s residual. Paired coordinates keep means zero; large constant coordinates stabilize normalization. All other layers remain identities.

```edits
# Letter embeddings: one positive/negative pair per letter.
E[36573, 0:2] = [1, -1]
E[8109, 2:4] = [1, -1]
E[1423, 4:6] = [1, -1]
E[36388, 6:8] = [1, -1]
E[33820, 8:10] = [1, -1]
E[797, 10:12] = [1, -1]
E[27063, 12:14] = [1, -1]
E[8357, 14:16] = [1, -1]
E[5703, 16:18] = [1, -1]
E[4100, 18:20] = [1, -1]
E[23631, 20:22] = [1, -1]
E[23710, 22:24] = [1, -1]
E[48473, 24:26] = [1, -1]
E[44179, 26:28] = [1, -1]
E[41397, 28:30] = [1, -1]
E[27569, 30:32] = [1, -1]
E[656, 32:34] = [1, -1]
E[38062, 34:36] = [1, -1]
E[44429, 36:38] = [1, -1]
E[14491, 38:40] = [1, -1]
E[19028, 40:42] = [1, -1]
E[3884, 42:44] = [1, -1]
E[48281, 44:46] = [1, -1]
E[24929, 46:48] = [1, -1]
E[31122, 48:50] = [1, -1]
E[30069, 50:52] = [1, -1]

# Arrow marker.
E[31425, 64:66] = [1, -1]

# Position, squared position, and normalization anchors.
P[0:14, 60] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
P[0:14, 61] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13]
P[0:14, 62] = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169]
P[0:14, 63] = [0, -1, -4, -9, -16, -25, -36, -49, -64, -81, -100, -121, -144, -169]
P[:, 760] = 10000
P[:, 761] = -10000

# Layer 0: attend to the arrow and copy its position.
L0.norm1.scale[60] = 510.310363
L0.norm1.scale[64] = 510.310363
L0.qkv.b[0] = 100
L0.qkv.W[64, 768] = 100
L0.qkv.W[60, 1536] = 1
L0.out.W[0, 66:68] = [1, -1]

# Layer 1: scores approximate 100 * (2 * target * i - i squared).
L1.norm1.scale[0:52:2] = 510.310363
L1.norm1.scale[60] = 510.310363
L1.norm1.scale[62] = 510.310363
L1.norm1.scale[66] = 510.310363
L1.qkv.W[66, 0] = 3200
L1.qkv.W[60, 0] = -1600
L1.qkv.b[0] = -1600
L1.qkv.b[1] = -800
L1.qkv.W[60, 768] = 1
L1.qkv.W[62, 769] = 1

# Read each letter's positive coordinate into the first head.
L1.qkv.W[0, 1536] = 1
L1.qkv.W[2, 1537] = 1
L1.qkv.W[4, 1538] = 1
L1.qkv.W[6, 1539] = 1
L1.qkv.W[8, 1540] = 1
L1.qkv.W[10, 1541] = 1
L1.qkv.W[12, 1542] = 1
L1.qkv.W[14, 1543] = 1
L1.qkv.W[16, 1544] = 1
L1.qkv.W[18, 1545] = 1
L1.qkv.W[20, 1546] = 1
L1.qkv.W[22, 1547] = 1
L1.qkv.W[24, 1548] = 1
L1.qkv.W[26, 1549] = 1
L1.qkv.W[28, 1550] = 1
L1.qkv.W[30, 1551] = 1
L1.qkv.W[32, 1552] = 1
L1.qkv.W[34, 1553] = 1
L1.qkv.W[36, 1554] = 1
L1.qkv.W[38, 1555] = 1
L1.qkv.W[40, 1556] = 1
L1.qkv.W[42, 1557] = 1
L1.qkv.W[44, 1558] = 1
L1.qkv.W[46, 1559] = 1
L1.qkv.W[48, 1560] = 1
L1.qkv.W[50, 1561] = 1

# Amplify the selected letter and restore its balanced pair.
L1.out.W[0, 0:2] = [100, -100]
L1.out.W[1, 2:4] = [100, -100]
L1.out.W[2, 4:6] = [100, -100]
L1.out.W[3, 6:8] = [100, -100]
L1.out.W[4, 8:10] = [100, -100]
L1.out.W[5, 10:12] = [100, -100]
L1.out.W[6, 12:14] = [100, -100]
L1.out.W[7, 14:16] = [100, -100]
L1.out.W[8, 16:18] = [100, -100]
L1.out.W[9, 18:20] = [100, -100]
L1.out.W[10, 20:22] = [100, -100]
L1.out.W[11, 22:24] = [100, -100]
L1.out.W[12, 24:26] = [100, -100]
L1.out.W[13, 26:28] = [100, -100]
L1.out.W[14, 28:30] = [100, -100]
L1.out.W[15, 30:32] = [100, -100]
L1.out.W[16, 32:34] = [100, -100]
L1.out.W[17, 34:36] = [100, -100]
L1.out.W[18, 36:38] = [100, -100]
L1.out.W[19, 38:40] = [100, -100]
L1.out.W[20, 40:42] = [100, -100]
L1.out.W[21, 42:44] = [100, -100]
L1.out.W[22, 44:46] = [100, -100]
L1.out.W[23, 46:48] = [100, -100]
L1.out.W[24, 48:50] = [100, -100]
L1.out.W[25, 50:52] = [100, -100]

# Expose only letter coordinates to the tied output embeddings.
F.scale[0:52] = 1
```