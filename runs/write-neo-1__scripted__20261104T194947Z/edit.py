W["L5.out.W"][0:64, :] = 0

_original = np.load("blocks.npz")
_changed = {name: W[name].astype(np.float32) for name in W if not np.array_equal(W[name], _original[name])}
np.savez('edited.npz', **_changed)
print("changed", len(_changed), "tensors,", sum(int((_changed[name] != _original[name]).sum()) for name in _changed), "values")
