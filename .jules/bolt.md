## 2026-05-07 - [Type safety in vector store optimizations]
**Learning:** In Python-based vector operations, while `list(vec)` is faster than list comprehensions, it can bypass critical type casting (like `float(x)`) required for JSON serialization or compatibility with the `math` module when the input comes from libraries like NumPy or PyTorch. Using `list(map(float, vec))` provides a significant speedup over list comprehensions while maintaining type safety.
**Action:** Always ensure explicit type casting to standard Python primitives when optimizing data structures that will be serialized or processed by the standard library.

## 2026-05-08 - [Reciprocal multiplication for vector normalization]
**Learning:** In Python's numeric list comprehensions (e.g., `[x / norm for x in vec]`), replacing repeated division with multiplication by the reciprocal (e.g., `inv = 1.0 / norm; [x * inv for x in vec]`) provides a ~15% performance improvement. This is a simple but effective win in hot paths like embedding generation or vector store normalization. Similarly, in complex formulas like `cosine_similarity`, algebraically reducing the number of divisions can yield measurable speedups.
**Action:** Prefer reciprocal multiplication and division-minimized formulas in hot numeric loops.
