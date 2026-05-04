## 2025-05-15 - [Optimize vector search in InMemoryVectorStore]
**Learning:** In pure Python (without NumPy), `sum(map(operator.mul, v1, v2))` is significantly faster than a manual `for` loop with `zip` for dot product calculations. Additionally, caching vector norms and using `heapq.nlargest` instead of sorting the entire list of scores provides a substantial speedup for vector search operations, especially as the number of vectors grows.
**Action:** Use pre-calculated norms, `operator.mul`, and `heapq` for in-process vector search implementations in similar Python-based RAG applications.

## 2025-05-16 - [Extreme speedup for L2-normalized vector search]
**Learning:** For L2-normalized vectors, cosine similarity (dot product) can be computed much faster using `1 - math.dist(a, b)**2 / 2`. `math.dist` and `math.hypot` are implemented in C and provide a ~3x speedup over `sum(map(operator.mul, a, b))` and a ~6x speedup over manual generator expressions in pure Python.
**Action:** Always prefer `math.dist` for similarity searches and `math.hypot` for normalization when working with pure Python and normalized vectors.

## 2025-05-17 - [Token deduplication for faster feature hashing]
**Learning:** In feature-hashing embedders (like `HashingEmbedder`), counting token frequencies within a document *before* performing hashing operations is significantly faster for repetitive text. This reduces the number of calls to expensive cryptographic hash functions (like `blake2b`) and replaces them with simple scalar multiplications.
**Action:** Always deduplicate/count tokens before applying per-token operations in text processing pipelines.
