## 2025-05-15 - [Optimize vector search in InMemoryVectorStore]
**Learning:** In pure Python (without NumPy), `sum(map(operator.mul, v1, v2))` is significantly faster than a manual `for` loop with `zip` for dot product calculations. Additionally, caching vector norms and using `heapq.nlargest` instead of sorting the entire list of scores provides a substantial speedup for vector search operations, especially as the number of vectors grows.
**Action:** Use pre-calculated norms, `operator.mul`, and `heapq` for in-process vector search implementations in similar Python-based RAG applications.
