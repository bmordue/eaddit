## 2026-05-20 - [str.translate for C-speed string sanitization]
**Learning:** For replacing multiple non-printable characters (like newlines, tabs, and carriage returns), using 'str.translate' with a precomputed translation table is significantly faster (approx. 10x) than a Python-level loop or generator expression. This is because 'str.translate' is implemented in C and can process the entire string in a single pass.
**Action:** Use 'str.translate' with a pre-calculated mapping table for high-volume character replacement tasks.

## 2026-05-20 - [List comprehension vs Generator in "".join()]
**Learning:** In pure Python, `"".join([c for c in s])` (list comprehension) is roughly 25% faster than `"".join(c for c in s)` (generator expression) when the list is immediately consumed by `join`. This is likely because the list comprehension can pre-allocate or more efficiently manage the intermediate collection before the C-level join operation begins.
**Action:** Prefer list comprehensions over generator expressions when passing them directly to `"".join()` in performance-critical paths.

## 2026-05-20 - [Instance-level vs Batch-local memoization]
**Learning:** For deterministic operations like HashingEmbedder, using a persistent instance-level memoization dictionary is significantly faster than a batch-local one, especially when the same texts are embedded across multiple calls. In benchmarks, this provided a ~10x speedup for subsequent calls with the same data.
**Action:** Use instance-level caching for deterministic, expensive operations that may be called repeatedly with the same inputs.

## 2026-05-20 - [Redundant Normalization in Vector Store]
**Learning:** In 'InMemoryVectorStore.add', calling 'list(map(float, vec))' followed by 'math.hypot(*v_list)' and then a list comprehension for normalization results in up to three (N)$ passes and multiple intermediate list allocations. By checking the norm directly on the input vector and only converting/normalizing if necessary, we can achieve a ~70% speedup for the common case where vectors are already normalized.
**Action:** Always check if a vector is already normalized using 'math.isclose(math.hypot(*vec), 1.0)' before performing expensive re-normalization or type conversion.

## 2024-05-21 - [Sparse accumulation for HashingEmbedder]
**Learning:** For high-dimensional feature hashing (e.g., dim=4096+), accumulating token counts into a dense list of size 'dim' and then iterating over it for normalization is inefficient ((dim)$). Using a sparse dictionary for accumulation and only populating non-zero buckets in the final dense vector reduces the complexity to (unique\_tokens)$ for most steps, providing up to a 16x speedup for typical short-ish texts.
**Action:** Use sparse data structures for intermediate accumulation in high-dimensional vector operations when the expected density is low.
