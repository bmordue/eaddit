## 2026-05-07 - [Type safety in vector store optimizations]
**Learning:** In Python-based vector operations, while `list(vec)` is faster than list comprehensions, it can bypass critical type casting (like `float(x)`) required for JSON serialization or compatibility with the `math` module when the input comes from libraries like NumPy or PyTorch. Using `list(map(float, vec))` provides a significant speedup over list comprehensions while maintaining type safety.
**Action:** Always ensure explicit type casting to standard Python primitives when optimizing data structures that will be serialized or processed by the standard library.

## 2026-05-15 - [Manual dictionary construction for simple dataclasses]
**Learning:** Replacing `dataclasses.asdict(self)` with manual dictionary construction in simple, flat dataclasses provides a significant performance gain (approx. 7x speedup in this codebase). `asdict` is expensive because it uses recursion and introspection to handle nested structures, which is unnecessary for flat models like `Post` or `Comment`.
**Action:** Prefer manual dictionary construction in `.to_dict()` methods for hot-path flat dataclasses.

## 2026-05-08 - [Reciprocal multiplication for vector normalization]
**Learning:** In Python's numeric list comprehensions (e.g., `[x / norm for x in vec]`), replacing repeated division with multiplication by the reciprocal (e.g., `inv = 1.0 / norm; [x * inv for x in vec]`) provides a ~15% performance improvement. This is a simple but effective win in hot paths like embedding generation or vector store normalization. Similarly, in complex formulas like `cosine_similarity`, algebraically reducing the number of divisions can yield measurable speedups.
**Action:** Prefer reciprocal multiplication and division-minimized formulas in hot numeric loops.

## 2026-05-11 - [Early filtering in data ingestion]
**Learning:** When ingesting large datasets (like JSON fixtures in `JSONFixtureCollector`), filtering raw dictionaries by ID and score *before* instantiating complex dataclasses or models provides a massive performance boost (~35%). This avoids the overhead of dictionary-to-object mapping and subsequent garbage collection for items that will immediately be filtered out anyway.
**Action:** Implement early-filtering on raw data before object creation in collectors and pipelines handling large inputs.

## 2026-05-12 - [Slice before string operations on long texts]
**Learning:** When generating snippets from potentially long user-generated text, performing expensive $O(N)$ operations like `.replace("\n", " ")` on the full text before truncation is a significant bottleneck. Slicing the string to the desired limit *before* replacement reduces the workload to a constant small size, yielding over 80% performance improvement on large inputs (e.g., 100KB comments).
**Action:** Always slice long strings to the required snippet length before performing character replacements or complex regex operations.

## 2026-05-14 - [Fast-path for string sanitization]
**Learning:** The `_sanitize` helper, used frequently for metadata like IDs and authors, was a hidden bottleneck due to unconditional character-by-character iteration. Since most identifiers are already printable, using `s.isprintable()` as a fast-path check allows skipping the expensive loop entirely, resulting in a ~7.4x performance improvement for safe strings.
**Action:** Use `isprintable()` to bypass complex sanitization or replacement loops when the input is already compliant with safety requirements.

## 2026-05-18 - [Optimizing token hashing with copy and struct]
**Learning:** In hot loops involving cryptographic hashing (like 'HashingEmbedder'), using 'hashlib.blake2b().copy()' is significantly faster (~40% gain) than re-instantiating the hasher for every token. Additionally, 'struct.unpack_from' is nearly twice as fast as 'int.from_bytes' for converting hash digests to integers. Finally, implementing a cache using the '__missing__' protocol avoids repeated 'if token in cache' checks, further streamlining the hot path.
**Action:** Use hasher copying and 'struct.unpack_from' in performance-critical hashing logic. Leverage '__missing__' for high-performance caches.

## 2026-05-18 - [Safe deduplication in batch processing]
**Learning:** When implementing batch-level deduplication to save computation (e.g., in 'HashingEmbedder.embed'), returning the same mutable list object for identical inputs can cause unexpected side effects if the caller modifies the results in-place. To maintain a safe API while still benefiting from memoization, always return a fresh copy (e.g., 'list(memo[text])').
**Action:** Always return fresh copies of memoized mutable objects in public APIs to prevent shared state regressions.
