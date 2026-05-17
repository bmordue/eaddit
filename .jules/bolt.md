## 2026-05-20 - [str.translate for C-speed string sanitization]
**Learning:** For replacing multiple non-printable characters (like newlines, tabs, and carriage returns), using 'str.translate' with a precomputed translation table is significantly faster (approx. 10x) than a Python-level loop or generator expression. This is because 'str.translate' is implemented in C and can process the entire string in a single pass.
**Action:** Use 'str.translate' with a pre-calculated mapping table for high-volume character replacement tasks.

## 2026-05-20 - [List comprehension vs Generator in "".join()]
**Learning:** In pure Python, `"".join([c for c in s])` (list comprehension) is roughly 25% faster than `"".join(c for c in s)` (generator expression) when the list is immediately consumed by `join`. This is likely because the list comprehension can pre-allocate or more efficiently manage the intermediate collection before the C-level join operation begins.
**Action:** Prefer list comprehensions over generator expressions when passing them directly to `"".join()` in performance-critical paths.
