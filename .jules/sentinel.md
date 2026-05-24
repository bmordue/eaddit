## 2025-05-17 - [Insecure Temporary File Creation]
**Vulnerability:** Use of predictable temporary file names (e.g., `filename.tmp`) during atomic write operations. This is vulnerable to symlink attacks where an attacker can pre-create a symlink at the predictable path to cause the application to overwrite arbitrary files.
**Learning:** Atomic writes implemented by manually appending a suffix to a file path are risky if the suffix is predictable and the directory is shared or writable by others.
**Prevention:** Use `tempfile.NamedTemporaryFile` with `delete=False` or similar secure methods that generate unpredictable filenames and ensure safe file creation (using `O_EXCL`).

## 2025-05-24 - [Unsanitized Metadata in RAG Prompts]
**Vulnerability:** Metadata fields like `author` or `subreddit` were used as-is from external sources (Reddit). Malicious actors could use newlines or carriage returns to perform prompt injection or log injection when these fields are included in LLM prompts or logged.
**Learning:** Attacker-controlled data in metadata can be just as dangerous as the main text if it is included in security-sensitive contexts like LLM prompts.
**Prevention:** Always sanitize metadata fields by truncating to a safe length and stripping non-printable characters/newlines before including them in prompts or logs.

## 2025-06-05 - [Vector Store ID Collision via Chunk Indexing]
**Vulnerability:** The chunking logic used a '#' delimiter to generate sub-IDs for long text (e.g., 'id#1'). An attacker could craft a root object ID that includes '#' to overwrite chunks of another object, or cause collisions between multiple root objects.
**Learning:** Using a simple character delimiter for derived IDs is dangerous if that character is not forbidden in the base ID. This can lead to subtle logic flaws where data is unintentionally overwritten in the vector store.
**Prevention:** Enforce strict character sets for IDs in the model layer and forbid any characters used as delimiters in downstream processing (like '#' for chunk indexing).

## 2025-06-15 - [Unbounded Resource Consumption in Embedding Cache]
**Vulnerability:** The `HashingEmbedder` used an unbounded dictionary to cache embeddings for all processed texts. In long-running applications or when processing large-scale unique datasets, this leads to continuous memory growth and eventual Denial of Service via OOM.
**Learning:** Caching mechanisms intended for performance must always be bounded, especially when handling untrusted or high-volume external data. Manual dictionary-based caches are prone to being forgotten during maintenance.
**Prevention:** Use standardized, size-limited caching decorators like `functools.lru_cache` with a conservative `maxsize` for all in-memory caching of variable-sized or high-cardinality data.

## 2025-07-10 - [Memory Exhaustion via Unbounded Token Cache]
**Vulnerability:** The `_HashCache` in `HashingEmbedder` used a dictionary to store token-to-bucket mappings for all processed tokens. Since the vocabulary is derived from untrusted Reddit content, an attacker could provide a large number of unique "tokens" (e.g., random strings) to cause continuous memory growth and eventual OOM.
**Learning:** Even low-level internal caches must be bounded if their keys are derived from external, untrusted input. Standard Python dictionaries used as caches are particularly dangerous as they lack a built-in size limit.
**Prevention:** Always enforce a hard limit on the number of entries in a cache. If the limit is reached, fall back to re-computing the value for new keys without storing them in the cache.
