# Sentinel's Journal - Critical Learnings Only

## 2025-05-14 - Sentinel Initialized
Sentinel is active and scanning for vulnerabilities.

## 2025-05-14 - Secure Atomic Writes
**Vulnerability:** Predictable temporary filenames for persistence (e.g., `file.json.tmp`) were susceptible to symlink attacks and race conditions on shared filesystems.
**Learning:** Standard Python `open()` followed by `os.replace()` with a fixed `.tmp` extension is not secure in multi-user environments.
**Prevention:** Use `tempfile.mkstemp` to generate unpredictable filenames with restricted permissions (0600) and atomic `os.replace` to ensure data integrity and security.
