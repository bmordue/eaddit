## 2025-05-17 - [Insecure Temporary File Creation]
**Vulnerability:** Use of predictable temporary file names (e.g., `filename.tmp`) during atomic write operations. This is vulnerable to symlink attacks where an attacker can pre-create a symlink at the predictable path to cause the application to overwrite arbitrary files.
**Learning:** Atomic writes implemented by manually appending a suffix to a file path are risky if the suffix is predictable and the directory is shared or writable by others.
**Prevention:** Use `tempfile.NamedTemporaryFile` with `delete=False` or similar secure methods that generate unpredictable filenames and ensure safe file creation (using `O_EXCL`).
