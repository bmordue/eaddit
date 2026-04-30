"""Allow running the CLI as `python -m eaddit`."""

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
