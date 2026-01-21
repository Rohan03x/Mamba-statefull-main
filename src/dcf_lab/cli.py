"""Legacy CLI wrapper forwarding to :mod:`dcf_lab.app`."""
from __future__ import annotations

from .app import main

if __name__ == "__main__":  # pragma: no cover
    main()
