from __future__ import annotations

import sys

from easy_rtdetr.cli import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "eval", *sys.argv[1:]]
    main()
