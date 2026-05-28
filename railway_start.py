from __future__ import annotations

import os
import sys

from app import main


if __name__ == "__main__":
    sys.argv = [
        "app.py",
        "--host",
        "0.0.0.0",
        "--port",
        os.environ.get("PORT", "8080"),
    ]
    main()
