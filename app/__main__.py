"""Allow ``python -m app`` as an entry point."""

import sys

from app.cli import main

sys.exit(main())
