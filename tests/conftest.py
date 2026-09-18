"""
Shared pytest setup: makes src/ importable as plain modules (e.g.
`import feature_engineering`) without needing a package/__init__.py
structure, matching how the project's own scripts are invoked
(`python src/some_script.py`).
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))
