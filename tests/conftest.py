import os
from pathlib import Path

# DSPy creates its disk cache during import, before application code disables it.
os.environ.setdefault("DSPY_CACHEDIR", str(Path.cwd() / ".pytest-dspy-cache"))
