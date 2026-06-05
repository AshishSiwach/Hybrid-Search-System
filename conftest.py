"""
Project-root conftest for pytest.

Ensures the repo root is on sys.path so test modules can do
`from querylens.<module> import ...` directly without depending on
the test runner's CWD.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
