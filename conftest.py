"""
Project-root conftest for pytest.

Tells pytest to skip the project root __init__.py during collection (it has
relative imports that don't resolve when the project isn't installed as a
package) and ensures the repo root is on sys.path so test modules can do
`from decision import ...` directly.
"""
import sys
from pathlib import Path

# Make sibling modules (decision.py, retrievers.py, ...) importable
sys.path.insert(0, str(Path(__file__).parent))

# Skip the broken root-level __init__.py during pytest collection
collect_ignore = ["__init__.py"]
collect_ignore_glob = ["__init__.py"]
