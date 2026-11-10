import sys
import pytest
from pathlib import Path

# Make examples/rmc importable as plain modules: `import problems`
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples' / 'rmc'))

@pytest.fixture
def run_in_tmpdir(monkeypatch, tmp_path):
    """Run each test from its own temporary working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path