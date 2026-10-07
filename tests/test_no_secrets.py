"""Nothing that looks like a credential may live in the repository tree."""

import os
import re

from lamas.backends.openai_compat import OpenAICompatBackend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {".venv", ".git", "node_modules", "__pycache__", ".pytest_cache"}
KEY = re.compile(rb"sk-or-v1-[0-9a-f]{32,}|sk-(?:proj|ant)-[\w\-]{20,}|sk-[A-Za-z0-9]{32,}|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}")


def test_no_api_keys_in_the_tree():
    leaks = []
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for name in files:
            path = os.path.join(folder, name)
            with open(path, "rb") as f:
                if KEY.search(f.read()):
                    leaks.append(os.path.relpath(path, ROOT))
    assert not leaks, f"key-shaped strings in: {leaks}"


def test_backend_errors_never_carry_the_key():
    backend = OpenAICompatBackend(base_url="http://localhost:1/v1", api_key="sk-test-0123456789abcdef")
    text = backend._scrub("Incorrect API key provided: sk-proj-ab****************yz. Sent: sk-test-0123456789abcdef")
    assert "sk-" not in text and "Incorrect API key provided" in text
