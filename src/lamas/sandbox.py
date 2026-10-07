"""Run model-written Python in a child process with a time limit.

This isolates crashes and infinite loops. It is NOT a security boundary: run untrusted
workloads inside a container.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from typing import Optional, Tuple

_BANNED_IMPORTS = ("subprocess", "multiprocessing", "matplotlib", "seaborn", "plotly", "tkinter", "PyQt5", "shutil")

_RUNNER = r"""
import json, sys, traceback
src = open(sys.argv[1], encoding="utf-8").read()
call = sys.argv[2]
env = {"__name__": "__lamas__"}
try:
    exec(compile(src, "<solution>", "exec"), env)
    out = None
    if call:
        if call not in env or not callable(env[call]):
            raise NameError(f"function {call!r} not found")
        out = env[call]()
    print("\n__LAMAS_OK__" + json.dumps(str(out)))
except BaseException as exc:
    tb = traceback.format_exc(limit=3)
    print("\n__LAMAS_ERR__" + json.dumps(f"{type(exc).__name__}: {exc}\n{tb}"))
"""


async def run_python(code: str, call: Optional[str] = None, timeout: float = 15.0) -> Tuple[bool, str]:
    """Execute ``code``; if ``call`` is given, call that zero-argument function and return its value.

    Returns ``(ok, text)``: the stringified return value, or the error message.
    """
    for name in _BANNED_IMPORTS:
        if f"import {name}" in code or f"from {name}" in code:
            return False, f"Prohibited import: {name}"
    with tempfile.TemporaryDirectory(prefix="lamas_run_") as tmp:
        path = os.path.join(tmp, "solution.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-c", _RUNNER, path, call or "",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, cwd=tmp,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, f"Execution timed out after {timeout:.0f} seconds"
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise
    text = stdout.decode("utf-8", errors="replace")
    # the result is the last line that starts with a marker (the payload itself may mention the markers)
    for line in reversed(text.splitlines()):
        for marker, ok in (("__LAMAS_OK__", True), ("__LAMAS_ERR__", False)):
            if line.startswith(marker):
                payload = line[len(marker):].strip()
                try:
                    return ok, json.loads(payload)
                except json.JSONDecodeError:
                    return ok, payload
    return False, "no result (the process exited early)"
