"""Load your own operators, tasks, controllers or stop policies from files, without packaging them.

    # in a config
    plugins: [tools.py, my_company.lamas_ops]      # .py files (relative to the config) or module names

Importing a plugin runs its ``@register_*`` decorators, so the names it defines can be used in
the same config. A plugin is ordinary Python and runs with your permissions. For a package that
others install, use entry points instead (see ``lamas.registry``).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Iterable, List, Optional, Union


def load_plugins(specs: Union[str, Iterable[str]], base_dir: Optional[str] = None) -> List[str]:
    """Import each plugin: a path to a ``.py`` file or a module name. Returns the module names."""
    if isinstance(specs, str):
        specs = [specs]
    loaded = []
    for spec in specs:
        spec = str(spec)
        if spec.endswith(".py") or os.sep in spec or "/" in spec:
            path = spec if os.path.isabs(spec) else os.path.join(base_dir or os.getcwd(), spec)
            path = os.path.abspath(path)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"plugin {spec!r} not found at {path}")
            module_name = "lamas_plugin_" + os.path.splitext(os.path.basename(path))[0]
            if module_name not in sys.modules:
                module_spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(module_spec)
                sys.modules[module_name] = module
                try:
                    module_spec.loader.exec_module(module)
                except BaseException:
                    del sys.modules[module_name]
                    raise
            loaded.append(module_name)
        else:
            importlib.import_module(spec)
            loaded.append(spec)
    return loaded
