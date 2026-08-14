import importlib
import importlib.util
import sys
from pathlib import Path

from rel_source_aligned.reference import hha_compat


def load_official_rel_module(source_root):
    """Load the pinned author getREL.py while supplying its omitted import."""
    source_root = Path(source_root).resolve()
    entry = source_root / "getREL.py"
    geometry = source_root / "utils" / "rgbd_util.py"
    if not entry.is_file() or not geometry.is_file():
        raise FileNotFoundError("authority root must contain getREL.py and utils/rgbd_util.py")

    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "utils" or name.startswith("utils.")
    }
    for name in saved_modules:
        del sys.modules[name]
    sys.path.insert(0, str(source_root))
    try:
        importlib.import_module("utils")
        sys.modules["utils.hha_util"] = hha_compat
        module_name = "_rel_authority_getREL"
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, str(entry))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        for name in list(sys.modules):
            if name == "utils" or name.startswith("utils."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
    return module

