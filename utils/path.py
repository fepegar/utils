from pathlib import Path
from collections import Iterable

def ensure_dir(paths):
    """Make sure that the directory and its parents exists"""
    if not isinstance(paths, Iterable):
        paths = [paths]
    for path in paths:
        path = Path(path)
        if path.exists():
            return
        is_dir = not path.suffixes
        if is_dir:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
