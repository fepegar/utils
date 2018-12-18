from pathlib import Path


def ensure_dir(paths):
    """Make sure that the directory and its parents exists"""
    if not isinstance(paths, (list, tuple)):
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
