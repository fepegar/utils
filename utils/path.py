from pathlib import Path
from typing import List, Union

def ensure_dir(paths: Union[list, tuple, str, Path]) -> None:
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


def sglob(path: Union[Path, str], pattern) -> List[Path]:
    path = Path(path)
    return sorted(list(path.glob(pattern)))
