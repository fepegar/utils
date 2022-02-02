try:
    from . import nifti
except Exception:  # probably not using it for medical images
    pass
from . import console
from .core import rgetattr
from .time import chop_microseconds
from .path import ensure_dir, sglob, get_stem
from .registration.affine import AffineMatrix
