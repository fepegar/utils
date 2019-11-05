from pathlib import Path
from os.path import isfile, splitext

import numpy as np

from . import io


VALID_TRSF_EXTENSIONS = io.NIFTYREG_EXT, io.ITK_EXT



class AffineMatrix(object):
    """Stores an affine matrix going from floating to reference"""

    def __init__(self, affine=None, invert=False):
        """The input parameter affine can be a path or an array"""
        self.matrix = np.eye(4)

        if affine is None:
            return

        if isinstance(affine, (str, Path)):  # path
            path = str(affine)
            self.matrix = self.read(path)

        elif isinstance(affine, np.ndarray):
            if affine.shape != (4, 4):
                raise ValueError('Input array must have shape (4, 4), '
                                 'not {}'.format(affine.shape))
            self.matrix = affine.astype(np.float)

        elif isinstance(affine, AffineMatrix):
            self.matrix = affine.matrix

        else:
            raise ValueError('Input must be a path, '
                             'an AffineMatrix or an array')

        if invert:
            self.invert()


    def __repr__(self):
        np.set_printoptions(precision=2, suppress=True)
        string = '{}\n{}'.format(self.__class__.__name__,
                                 str(self.matrix))
        return string


    def __mul__(self, other):
        result = AffineMatrix(self)
        result.right_multiply(other)
        return result


    @property
    def inverse(self):
        return AffineMatrix(np.linalg.inv(self.matrix))


    def invert(self):
        self.matrix = np.linalg.inv(self.matrix)


    def parse_path(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'The transform {path} does not exist')
        self.parse_extension(path)
        return path


    def parse_extension(self, path):
        path = Path(path)
        extension = path.suffix
        if extension not in VALID_TRSF_EXTENSIONS:
            raise IOError('Transform extension must be .txt or .tfm, '
                          'not {}'.format(extension))


    def read(self, path):
        path = self.parse_path(path)
        if path.suffix == io.NIFTYREG_EXT:
            matrix = io.read_niftyreg_matrix(path)
        elif path.suffix == io.ITK_EXT:
            matrix = io.read_itk_matrix(path)
        return matrix


    def write(self, path):
        self.parse_extension(path)
        if path.endswith(io.NIFTYREG_EXT):  # NiftyReg
            io.write_niftyreg_matrix(self.matrix, path)
        elif path.endswith(io.ITK_EXT):
            io.write_itk_matrix(self.matrix, path)


    def left_multiply(self, affine):
        affine = AffineMatrix(affine)
        self.matrix = np.dot(affine.matrix, self.matrix)


    def right_multiply(self, affine):
        affine = AffineMatrix(affine)
        self.matrix = np.dot(self.matrix, affine.matrix)


    def get_itk_transform(self):
        return io.matrix_to_itk_transform(self.matrix)
