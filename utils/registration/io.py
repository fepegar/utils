import numpy as np

FLIPXY = np.diag([-1, -1, 1, 1])
ITK_H5 = '.h5'
ITK_TFM = '.tfm'
ITK_EXT = ITK_H5, ITK_TFM
NIFTYREG_EXT = '.txt'


def read_niftyreg_matrix(trsf_path):
    """Read a NiftyReg matrix and return it as a NumPy array"""
    matrix = np.loadtxt(trsf_path)
    matrix = np.linalg.inv(matrix)
    return matrix


def write_niftyreg_matrix(matrix, txt_path):
    """Write an affine transform in NiftyReg's .txt format (ref -> flo)"""
    matrix = np.linalg.inv(matrix)
    np.savetxt(txt_path, matrix, fmt='%.8f')


def to_itk_convention(matrix):
    """Apply some operations so that the transform is in ITK's LPS format"""
    matrix = np.dot(FLIPXY, matrix)
    matrix = np.dot(matrix, FLIPXY)
    matrix = np.linalg.inv(matrix)
    return matrix


def from_itk_convention(matrix):
    """From floating to reference, LPS"""
    matrix = np.dot(matrix, FLIPXY)
    matrix = np.dot(FLIPXY, matrix)
    matrix = np.linalg.inv(matrix)
    return matrix


def read_itk_matrix(trsf_tfm_path):
    """Read an affine transform in ITK's .tfm format"""
    import SimpleITK as sitk
    transform = sitk.ReadTransform(str(trsf_tfm_path))
    parameters = transform.GetParameters()

    rotation_parameters = parameters[:9]
    rotation_matrix = np.array(rotation_parameters).reshape(3, 3)

    translation_parameters = parameters[9:]
    translation_vector = np.array(translation_parameters).reshape(3, 1)

    matrix = np.hstack([rotation_matrix, translation_vector])
    homogeneous_matrix_lps = np.vstack([matrix, [0, 0, 0, 1]])
    homogeneous_matrix_ras = from_itk_convention(homogeneous_matrix_lps)
    return homogeneous_matrix_ras


def write_itk_matrix(matrix, tfm_path):
    """The tfm file contains the matrix from floating to reference."""
    import SimpleITK as sitk
    transform = matrix_to_itk_transform(matrix)
    transform.WriteTransform(str(tfm_path))


def matrix_to_itk_transform(matrix, dimensions=3):
    import SimpleITK as sitk
    matrix = to_itk_convention(matrix)
    rotation = matrix[:dimensions, :dimensions].ravel().tolist()
    translation = matrix[:dimensions, 3].tolist()
    transform = sitk.AffineTransform(rotation, translation)
    return transform
