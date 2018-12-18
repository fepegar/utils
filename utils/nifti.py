import numpy as np
import nibabel as nib
import SimpleITK as sitk

from .path import ensure_dir


def load(path, itk=False):
    if itk:
        image = sitk.ReadImage(str(path))
        return image
    else:
        nii = nib.load(str(path))
        return nii


def save(data, affine, path, rgb=False):
    nii = nib.Nifti1Image(data, affine)
    nii.header['qform_code'] = 1
    nii.header['sform_code'] = 0
    if rgb:
        nii.header.set_intent('vector')
    ensure_dir(path)
    nib.save(nii, str(path))


def get_voxel_volume(nii):
    dims = nii.header['pixdim'][1:4]
    voxel_volume = np.prod(dims)
    return voxel_volume


def get_shape(path):
    return load(path).shape


def get_spacing(path):
    return load(path).header.get_zooms()


def transform_points(points, affine, discretize=False):
    transformed = nib.affines.apply_affine(affine, points)
    if discretize:
        transformed = np.round(transformed).astype(np.uint16)
    return transformed


read = load
write = save
