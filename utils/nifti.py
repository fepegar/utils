from pathlib import Path
from typing import Union, Optional
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from .path import ensure_dir


def load(
        path: Union[str, Path],
        itk: bool = False,
        ) -> Union[nib.Nifti1Image, sitk.Image]:
    if itk:
        image = sitk.ReadImage(str(path))
        return image
    else:
        nii = nib.load(str(path))
        return nii


def save(
        data: Union[np.ndarray, sitk.Image],
        path: Optional[Union[str, Path]] = None,
        affine: Optional[np.ndarray] = None,
        rgb: bool = False,
        itk: bool = False,
        ) -> None:
    itk = isinstance(data, sitk.Image)
    if itk:
        image = data
        sitk.WriteImage(image, str(path))
    else:
        nii = nib.Nifti1Image(data, affine)
        nii.header['qform_code'] = 1
        nii.header['sform_code'] = 0
        if rgb:
            nii.header.set_intent('vector')
        ensure_dir(path)
        nib.save(nii, str(path))


def get_voxel_volume(nii):
    pixdim = get_spacing(nii)
    voxel_volume = np.prod(pixdim)
    return voxel_volume


def get_shape(path):
    return load(path).shape


def get_spacing(path_or_nii):
    if isinstance(path_or_nii, nib.Nifti1Image):
        nii = path_or_nii
    else:
        nii = load(path_or_nii)
    return nii.header.get_zooms()


def get_data(path: Union[str, Path]) -> np.ndarray:
    data = load(path).get_data()
    # I'm getting an error because GridSampler.array is read as a memmap
    if isinstance(data, np.memmap):
        data = np.array(data)
    return data


def transform_points(points, affine, discretize=False):
    transformed = nib.affines.apply_affine(affine, points)
    if discretize:
        transformed = np.round(transformed).astype(np.uint16)
    return transformed


read = load
write = save
