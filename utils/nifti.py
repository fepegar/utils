"""
See https://github.com/ANTsX/ANTs/wiki/How-does-ANTs-handle-qform-and-sform-in-NIFTI-1-images%3F

"When writing NIFTI files, ITK encodes the rotation, translation and scaling in
 the qform. The qform code is set to 1 (NIFTI_XFORM_SCANNER_ANAT), and the sform
 code is set to 0. The loss of the sform is unavoidable because the ITK I/O
 code reads images into a common internal structure, independent of the
    individual image format on disk, and only supports a rigid transformation."

"""

import warnings
from pathlib import Path
from typing import Union, Optional

import numpy as np
from .path import ensure_dir

try:
    import nibabel as nib
    import SimpleITK as sitk
except Exception:  # probably not using it for medical images
    pass


SFORM_CODES = {
    0: 'unknown (sform not defined)',
    1: 'scanner (RAS+ in scanner coordinates)',
    2: 'aligned (RAS+ aligned to some other scan)',
    3: 'talairach (RAS+ in Talairach atlas space)',
    4: 'mni',
}


def load(
        path: Union[str, Path],
        itk: bool = False,
        mmap: bool = True,
        warn_sform: bool = True,
        ) -> Union[nib.Nifti1Image, sitk.Image]:
    nii = nib.load(str(path), mmap=mmap)
    if warn_sform:
        sform_code = int(nii.header['sform_code'])
        if sform_code == 2:
            meaning = SFORM_CODES[sform_code]
            path = Path(path)
            warnings.warn(f'{path.name} has sform code {sform_code}: {meaning}')
    if itk:
        image = sitk.ReadImage(str(path))
        image += 0  # https://discourse.itk.org/t/simpleitk-writing-nifti-with-invalid-header/2498/4
        return image
    else:
        nii = nib.load(str(path), mmap=mmap)
        return nii


def save(
        data: Union[np.ndarray, sitk.Image],
        path: Union[str, Path],
        affine: Optional[np.ndarray] = None,
        rgb: bool = False,
        header: Optional[nib.nifti1.Nifti1Header] = None,
        ) -> None:
    ensure_dir(path)
    itk = isinstance(data, sitk.Image)
    if itk:
        image = data
        sitk.WriteImage(image, str(path))
    else:
        nii = nib.Nifti1Image(data, affine, header=header)
        nii.header['qform_code'] = 1
        nii.header['sform_code'] = 0
        if rgb:
            nii.header.set_intent('vector')
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
