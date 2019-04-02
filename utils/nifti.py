import numpy as np

from .path import ensure_dir


def load(path, itk=False):
    if itk:
        import SimpleITK as sitk
        image = sitk.ReadImage(str(path))
        return image
    else:
        import nibabel as nib
        nii = nib.load(str(path))
        return nii


def save(data, path=None, affine=None, rgb=False, itk=False):
    import SimpleITK as sitk
    itk = isinstance(data, sitk.Image)
    if itk:
        image = data
        sitk.WriteImage(image, str(path))
    else:
        import nibabel as nib
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
    import nibabel as nib
    if isinstance(path_or_nii, nib.Nifti1Image):
        nii = path_or_nii
    else:
        nii = load(path_or_nii)
    return nii.header.get_zooms()


def transform_points(points, affine, discretize=False):
    import nibabel as nib
    transformed = nib.affines.apply_affine(affine, points)
    if discretize:
        transformed = np.round(transformed).astype(np.uint16)
    return transformed


read = load
write = save
