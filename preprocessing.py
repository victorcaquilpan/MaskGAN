import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import ants
import pandas as pd
# import cv2
# import imageio
from PIL import Image
from skimage.measure import label   
from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)
import skimage
import os
import glob
import imageio


def visualize(img, filename, step=10):
    shapes = img.shape
    for i, shape in enumerate(shapes):
        fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(12,12))
        start = shape//2 - step*4
        for t, ax in enumerate(axes.flatten()):
            if i == 0:
                data = img[start + t*step, :, :]
            elif i == 1:
                data = img[:, start+t*step, :]
            else:
                data = img[:, :, start+t*step]
            ax.imshow(data, cmap='gray', origin='lower')
            ax.axis('off')
        fig.tight_layout()
        plt.savefig(f'{filename}_{i}.png')
        plt.clf()

def normalize(img, min_=None, max_=None):
    if min_ is None:
        min_ = img.min()
    if max_ is None:
        max_ = img.max()
    return (img - min_)/(max_ - min_)


def crop_scan(img, mask, crop=0, crop_h=0, ignore_zero=True):
    img = np.transpose(img, (0,2,1))[:,::-1,::-1]
    if mask is not None:
        mask = np.transpose(mask, (0,2,1))[:,::-1,::-1]
    if ignore_zero:
        mask_ = img.sum(axis=(1,2)) > 0
        img = img[mask_]
        if mask is not None:
            mask = mask[mask_]
    if crop > 0:
        length = img.shape[0]
        img = img[int(crop*length): int((1-crop)*length)]
        if mask is not None:
            mask = mask[int(crop*length): int((1-crop)*length)]
    
    if crop_h > 0:
        if img.shape[1] > 200:
            crop_h = 0.8
        new_h = int(crop_h*img.shape[1])
        img = img[:, :new_h]
        if mask is not None:
            mask = mask[:, :new_h]

    return img, mask

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def get_3d_mask(img, min_, max_=None, th=50, width=2):
    if max_ is None:
        max_ = img.max()
    img = np.clip(img, min_, max_)
    img = np.uint8(255*normalize(img, min_, max_))

    ## Remove holes
    mask = np.zeros(img.shape).astype(int)
    mask[img > th] = 1

    ## Opening np.ones((3,3,3))
    mask = morphology.binary_opening(mask, )

    remove_holes = morphology.remove_small_holes(
        mask, 
        area_threshold=width ** 3
    )

    largest_cc = getLargestCC(remove_holes)
    return img, largest_cc.astype(int)

def float_to_padded_string(number, total_digits=3):
    formatted_number = format(number, f".{total_digits}f")
    return formatted_number.lstrip('0.') or '0'

def save_slice(img, mask, data_dir, data_mask_dir, filename):
    assert img.shape == mask.shape, f"Shape not match - img {img.shape} vs mask {mask.shape}"
    for i in range(len(img)):
        #im = np.uint8(255*normalize(img[i]))
        im = img[i]
        m = 255*mask[i].astype(np.uint8)
        #m = np.uint8(255*normalize(mask[i]))
        imageio.imwrite(f'{data_dir}/{filename}_{str(i).zfill(3)}_{float_to_padded_string(round(i/len(img),2), 3)}.png', im)
        imageio.imwrite(f'{data_mask_dir}/{filename}_{str(i).zfill(3)}_{float_to_padded_string(round(i/len(img),2), 3)}.png', m)

print("Defining main directories")

match = pd.read_csv('../../data/match.csv')
match_mri = match['MRI_filepath'].tolist()
match_ct = match['CT_filepath'].tolist()
### Train
out_dir = '../../data/arranged_images'
root_ct = '../../data/CT_filtered/*.nii.gz'
root_mri = '../../data/MR_filtered_renamed/*/*.nii.gz'
output_ct_dir = f'{out_dir}/train_B'
output_mri_dir = f'{out_dir}/train_A'
output_ct_mask_dir = f'{out_dir}/train_maskB'
output_mri_mask_dir = f'{out_dir}/train_maskA'

os.makedirs(output_ct_dir, exist_ok=True)
os.makedirs(output_mri_dir, exist_ok=True)
os.makedirs(output_ct_mask_dir, exist_ok=True)
os.makedirs(output_mri_mask_dir, exist_ok=True)
ct_files = glob.glob(root_ct)
ct_files = [f for f in ct_files if f not in match_ct]
mri_files = glob.glob(root_mri)
mri_files = [f for f in mri_files if 'gad' not in f.lower() and f not in match_mri]
resample=(1.0, 1.0, 1.0)
min_ct, max_ct = -800, 1900
th = 50
results = 'vis'
os.makedirs(results, exist_ok=True)

crop = 0.0
crop_h = 0.9
print("Creating CT images")
for idx, filepath in enumerate(ct_files):
    img = ants.image_read(filepath)
    img = ants.resample_image(img, resample, False, 1)
    img = img.numpy()
    if '0db9b48e-2903-41a8-95f2-8f4a710d45ab_Thin_Bone' in filepath:
        img = np.transpose(img, (1, 0, 2))
    #filename = os.path.splitext(os.path.basename(filepath))[0]

    img, mask = get_3d_mask(img, min_=min_ct, max_=max_ct, th=th)
    # Our scans have irregular size, crop to adjust, comment out as needed
    img, mask = crop_scan(img, mask, crop,crop_h)

    # Remove images with zero values in the mask
    non_zero_slices_mask = np.any(mask, axis=(1, 2))
    img = img[non_zero_slices_mask]
    mask = mask[non_zero_slices_mask]

    save_slice(img, mask, output_ct_dir, output_ct_mask_dir, str(idx).zfill(3))
    visualize(img, f'{results}/ct')
    visualize(mask, f'{results}/ct_mask')   

print("Creating MR images")
for idx, filepath in enumerate(mri_files):
    img = ants.image_read(filepath)
    img = ants.resample_image(img, resample, False, 1)
    img = img.numpy()
    #filename = os.path.splitext(os.path.basename(filepath))[0]
    img, mask = get_3d_mask(img, min_=0, th=th, width=10)
    # Our scans have irregular size, crop to adjust, comment out as needed
    img, mask = crop_scan(img, mask, crop,crop_h)

    # Remove images with zero values in the mask
    non_zero_slices_mask = np.any(mask, axis=(1, 2))
    img = img[non_zero_slices_mask]
    mask = mask[non_zero_slices_mask]

    save_slice(img, mask, output_mri_dir, output_mri_mask_dir, str(idx).zfill(3))
    
    visualize(img, f'{results}/mri')
    visualize(mask, f'{results}/mri_mask')

root_ct = '../../data/matched/*/*CT*.nii.gz'
root_mri = '../../data/matched/*/*MR*.nii.gz'
output_ct_dir = f'{out_dir}/test_B'
output_mri_dir = f'{out_dir}/test_A'
output_ct_mask_dir = f'{out_dir}/test_maskB'
output_mri_mask_dir = f'{out_dir}/test_maskA'

os.makedirs(output_ct_dir, exist_ok=True)
os.makedirs(output_mri_dir, exist_ok=True)
os.makedirs(output_ct_mask_dir, exist_ok=True)
os.makedirs(output_mri_mask_dir, exist_ok=True)

# Val
os.makedirs(output_ct_dir.replace('test', 'val'), exist_ok=True)
os.makedirs(output_mri_dir.replace('test', 'val'), exist_ok=True)
os.makedirs(output_ct_mask_dir.replace('test', 'val'), exist_ok=True)
os.makedirs(output_mri_mask_dir.replace('test', 'val'), exist_ok=True)

for i, subset in enumerate(sorted(os.listdir('../../data/matched'))):
    ct_file = glob.glob(f'../../data/matched/{subset}/*CT*')[0]
    mri_files = glob.glob(f'../../data/matched/{subset}/*MR*')
    mri_file = [f for f in mri_files if 'gad' not in f][0]
    filename = os.path.splitext(os.path.basename(ct_file))[0]
    filename = filename.replace('_CT', '')

    mri = ants.image_read(mri_file)
    mri = ants.resample_image(mri, resample, False, 1).numpy()
    mri, mask = get_3d_mask(mri, min_=0, th=th, width=10)
    mri, mask = crop_scan(mri, mask, crop, crop_h, ignore_zero=False)
    outdir = output_mri_dir.replace('test', 'val') if i == 0 else output_mri_dir
    outmask_dir = output_mri_mask_dir.replace('test', 'val') if i == 0 else output_mri_mask_dir

    # Remove images with zero values in the mask
    non_zero_slices_mask = np.any(mask, axis=(1, 2))
    mri = mri[non_zero_slices_mask]
    mask = mask[non_zero_slices_mask]

    save_slice(mri, mask, outdir, outmask_dir, filename)

    ct = ants.image_read(ct_file)
    ct = ants.resample_image(ct, resample, False, 1).numpy()
    ct, mask = get_3d_mask(ct, min_=min_ct,th=th, max_=max_ct)
    ct, mask = crop_scan(ct, mask, crop, crop_h, ignore_zero=False)
    outdir = output_ct_dir.replace('test', 'val') if i == 0 else output_ct_dir
    outmask_dir = output_ct_mask_dir.replace('test', 'val') if i == 0 else output_ct_mask_dir

    # Remove images with zero values in the mask
    non_zero_slices_mask = np.any(mask, axis=(1, 2))
    ct = ct[non_zero_slices_mask]
    mask = mask[non_zero_slices_mask]

    save_slice(ct, mask, outdir, outmask_dir, filename)



