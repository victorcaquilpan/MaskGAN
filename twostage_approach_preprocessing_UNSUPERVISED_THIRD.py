import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import ants
import pandas as pd
import cv2
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
import shutil

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

def crop_scan_paired(img1, img2, mask, crop=0, crop_h=0, ignore_zero=True):
    img1 = np.transpose(img1, (0,2,1))[:,::-1,::-1]
    img2 = np.transpose(img2, (0,2,1))[:,::-1,::-1]
    if mask is not None:
        mask = np.transpose(mask, (0,2,1))[:,::-1,::-1]
    if ignore_zero:
        mask1_ = img1.sum(axis=(1,2)) > 0
        mask2_ = img2.sum(axis=(1,2)) > 0
        mask_ = mask1_ * mask2_
        img1 = img1[mask_]
        img2 = img2[mask_]
        if mask is not None:
            mask = mask[mask_]
    if crop > 0:
        length = img1.shape[0]
        img1 = img1[int(crop*length): int((1-crop)*length)]
        img2 = img2[int(crop*length): int((1-crop)*length)]
        if mask is not None:
            mask = mask[int(crop*length): int((1-crop)*length)]
    
    if crop_h > 0:
        if img1.shape[1] > 200:
            crop_h = 0.8
        new_h = int(crop_h*img1.shape[1])
        img1 = img1[:, :new_h]
        img2 = img2[:, :new_h]
        if mask is not None:
            mask = mask[:, :new_h]

    return img1, img2, mask

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def get_3d_mask(img, min_, max_=None, th=50, width=2):
    if max_ is None:
        max_ = img.max()
    #img = np.clip(img, min_, max_)
    #img = np.uint8(255*normalize(img, min_, max_))

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
        im = img[i].astype(np.uint8)
        m = 255*mask[i].astype(np.uint8)
        # Remove noise
        kernel = np.ones((10,10), np.uint8) # Define the kernel size
        dilated_mask = cv2.dilate(m.astype(np.uint8), kernel, iterations = 2)
        im[dilated_mask == 0] = 0

        #m = np.uint8(255*normalize(mask[i]))
        imageio.imwrite(f'{data_dir}/{filename}_{str(i).zfill(3)}_{float_to_padded_string(round(i/len(img),2), 3)}.png', im)
        imageio.imwrite(f'{data_mask_dir}/{filename}_{str(i).zfill(3)}_{float_to_padded_string(round(i/len(img),2), 3)}.png', m)

print("Defining main directories")

### TRAIN
out_dir = '../../data/two_stage_approach/slicing_second_inference_goodimages'

# Remove output folder if already exists
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
# Create output folder
os.makedirs(out_dir)

# Define the sct folder input
root_intermediate = 'twostage_readyforthirdinference_results_goodimages/'
test_sct_root =  root_intermediate + 'fake_B/'
test_ct_root =  root_intermediate + 'real_B/'

# Extract the images
test_ct = test_ct_root + '*.nii.gz'
test_sct = test_sct_root + '*.nii.gz'

# root_ct = '../../data/structured-data-3d//*.nii.gz'
# root_mri = '../../data/MR_filtered_renamed/*/*.nii.gz'
#resample=(1.0, 1.0, 1.0)
#min_ct, max_ct = -800, 1900
th_ct = 0
# Set clip CT intensity
min_ct, max_ct = 0, 255
#th = 10 # Consider pixel less than certain threshold as background (remove noise, artifacts)

results = 'vis'
os.makedirs(results, exist_ok=True)

crop = 0.0
crop_h = 0
resample = [1.0, 1.0, 1.0]

### TESTING
output_ct_dir = f'{out_dir}/test_B'
output_sct_dir = f'{out_dir}/test_A'
output_ct_mask_dir = f'{out_dir}/test_maskB'
output_sct_mask_dir = f'{out_dir}/test_maskA'

os.makedirs(output_ct_dir, exist_ok=True)
os.makedirs(output_sct_dir, exist_ok=True)
os.makedirs(output_ct_mask_dir, exist_ok=True)
os.makedirs(output_sct_mask_dir, exist_ok=True)

ct_files_test = glob.glob(test_ct)
sct_files_test = glob.glob(test_sct)

print("Creating sCT images and CT scans for testing")

for sct_path, ct_path in zip(sct_files_test,ct_files_test):

    sct = ants.image_read(sct_path).numpy()
    sct = np.rot90(sct, axes = (2,0))
    sct = np.rot90(sct,axes = (2,1))

    #sct = ants.resample_image(sct, resample, False, 1)
    #filename = os.path.splitext(os.path.basename(filepath))[0]
    sct, sct_mask = get_3d_mask(sct, min_=min_ct, max_= max_ct, th=th_ct)

    ct = ants.image_read(ct_path).numpy()
    ct = np.rot90(ct, axes = (2,0))
    ct = np.rot90(ct,axes = (2,1))
    #ct = ants.resample_image(ct, resample, False, 1)
    ct, ct_mask = get_3d_mask(ct, min_=min_ct, max_=max_ct, th=th_ct)

    # Getting a uniform mask template for paired images
    uniform_mask = sct_mask * ct_mask

    # Our scans have irregular size, crop to adjust, comment out as needed
    sct, ct, uniform_mask = crop_scan_paired(sct, ct, uniform_mask, crop,crop_h)

    # Remove images with zero values in the mask
    non_zero_slices_mask_axis1_2 = np.any(uniform_mask, axis=(1, 2))
    sct = sct[non_zero_slices_mask_axis1_2]
    ct = ct[non_zero_slices_mask_axis1_2]
    uniform_mask = uniform_mask[non_zero_slices_mask_axis1_2]
    non_zero_slices_mask_axis0_1 = np.any(uniform_mask, axis=(0, 1))
    sct = sct[:,:,non_zero_slices_mask_axis0_1]
    ct = ct[:,:,non_zero_slices_mask_axis0_1]
    uniform_mask = uniform_mask[:,:,non_zero_slices_mask_axis0_1]
    non_zero_slices_mask_axis0_2 = np.any(uniform_mask, axis=(0, 2))
    sct = sct[:,non_zero_slices_mask_axis0_2,:]
    ct = ct[:,non_zero_slices_mask_axis0_2,:]
    uniform_mask = uniform_mask[:,non_zero_slices_mask_axis0_2,:]

    # Enter the name of the file
    filename = sct_path.split('/')[-1].replace('.nii.gz','')
    # Create a generic format
    filename = filename.zfill(3)
    filename = filename.replace("coronal","axial")
    
    save_slice(sct, uniform_mask, output_sct_dir, output_sct_mask_dir, filename)
    save_slice(ct, uniform_mask, output_ct_dir, output_ct_mask_dir, filename)
    print(f'Saving {filename} image')