import numpy as np
import matplotlib.pyplot as plt
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti, load_nifti_data
from dipy.segment.mask import median_otsu
from dipy.io.image import load_nifti
from dipy.viz import regtools
import SimpleITK as sitk
from scipy import ndimage, misc
import shutil 
import os
import nibabel as nib

def resize_volume(img,desired_depth,desired_width, desired_height):

    current_depth = img.shape[0]
    current_width = img.shape[1]
    current_height = img.shape[2]
 
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    
    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
    return img

# Path to original CT
ct_path = 'thirdstage_final_results_goodimages/real_B/'

# Path to the inference sCT
sct1_path = '../../data/two_stage_approach/intermediate_voxels/unsupervised/test/sct/'
sct2_path = 'thirdstage_final_results_goodimages/real_A/'
sct3_path = 'thirdstage_final_results_goodimages/fake_B/'

# Generate final output
two_stage_results = 'outcome_two_stages_goodimages/'

# Remove output folder if already exists
if os.path.exists(two_stage_results):
    shutil.rmtree(two_stage_results)
# Create output folder
os.makedirs(two_stage_results)

imgs_ct = os.listdir(ct_path)
imgs_sct1 = os.listdir(sct1_path)
imgs_sct2 = os.listdir(sct2_path)
imgs_sct3 = os.listdir(sct3_path)

# Define a padding style
pad_width = ((20, 20), (20, 20), (20, 20))  # 20 pixels of padding in each dimension

for img_test in [img[6:9] for img in imgs_ct]:

    ct_, _ = load_nifti(ct_path + '/' + [img for img in imgs_ct if img_test in img][0])
    sct1_, _ = load_nifti(sct1_path + '/' + [img for img in imgs_sct1 if img_test in img][0])
    sct2_, _ = load_nifti(sct2_path + '/' + [img for img in imgs_sct2 if img_test in img][0])
    sct3_, _ = load_nifti(sct3_path + '/' + [img for img in imgs_sct3 if img_test in img][0])

    # Remove images with zero values in the mask
    non_zero_slices_mask_axis1_2 = np.any(ct_, axis=(1, 2))
    ct_ = ct_[non_zero_slices_mask_axis1_2]
    non_zero_slices_mask_axis0_1 = np.any(ct_, axis=(0, 1))
    ct_ = ct_[:,:,non_zero_slices_mask_axis0_1]
    non_zero_slices_mask_axis0_2 = np.any(ct_, axis=(0, 2))
    ct_ = ct_[:,non_zero_slices_mask_axis0_2,:]

    slices_with_minus_800 = sct1_ != -800.0

    non_zero_slices_mask_axis1_2 = np.any(slices_with_minus_800, axis=(1, 2))
    sct1_ = sct1_[non_zero_slices_mask_axis1_2]
    non_zero_slices_mask_axis0_1 = np.any(slices_with_minus_800, axis=(0, 1))
    sct1_ = sct1_[:,:,non_zero_slices_mask_axis0_1]
    non_zero_slices_mask_axis0_2 = np.any(slices_with_minus_800, axis=(0, 2))
    sct1_ = sct1_[:,non_zero_slices_mask_axis0_2,:]

    non_zero_slices_mask_axis1_2 = np.any(sct2_, axis=(1, 2))
    sct2_ = sct2_[non_zero_slices_mask_axis1_2]
    non_zero_slices_mask_axis0_1 = np.any(sct2_, axis=(0, 1))
    sct2_ = sct2_[:,:,non_zero_slices_mask_axis0_1]
    non_zero_slices_mask_axis0_2 = np.any(sct2_, axis=(0, 2))
    sct2_ = sct2_[:,non_zero_slices_mask_axis0_2,:]

    sct2_ = np.pad(sct2_, [(1, 1), (1, 1), (1, 1)], mode='constant', constant_values=0)

    non_zero_slices_mask_axis1_2 = np.any(sct3_, axis=(1, 2))
    sct3_ = sct3_[non_zero_slices_mask_axis1_2]
    non_zero_slices_mask_axis0_1 = np.any(sct3_, axis=(0, 1))
    sct3_ = sct3_[:,:,non_zero_slices_mask_axis0_1]
    non_zero_slices_mask_axis0_2 = np.any(sct3_, axis=(0, 2))
    sct3_ = sct3_[:,non_zero_slices_mask_axis0_2,:]

    # Get the shapes
    ct_shape = ct_.shape
    sct1_shape = sct1_.shape    
    sct2_shape = sct2_.shape
    sct3_shape = sct3_.shape

    # Obtaining the max value of each dim
    x_ = max(ct_shape[0],sct1_shape[0],sct2_shape[0],sct3_shape[0])
    y_ = max(ct_shape[1],sct1_shape[1],sct2_shape[1],sct3_shape[1])
    z_ = max(ct_shape[2],sct1_shape[2],sct2_shape[2],sct3_shape[2])

    # Resizing
    ct_out = resize_volume(ct_, x_,y_,z_)
    sct1_out = resize_volume(sct1_, x_,y_,z_)
    sct2_out = resize_volume(sct2_, x_,y_,z_)
    sct3_out = resize_volume(sct3_, x_,y_,z_)

    # Pass to floats
    ct_out = ct_out.astype(np.float32)
    sct1_out = sct1_out.astype(np.float32)
    sct2_out = sct2_out.astype(np.float32)
    sct3_out = sct3_out.astype(np.float32)

    # We need to convert img_a to HU
    ct_out = ((ct_out* (2000 - (-800)))/ 255) + (-800)
    ct_out = np.clip(ct_out,a_min= -800, a_max= 2000)
    sct2_out = ((sct2_out* (2000 - (-800)))/ 255) + (-800)
    sct2_out = np.clip(sct2_out,a_min= -800, a_max= 2000)
    sct3_out = ((sct3_out* (2000 - (-800)))/ 255) + (-800)
    sct3_out = np.clip(sct3_out,a_min= -800, a_max= 2000)

    # Padding
    ct_out = np.pad(ct_out, pad_width, mode='constant', constant_values=-800)
    sct1_out = np.pad(sct1_out, pad_width, mode='constant', constant_values=-800)
    sct2_out = np.pad(sct2_out, pad_width, mode='constant', constant_values=-800)
    sct3_out = np.pad(sct3_out, pad_width, mode='constant', constant_values=-800)

    # Create a NIfTI image
    ct_out = nib.Nifti1Image(ct_out, affine=np.diag([1, 1, 1, 1]))
    sct1_out = nib.Nifti1Image(sct1_out, affine=np.diag([1, 1, 1, 1]))
    sct2_out = nib.Nifti1Image(sct2_out, affine=np.diag([1, 1, 1, 1]))
    sct3_out = nib.Nifti1Image(sct3_out, affine=np.diag([1, 1, 1, 1]))

    # Save the NIfTI image to a file
    print(f'saving images {img_test}')
    nib.save(ct_out, f'{two_stage_results}/ct_{img_test}.nii.gz')
    nib.save(sct1_out, f'{two_stage_results}/sct1_{img_test}.nii.gz')
    nib.save(sct2_out, f'{two_stage_results}/sct2_{img_test}.nii.gz')
    nib.save(sct3_out, f'{two_stage_results}/sct3_{img_test}.nii.gz')

# Evaluation
# Loading libraries
import PIL.Image
import numpy as np
import PIL
import argparse
import os
from skimage.metrics import structural_similarity
from math import log10, sqrt 

# Definition of peak signal noise ratio
def psnr_function(img_a, img_b): 
    mse = np.mean((img_a - img_b) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                # Therefore PSNR have no importance. 
        return 100.0
    max_pixel = 2800.0 # Maximum range
    psnr_val = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr_val 

# Definition of metrics
def getting_metrics(img_a, img_b):
    # Getting MAE
    metric_mae = np.absolute(np.subtract(img_a, img_b)).mean()
    # Getting SSIM. It is necessary to set a distance between max and min value
    metric_ssim = structural_similarity(img_a, img_b, data_range = 2800) 
    # Getting PSNR
    metric_psnr = psnr_function(img_a,img_b)
    return metric_mae, metric_ssim, metric_psnr

ct_imgs = os.listdir(two_stage_results)
ct_imgs = [img for img in ct_imgs if 'ct_' in img]
ct_imgs = sorted(ct_imgs)

for type in ['3d', '2d-sagittal']:

    for results in ['sct1', 'sct2','sct3']:

        # Set the metrics to zero
        mae = 0
        ssim = 0
        psnr = 0

        files = os.listdir(two_stage_results)
        files = [file for file in files if results in file]
        files = sorted(files)

        if type == '2d-sagittal':
            mae_per_slice = []
            ssim_per_slice = []
            psnr_per_slice = []

            # Getting the metrics for images set B
            for ct_path, sct_path in zip(ct_imgs, files):
            
                # Read images
                img_a, _ = load_nifti(two_stage_results + ct_path)
                img_b, _ = load_nifti(two_stage_results + sct_path)

                # Iterate over each slice of the volume
                for slice in range(img_a.shape[0]):
                    # Compute metrics for the current slice
                    mae_val, ssim_val, psnr_val = getting_metrics(img_a[slice],img_b[slice])

                    # Append metrics to lists
                    mae_per_slice.append(mae_val)
                    ssim_per_slice.append(ssim_val)
                    psnr_per_slice.append(psnr_val)

                mae += np.mean(np.array(mae_per_slice))
                ssim += np.mean(np.array(ssim_per_slice))
                psnr += np.mean(np.array(psnr_per_slice))

        elif type == '3d':

            # Getting the metrics for images set B
            for ct_path, sct_path in zip(ct_imgs, files):
            
                # Read images
                img_a, _ = load_nifti(two_stage_results + ct_path)
                img_b, _ = load_nifti(two_stage_results + sct_path)

                # Calculate metrics
                mae_val, ssim_val, psnr_val = getting_metrics(img_a,img_b)
                # Adding values
                mae += mae_val
                ssim += ssim_val
                psnr += psnr_val

        if results == 'sct1':
            text = f"First inference (Sagittal - sCT1) {type}"
        elif results == 'sct2': 
            text = f"Second inference (Coronal - sCT2) {type}"
        elif results == 'sct3':
            text = f"Third inference (Axial - sCT3) {type}"

        print('')
        print(f'Results in {text}')
        print('MAE: ', mae/len(files))
        print('SSIM: ', ssim/len(files))
        print('PSNR: ', psnr/len(files))
    print("===========================================================")


















