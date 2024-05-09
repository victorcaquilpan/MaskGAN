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
import PIL.Image
import numpy as np
import PIL
import argparse
import os
from skimage.metrics import structural_similarity
from math import log10, sqrt 
from dipy.align import affine_registration

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
#ct_path = 'erase/real_B/'
ct_path = 'thirdstage_final_results_goodimages/real_B/'
#sct1_path = 'erase/real_A/'

sct1_path = '../../data/two_stage_approach/intermediate_voxels/unsupervised/test/sct/'

# Path to the inference sCT
sct2_path = 'erase/fake_B/'
sct2_path_c = 'thirdstage_final_results_goodimages/real_A/'

# Generate final output
two_stage_results = 'outcome_erase/'





# # # Remove output folder if already exists
# if os.path.exists(two_stage_results):
#      shutil.rmtree(two_stage_results)
# # Create output folder
# os.makedirs(two_stage_results)

# imgs_ct = os.listdir(ct_path)
# imgs_ct = [ct for ct in imgs_ct if 'axial' in ct]
# imgs_sct1 = os.listdir(sct1_path)
# imgs_sct1 = [ct for ct in imgs_sct1]
# imgs_ct_axial = os.listdir(sct2_path)
# imgs_ct_axial = [ct for ct in imgs_ct_axial if 'axial' in ct]
# imgs_ct_coronal = os.listdir(sct2_path_c)
# imgs_ct_coronal = [ct for ct in imgs_ct_coronal if 'axial' in ct]

# # Define a padding style
# pad_width = ((20, 20), (20, 20), (20, 20))  # 20 pixels of padding in each dimension

# for img_test in [img[6:9] for img in imgs_ct]:

#     ct_, _ = load_nifti(ct_path + '/' + [img for img in imgs_ct if img_test in img][0])
#     sct1_, _ = load_nifti(sct1_path + '/' + [img for img in imgs_sct1 if img_test in img][0])
#     sct2_a, _ = load_nifti(sct2_path + '/' + [img for img in imgs_ct_axial if img_test in img][0])
#     sct2_c, _ = load_nifti(sct2_path_c + '/' + [img for img in imgs_ct_coronal if img_test in img][0])

#     # Remove images with zero values in the mask
#     non_zero_slices_mask_axis1_2 = np.any(ct_, axis=(1, 2))
#     ct_ = ct_[non_zero_slices_mask_axis1_2]
#     non_zero_slices_mask_axis0_1 = np.any(ct_, axis=(0, 1))
#     ct_ = ct_[:,:,non_zero_slices_mask_axis0_1]
#     non_zero_slices_mask_axis0_2 = np.any(ct_, axis=(0, 2))
#     ct_ = ct_[:,non_zero_slices_mask_axis0_2,:]

#     slices_with_minus_800 = sct1_ != -800.0
#     non_zero_slices_mask_axis1_2 = np.any(slices_with_minus_800, axis=(1, 2))
#     sct1_ = sct1_[non_zero_slices_mask_axis1_2]
#     non_zero_slices_mask_axis0_1 = np.any(slices_with_minus_800, axis=(0, 1))
#     sct1_ = sct1_[:,:,non_zero_slices_mask_axis0_1]
#     non_zero_slices_mask_axis0_2 = np.any(slices_with_minus_800, axis=(0, 2))
#     sct1_ = sct1_[:,non_zero_slices_mask_axis0_2,:]
    
#     non_zero_slices_mask_axis1_2 = np.any(sct2_a, axis=(1, 2))
#     sct2_a = sct2_a[non_zero_slices_mask_axis1_2]
#     non_zero_slices_mask_axis0_1 = np.any(sct2_a, axis=(0, 1))
#     sct2_a = sct2_a[:,:,non_zero_slices_mask_axis0_1]
#     non_zero_slices_mask_axis0_2 = np.any(sct2_a, axis=(0, 2))
#     sct2_a = sct2_a[:,non_zero_slices_mask_axis0_2,:]

#     #sct2_a = np.pad(sct2_a, [(1, 1), (1, 1), (1, 1)], mode='constant', constant_values=0)
#     sct2_a = np.rot90(sct2_a, axes = (2,1))
#     sct2_a = np.flip(sct2_a,axis=(1,0))

#     non_zero_slices_mask_axis1_2 = np.any(sct2_c, axis=(1, 2))
#     sct2_c = sct2_c[non_zero_slices_mask_axis1_2]
#     non_zero_slices_mask_axis0_1 = np.any(sct2_c, axis=(0, 1))
#     sct2_c = sct2_c[:,:,non_zero_slices_mask_axis0_1]
#     non_zero_slices_mask_axis0_2 = np.any(sct2_c, axis=(0, 2))
#     sct2_c = sct2_c[:,non_zero_slices_mask_axis0_2,:]

#     sct2_c = np.pad(sct2_c, [(1, 1), (1, 1), (1, 1)], mode='constant', constant_values=0)

#     # Get the shapes
#     ct_shape = ct_.shape
#     sct1_shape = sct1_.shape
#     sct2_shape = sct2_a.shape    
#     sct2_shape = sct2_c.shape
   
#     # Obtaining the max value of each dim
#     x_ = max(ct_shape[0],sct1_shape[0],sct2_shape[0],sct2_shape[0])
#     y_ = max(ct_shape[1],sct1_shape[1],sct2_shape[1],sct2_shape[1])
#     z_ = max(ct_shape[2],sct1_shape[2],sct2_shape[2],sct2_shape[2])

#     x_ = 134
#     y_ = 207
#     z_ = 226

#     # Resizing
#     ct_out = resize_volume(ct_, x_,y_,z_)
#     sct1_out = resize_volume(sct1_, x_,y_,z_)
#     sct2_a_out = resize_volume(sct2_a, x_,y_,z_)
#     sct2_c_out = resize_volume(sct2_c, x_,y_,z_)

#     # Pass to floats
#     ct_out = ct_out.astype(np.float32)
#     sct1_out = sct1_out.astype(np.float32)
#     sct2_a_out = sct2_a_out.astype(np.float32)
#     sct2_c_out = sct2_c_out.astype(np.float32)

#     # We need to convert img_a to HU
#     ct_out = ((ct_out* (2000 - (-800)))/ 255) + (-800)
#     ct_out = np.clip(ct_out,a_min= -800, a_max= 2000)
#     #sct1_out = ((sct1_out* (2000 - (-800)))/ 255) + (-800)
#     #sct1_out = np.clip(sct1_out,a_min= -800, a_max= 2000)
#     sct2_a_out = ((sct2_a_out* (2000 - (-800)))/ 255) + (-800)
#     sct2_a_out = np.clip(sct2_a_out,a_min= -800, a_max= 2000)
#     sct2_c_out = ((sct2_c_out* (2000 - (-800)))/ 255) + (-800)
#     sct2_c_out = np.clip(sct2_c_out,a_min= -800, a_max= 2000)

#     # Padding
#     ct_out = np.pad(ct_out, pad_width, mode='constant', constant_values=-800)
#     sct1_out = np.pad(sct1_out, pad_width, mode='constant', constant_values=-800)
#     sct2_a_out = np.pad(sct2_a_out, pad_width, mode='constant', constant_values=-800)
#     sct2_c_out = np.pad(sct2_c_out, pad_width, mode='constant', constant_values=-800)

#     # Generate average image
#     sct_ave_out = (sct2_a_out + sct2_c_out) / 2

#     # Create a NIfTI image
#     ct_out = nib.Nifti1Image(ct_out, affine=np.diag([1, 1, 1, 1]))
#     sct1_out = nib.Nifti1Image(sct1_out, affine=np.diag([1, 1, 1, 1]))
#     sct2_a_out = nib.Nifti1Image(sct2_a_out, affine=np.diag([1, 1, 1, 1]))
#     sct2_c_out = nib.Nifti1Image(sct2_c_out, affine=np.diag([1, 1, 1, 1]))
#     sct_ave_out = nib.Nifti1Image(sct_ave_out, affine=np.diag([1, 1, 1, 1]))

#     # Save the NIfTI image to a file
#     print(f'saving images {img_test}')
#     nib.save(ct_out, f'{two_stage_results}/ct_{img_test}.nii.gz')
#     nib.save(sct1_out, f'{two_stage_results}/sct1_{img_test}.nii.gz')
#     nib.save(sct2_a_out, f'{two_stage_results}/sct_a_{img_test}.nii.gz')
#     nib.save(sct2_c_out, f'{two_stage_results}/sct_c_{img_test}.nii.gz')
#     nib.save(sct_ave_out, f'{two_stage_results}/sct_ave_{img_test}.nii.gz')

#     pipeline = ["center_of_mass", "translation"]
#     level_iters = [10]
#     sigmas = [3.0, 1.0, 0.0]
#     factors = [4, 2, 1]

#     sct_a, sct_a_affine = load_nifti(f'{two_stage_results}/sct_a_{img_test}.nii.gz')
#     sct_c, sct_c_affine = load_nifti(f'{two_stage_results}/sct_c_{img_test}.nii.gz')

#     xformed_img, reg_affine = affine_registration(
#     sct_a,
#     sct_c,
#     moving_affine=sct_a_affine,
#     static_affine=sct_c_affine,
#     nbins=32,
#     metric='MI',
#     pipeline=pipeline,
#     level_iters=level_iters,
#     sigmas=sigmas,
#     factors=factors)

#     # Generate average image
#     sct_reg_ave_out = (xformed_img + sct_c) / 2
#     save_nifti(f'{two_stage_results}/sct_reg_ave_{img_test}.nii.gz', sct_reg_ave_out, reg_affine)

ct_imgs = ['ct_271.nii.gz','ct_272.nii.gz','ct_273.nii.gz','ct_274.nii.gz']

for type in ['3d', '2d-sagittal', '2d-axial', '2d-coronal']:
        
    for results in ['sct1_','sct_a_', 'sct_c_','sct_ave_','sct_reg_ave_']:

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

        elif type == '2d-coronal':
            mae_per_slice = []
            ssim_per_slice = []
            psnr_per_slice = []

            # Getting the metrics for images set B
            for ct_path, sct_path in zip(ct_imgs, files):
            
                # Read images
                img_a, _ = load_nifti(two_stage_results + ct_path)
                img_b, _ = load_nifti(two_stage_results + sct_path)

                # Iterate over each slice of the volume
                for slice in range(img_a.shape[1]):
                    # Compute metrics for the current slice
                    mae_val, ssim_val, psnr_val = getting_metrics(img_a[:,slice,:],img_b[:,slice,:])

                    # Append metrics to lists
                    mae_per_slice.append(mae_val)
                    ssim_per_slice.append(ssim_val)
                    psnr_per_slice.append(psnr_val)

                mae += np.mean(np.array(mae_per_slice))
                ssim += np.mean(np.array(ssim_per_slice))
                psnr += np.mean(np.array(psnr_per_slice))

        elif type == '2d-axial':
            mae_per_slice = []
            ssim_per_slice = []
            psnr_per_slice = []

            # Getting the metrics for images set B
            for ct_path, sct_path in zip(ct_imgs, files):
            
                # Read images
                img_a, _ = load_nifti(two_stage_results + ct_path)
                img_b, _ = load_nifti(two_stage_results + sct_path)

                # Iterate over each slice of the volume
                for slice in range(img_a.shape[2]):
                    # Compute metrics for the current slice
                    mae_val, ssim_val, psnr_val = getting_metrics(img_a[:,:,slice],img_b[:,:,slice])

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

        if results == 'sct1_':
            text = f"First infence (Sagittal) {type}"
        elif results == 'sct_a_':
            text = f"Second inference (Sagittal -> Axial) {type}"
        elif results == 'sct_c_':
            text = f"Second inference (Sagittal -> Coronal) {type}"
        elif results == 'sct_ave_':
            text = f"Second inference (Sagittal -> Axial/Coronal) {type}"
        elif results == 'sct_reg_ave_':
            text = f"Second inference registered (Sagittal -> Axial/Coronal) {type}"
        print('')
        print(f'Results in {text}')
        print('MAE: ', mae/len(files))
        print('SSIM: ', ssim/len(files))
        print('PSNR: ', psnr/len(files))
    print("===========================================================")

