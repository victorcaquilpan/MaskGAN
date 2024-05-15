import numpy as np
from dipy.io.image import load_nifti
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
import argparse

# Call the parser
parser = argparse.ArgumentParser()
parser.add_argument('--results_folder', default='secondstage_solved/', help='path to intermediate results (2nd stage)')
parser.add_argument('--final_voxels_folder', default='secondstage_solved/', help='path to leave results')
args = parser.parse_args()

ct_path = 'vol_results/' + args.results_folder  +'real_B/'
sct1_path = 'vol_results/' + args.results_folder + 'real_A/'
sct2_path_c = 'vol_results/' + args.results_folder + 'fake_B/'
# Generate final output
two_stage_results = 'adjusted_results/' + args.final_voxels_folder

# # Remove output folder if already exists
if os.path.exists(two_stage_results):
     shutil.rmtree(two_stage_results)
# Create output folder
os.makedirs(two_stage_results)

# Definition of peak signal noise ratio
def psnr_function(img_a, img_b): 
    mse = np.mean((img_a - img_b) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                # Therefore PSNR have no importance. 
        return 100.0
    max_pixel = 2800.0 # Maximum range
    psnr_val = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr_val 

def ssim_function(img_a, img_b):
    img_a = ((img_a) + 800)
    img_b = ((img_b) + 800)

    return structural_similarity(img_a, img_b, data_range = 2800) 

# Definition of metrics
def getting_metrics(img_a, img_b):
    # Getting MAE
    metric_mae = np.absolute(np.subtract(img_a, img_b)).mean()
    # Getting SSIM. It is necessary to set a distance between max and min value
    metric_ssim = ssim_function(img_a, img_b)
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

# Getting images
imgs_ct = os.listdir(ct_path)
imgs_ct = [ct for ct in imgs_ct]
imgs_sct1 = os.listdir(sct1_path)
imgs_sct1 = [ct for ct in imgs_sct1]
imgs_ct_coronal = os.listdir(sct2_path_c)
imgs_ct_coronal = [ct for ct in imgs_ct_coronal]

# Define a padding style
pad_width = ((20, 20), (20, 20), (20, 20))  # 20 pixels of padding in each dimension

for img_test in [img[0:3] for img in imgs_ct]:

    ct_, _ = load_nifti(ct_path + '/' + [img for img in imgs_ct if img_test in img][0])
    sct1_, _ = load_nifti(sct1_path + '/' + [img for img in imgs_sct1 if img_test in img][0])
    sct2_c, _ = load_nifti(sct2_path_c + '/' + [img for img in imgs_ct_coronal if img_test in img][0])

    #Remove images with zero values in the mask
    non_zero_slices_mask_axis1_2 = np.any(ct_, axis=(1, 2))
    ct_ = ct_[non_zero_slices_mask_axis1_2]
    non_zero_slices_mask_axis0_1 = np.any(ct_, axis=(0, 1))
    ct_ = ct_[:,:,non_zero_slices_mask_axis0_1]
    non_zero_slices_mask_axis0_2 = np.any(ct_, axis=(0, 2))
    ct_ = ct_[:,non_zero_slices_mask_axis0_2,:]

    non_zero_slices_mask_axis1_2 = np.any(sct1_, axis=(1, 2))
    sct1_ = sct1_[non_zero_slices_mask_axis1_2]
    non_zero_slices_mask_axis0_1 = np.any(sct1_, axis=(0, 1))
    sct1_ = sct1_[:,:,non_zero_slices_mask_axis0_1]
    non_zero_slices_mask_axis0_2 = np.any(sct1_, axis=(0, 2))
    sct1_ = sct1_[:,non_zero_slices_mask_axis0_2,:]

    non_zero_slices_mask_axis1_2 = np.any(sct2_c, axis=(1, 2))
    sct2_c = sct2_c[non_zero_slices_mask_axis1_2]
    non_zero_slices_mask_axis0_1 = np.any(sct2_c, axis=(0, 1))
    sct2_c = sct2_c[:,:,non_zero_slices_mask_axis0_1]
    non_zero_slices_mask_axis0_2 = np.any(sct2_c, axis=(0, 2))
    sct2_c = sct2_c[:,non_zero_slices_mask_axis0_2,:]

    x_ = 134
    y_ = 207
    z_ = 226

    # # Resizing
    ct_ = resize_volume(ct_, x_,y_,z_)
    sct1_ = resize_volume(sct1_, x_,y_,z_)
    sct2_c = resize_volume(sct2_c, x_,y_,z_)

    # Pass to floats
    ct_out = ct_.astype(np.float32)
    sct1_out = sct1_.astype(np.float32)
    sct2_c_out = sct2_c.astype(np.float32)

    # We need to convert img_a to HU
    ct_out = ((ct_out* (2000 - (-800)))/ 255) + (-800)
    ct_out = np.clip(ct_out,a_min= -800, a_max= 2000)
    sct1_out = ((sct1_out* (2000 - (-800)))/ 255) + (-800)
    sct1_out = np.clip(sct1_out,a_min= -800, a_max= 2000)
    sct2_c_out = ((sct2_c_out* (2000 - (-800)))/ 255) + (-800)
    sct2_c_out = np.clip(sct2_c_out,a_min= -800, a_max= 2000)

    #Padding
    ct_out = np.pad(ct_out, pad_width, mode='constant', constant_values=-800)
    sct1_out = np.pad(sct1_out, pad_width, mode='constant', constant_values=-800)
    sct2_c_out = np.pad(sct2_c_out, pad_width, mode='constant', constant_values=-800)

    # Create a NIfTI image
    ct_out = nib.Nifti1Image(ct_out, affine=np.diag([1, 1, 1, 1]))
    sct1_out = nib.Nifti1Image(sct1_out, affine=np.diag([1, 1, 1, 1]))
    sct2_c_out = nib.Nifti1Image(sct2_c_out, affine=np.diag([1, 1, 1, 1]))

    # Save the NIfTI image to a file
    print(f'saving images {img_test}')
    nib.save(ct_out, f'{two_stage_results}/ct_{img_test}.nii.gz')
    nib.save(sct1_out, f'{two_stage_results}/sct1_{img_test}.nii.gz')
    nib.save(sct2_c_out, f'{two_stage_results}/sct2_{img_test}.nii.gz')

files = os.listdir(two_stage_results)
ct_imgs = [file for file in files if 'ct_' in file]
ct_imgs = sorted(ct_imgs)

#for type in ['3d', '2d-sagittal', '2d-axial', '2d-coronal']:
for type in ['3d']:
        
    for results in ['sct1_', 'sct2_']:

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
        elif results == 'sct2_':
            text = f"Second inference (Sagittal -> Coronal) {type}"
        # elif results == 'sct_reg_c_':
        #     text = f"Second inference registered (Sagittal -> Coronal) {type}"

        print('')
        print(f'Results in {text}')
        print('MAE: ', mae/len(files))
        print('SSIM: ', ssim/len(files))
        print('PSNR: ', psnr/len(files))
    print("===========================================================")



