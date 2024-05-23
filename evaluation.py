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
parser.add_argument('--stage', default='second', help='first or second')
args = parser.parse_args()


real_B_path = 'vol_results/' + args.results_folder  +'real_B/'
real_A_path = 'vol_results/' + args.results_folder + 'real_A/'
fake_B_path = 'vol_results/' + args.results_folder  +'fake_B/'
fake_A_path = 'vol_results/' + args.results_folder + 'fake_A/'
# Generate final output
adjusted_results = 'adjusted_results/' + args.final_voxels_folder

# # Remove output folder if already exists
if os.path.exists(adjusted_results):
     shutil.rmtree(adjusted_results)
# Create output folder
os.makedirs(adjusted_results)
os.makedirs(adjusted_results + 'real_A/')
os.makedirs(adjusted_results + 'real_B/')
os.makedirs(adjusted_results + 'fake_A/')
os.makedirs(adjusted_results + 'fake_B/')

# Definition of peak signal noise ratio
def psnr_function(img_a, img_b, modality): 
    mse = np.mean((img_a - img_b) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                # Therefore PSNR have no importance. 
        return 100.0
    if modality == 'mri2ct':
        max_pixel = 2800.0 # Maximum range
    elif modality == 'ct2mri':
        max_pixel = 2000.0
    psnr_val = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr_val 

def ssim_function(img_a, img_b, modality):
    if modality == 'mri2ct':
        img_a = ((img_a) + 800)
        img_b = ((img_b) + 800)
        return structural_similarity(img_a, img_b, data_range = 2800) 
    elif modality == 'ct2mri':
        return structural_similarity(img_a, img_b, data_range = 2000) 

# Definition of metrics
def getting_metrics(img_a, img_b, modality):
    # Getting MAE
    metric_mae = np.absolute(np.subtract(img_a, img_b)).mean()
    # Getting SSIM. It is necessary to set a distance between max and min value
    metric_ssim = ssim_function(img_a, img_b, modality)
    # Getting PSNR
    metric_psnr = psnr_function(img_a,img_b, modality)
    return metric_mae, metric_ssim, metric_psnr

def resize_volume(img,desired_depth = 134,desired_width = 207, desired_height = 226):

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

def remove_background(img_array):
    #Remove images with zero values in the mask
    non_zero_slices_mask_axis1_2 = np.any(img_array, axis=(1, 2))
    img_array = img_array[non_zero_slices_mask_axis1_2]
    non_zero_slices_mask_axis0_1 = np.any(img_array, axis=(0, 1))
    img_array = img_array[:,:,non_zero_slices_mask_axis0_1]
    non_zero_slices_mask_axis0_2 = np.any(img_array, axis=(0, 2))
    img_array = img_array[:,non_zero_slices_mask_axis0_2,:]
    return img_array

# Getting images
real_B = os.listdir(real_B_path)

# Define a padding style
pad_width = ((20, 20), (20, 20), (20, 20))  # 20 pixels of padding in each dimension

# Go for every image
for img_ in real_B:

    r_B, _ = load_nifti(real_B_path + '/' + img_)
    r_A, _ = load_nifti(real_A_path + '/' + img_)
    f_A, _ = load_nifti(fake_A_path + '/' + img_)
    f_B, _ = load_nifti(fake_B_path + '/' + img_)

    #Remove images with zero values in the mask
    r_B = remove_background(r_B)
    r_A = remove_background(r_A)
    f_A = remove_background(f_A)
    f_B = remove_background(f_B)
    
    # # Resizing
    r_B = resize_volume(r_B)
    r_A = resize_volume(r_A)
    f_A = resize_volume(f_A)
    f_B = resize_volume(f_B)

    # Pass to floats
    r_B = r_B.astype(np.float32)
    r_A = r_A.astype(np.float32)
    f_A = f_A.astype(np.float32)
    f_B = f_B.astype(np.float32)

    # We need to convert img_a to original ranges (radiation or HU)
    if args.stage == 'first':
        r_B = ((r_B* (2000 - (-800)))/ 255) + (-800)
        r_B = np.clip(r_B,a_min= -800, a_max= 2000)
        r_A = ((r_A* (2000 - (0)))/ 255) + (0)
        r_A = np.clip(r_A,a_min= 0, a_max= 2000)
        f_A = ((f_A* (2000 - (0)))/ 255) + (0)
        f_A = np.clip(f_A,a_min= 0, a_max= 2000)
        f_B = ((f_B* (2000 - (-800)))/ 255) + (-800)
        f_B = np.clip(f_B,a_min= -800, a_max= 2000)
    elif args.stage == 'second':
        r_B = ((r_B* (2000 - (-800)))/ 255) + (-800)
        r_B = np.clip(r_B,a_min= -800, a_max= 2000)
        r_A = ((r_A* (2000 - (-800)))/ 255) + (-800)
        r_A = np.clip(r_A,a_min= -800, a_max= 2000)
        f_B = ((f_B* (2000 - (-800)))/ 255) + (-800)
        f_B = np.clip(f_B,a_min= -800, a_max= 2000)

    #Padding
    if args.stage == 'first':
        r_B = np.pad(r_B, pad_width, mode='constant', constant_values=-800)
        r_A = np.pad(r_A, pad_width, mode='constant', constant_values=0)
        f_A = np.pad(f_B, pad_width, mode='constant', constant_values=0)
        f_B = np.pad(f_B, pad_width, mode='constant', constant_values=-800)
    elif args.stage == 'second':
        r_B = np.pad(r_B, pad_width, mode='constant', constant_values=-800)
        r_A = np.pad(r_A, pad_width, mode='constant', constant_values=-800)
        f_B = np.pad(f_B, pad_width, mode='constant', constant_values=-800)

    # Create a NIfTI image
    r_B = nib.Nifti1Image(r_B, affine=np.diag([1, 1, 1, 1]))
    r_A = nib.Nifti1Image(r_A, affine=np.diag([1, 1, 1, 1]))
    f_B = nib.Nifti1Image(f_B, affine=np.diag([1, 1, 1, 1]))
    if args.stage == 'first':
        f_A = nib.Nifti1Image(f_A, affine=np.diag([1, 1, 1, 1]))

    # Save the NIfTI image to a file
    nib.save(r_B, f'{adjusted_results}/real_B/{img_}.nii.gz')
    nib.save(r_A, f'{adjusted_results}/real_A/{img_}.nii.gz')
    nib.save(f_B, f'{adjusted_results}/fake_B/{img_}.nii.gz')
    if args.stage == 'first':
        nib.save(f_A, f'{adjusted_results}/fake_A/{img_}.nii.gz')


if args.stage == 'first':

    # Getting the list of output files for evaluation for CT
    real_class = 'real_B'
    real_files = os.listdir(adjusted_results + real_class)
    real_files = sorted(real_files)

    # Results MRI2CT
    for type in ['3d']:
            
        for fake_class in ['fake_B']:

            # Set the metrics to zero   
            mae = 0
            ssim = 0
            psnr = 0

            if type == '2d-sagittal':
                mae_per_slice = []
                ssim_per_slice = []
                psnr_per_slice = []

                # Getting the metrics for images set B
                for path_img in real_files:
                
                    # Read images
                    img_a, _ = load_nifti(adjusted_results + real_class + '/' + path_img)
                    img_b, _ = load_nifti(adjusted_results + fake_class + '/' + path_img)

                    # Iterate over each slice of the volume
                    for slice in range(img_a.shape[0]):
                        # Compute metrics for the current slice
                        mae_val, ssim_val, psnr_val = getting_metrics(img_a[slice],img_b[slice], modality= "mri2ct")

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
                for path_img in real_files:
                
                    # Read images
                    img_a, _ = load_nifti(adjusted_results + real_class + '/' + path_img)
                    img_b, _ = load_nifti(adjusted_results + fake_class + '/' + path_img)

                    # Iterate over each slice of the volume
                    for slice in range(img_a.shape[1]):
                        # Compute metrics for the current slice
                        mae_val, ssim_val, psnr_val = getting_metrics(img_a[:,slice,:],img_b[:,slice,:], modality= "mri2ct")

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
                for path_img in real_files:
                    
                    # Read images
                    img_a, _ = load_nifti(adjusted_results + real_class + '/' + path_img)
                    img_b, _ = load_nifti(adjusted_results + fake_class + '/' + path_img)

                    # Iterate over each slice of the volume
                    for slice in range(img_a.shape[2]):
                        # Compute metrics for the current slice
                        mae_val, ssim_val, psnr_val = getting_metrics(img_a[:,:,slice],img_b[:,:,slice], modality= "mri2ct")

                        # Append metrics to lists
                        mae_per_slice.append(mae_val)
                        ssim_per_slice.append(ssim_val)
                        psnr_per_slice.append(psnr_val)

                    mae += np.mean(np.array(mae_per_slice))
                    ssim += np.mean(np.array(ssim_per_slice))
                    psnr += np.mean(np.array(psnr_per_slice))

            elif type == '3d':

                # Getting the metrics for images set B
                for path_img in real_files:
                
                    # Read images
                    img_a, _ = load_nifti(adjusted_results + real_class + '/' + path_img)
                    img_b, _ = load_nifti(adjusted_results + fake_class + '/' + path_img)

                    # Calculate metrics
                    mae_val, ssim_val, psnr_val = getting_metrics(img_a,img_b, modality= "mri2ct")
                    # Adding values
                    mae += mae_val
                    ssim += ssim_val
                    psnr += psnr_val

            print(f"Results in comparison {real_class} with {fake_class} ({args.stage}) - {type}")
            print('MAE: ', mae/len(real_files))
            print('SSIM: ', ssim/len(real_files))
            print('PSNR: ', psnr/len(real_files))
        print("===========================================================")

    
    # Results CT2MRI

    # Getting the list of output files for evaluation for CT
    real_class = 'real_A'
    real_files = os.listdir(adjusted_results + real_class)
    real_files = sorted(real_files)


    for type in ['3d']:
            
        for fake_class in ['fake_A']:

            # Set the metrics to zero   
            mae = 0
            ssim = 0
            psnr = 0

            if type == '2d-sagittal':
                mae_per_slice = []
                ssim_per_slice = []
                psnr_per_slice = []

                # Getting the metrics for images set B
                for path_img in real_files:
                
                    # Read images
                    img_a, _ = load_nifti(adjusted_results + real_class + '/' + path_img)
                    img_b, _ = load_nifti(adjusted_results + fake_class + '/' + path_img)

                    # Iterate over each slice of the volume
                    for slice in range(img_a.shape[0]):
                        # Compute metrics for the current slice
                        mae_val, ssim_val, psnr_val = getting_metrics(img_a[slice],img_b[slice], modality= "ct2mri")

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
                for path_img in real_files:
                
                    # Read images
                    img_a, _ = load_nifti(adjusted_results + real_class + '/' + path_img)
                    img_b, _ = load_nifti(adjusted_results + fake_class + '/' + path_img)

                    # Iterate over each slice of the volume
                    for slice in range(img_a.shape[1]):
                        # Compute metrics for the current slice
                        mae_val, ssim_val, psnr_val = getting_metrics(img_a[:,slice,:],img_b[:,slice,:], modality= "ct2mri")

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
                for path_img in real_files:
                    
                    # Read images
                    img_a, _ = load_nifti(adjusted_results + real_class + '/' + path_img)
                    img_b, _ = load_nifti(adjusted_results + fake_class + '/' + path_img)

                    # Iterate over each slice of the volume
                    for slice in range(img_a.shape[2]):
                        # Compute metrics for the current slice
                        mae_val, ssim_val, psnr_val = getting_metrics(img_a[:,:,slice],img_b[:,:,slice], modality= "ct2mri")

                        # Append metrics to lists
                        mae_per_slice.append(mae_val)
                        ssim_per_slice.append(ssim_val)
                        psnr_per_slice.append(psnr_val)

                    mae += np.mean(np.array(mae_per_slice))
                    ssim += np.mean(np.array(ssim_per_slice))
                    psnr += np.mean(np.array(psnr_per_slice))

            elif type == '3d':

                # Getting the metrics for images set B
                for path_img in real_files:
                
                    # Read images
                    img_a, _ = load_nifti(adjusted_results + real_class + '/' + path_img)
                    img_b, _ = load_nifti(adjusted_results + fake_class + '/' + path_img)

                    # Calculate metrics
                    mae_val, ssim_val, psnr_val = getting_metrics(img_a,img_b, modality= "ct2mri")
                    # Adding values
                    mae += mae_val
                    ssim += ssim_val
                    psnr += psnr_val

            print(f"Results in comparison {real_class} with {fake_class} ({args.stage} stage) - {type}")
            print('MAE: ', mae/len(real_files))
            print('SSIM: ', ssim/len(real_files))
            print('PSNR: ', psnr/len(real_files))
        print("===========================================================")


elif args.stage == 'second':

    # Getting the list of output files for evaluation
    real_class = 'real_B'
    real_files = os.listdir(adjusted_results + real_class)
    real_files = sorted(real_files)


    for type in ['3d']:
            
        for fake_class in ['real_A', 'fake_B']:

            # Set the metrics to zero   
            mae = 0
            ssim = 0
            psnr = 0

            if type == '2d-sagittal':
                mae_per_slice = []
                ssim_per_slice = []
                psnr_per_slice = []

                # Getting the metrics for images set B
                for path_img in real_files:
                
                    # Read images
                    img_a, _ = load_nifti(adjusted_results + real_class + '/' + path_img)
                    img_b, _ = load_nifti(adjusted_results + fake_class + '/' + path_img)

                    # Iterate over each slice of the volume
                    for slice in range(img_a.shape[0]):
                        # Compute metrics for the current slice
                        mae_val, ssim_val, psnr_val = getting_metrics(img_a[slice],img_b[slice],modality= "mri2ct")

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
                for path_img in real_files:
                
                    # Read images
                    img_a, _ = load_nifti(adjusted_results + real_class + '/' + path_img)
                    img_b, _ = load_nifti(adjusted_results + fake_class + '/' + path_img)

                    # Iterate over each slice of the volume
                    for slice in range(img_a.shape[1]):
                        # Compute metrics for the current slice
                        mae_val, ssim_val, psnr_val = getting_metrics(img_a[:,slice,:],img_b[:,slice,:],modality= "mri2ct")

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
                for path_img in real_files:
                    
                    # Read images
                    img_a, _ = load_nifti(adjusted_results + real_class + '/' + path_img)
                    img_b, _ = load_nifti(adjusted_results + fake_class + '/' + path_img)

                    # Iterate over each slice of the volume
                    for slice in range(img_a.shape[2]):
                        # Compute metrics for the current slice
                        mae_val, ssim_val, psnr_val = getting_metrics(img_a[:,:,slice],img_b[:,:,slice],modality= "mri2ct")

                        # Append metrics to lists
                        mae_per_slice.append(mae_val)
                        ssim_per_slice.append(ssim_val)
                        psnr_per_slice.append(psnr_val)

                    mae += np.mean(np.array(mae_per_slice))
                    ssim += np.mean(np.array(ssim_per_slice))
                    psnr += np.mean(np.array(psnr_per_slice))

            elif type == '3d':

                # Getting the metrics for images set B
                for path_img in real_files:
                
                    # Read images
                    img_a, _ = load_nifti(adjusted_results + real_class + '/' + path_img)
                    img_b, _ = load_nifti(adjusted_results + fake_class + '/' + path_img)

                    # Calculate metrics
                    mae_val, ssim_val, psnr_val = getting_metrics(img_a,img_b,modality= "mri2ct")
                    # Adding values
                    mae += mae_val
                    ssim += ssim_val
                    psnr += psnr_val

            if fake_class == 'real_A':
                text = f"Real A comparison {type}"
            elif fake_class == 'fake_B':
                text = f"Fake B comparison {type}"

            print('')
            print(f'Results in {text}')
            print('MAE: ', mae/len(real_files))
            print('SSIM: ', ssim/len(real_files))
            print('PSNR: ', psnr/len(real_files))
        print("===========================================================")



