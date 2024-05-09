import numpy as np
import os
import shutil
import dipy
from dipy.io.image import load_nifti, save_nifti
from dipy.align import affine_registration
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

# Loading images 
input_path = 'outcome_averaging/'
input_files = os.listdir(input_path)
output_path = 'outcome_after_registration'

# Remove output folder if already exists
if os.path.exists(output_path):
    shutil.rmtree(output_path)
# Create output folder
os.makedirs(output_path)

# Loading ids
file_ids = [img[6:9] for img in input_files if 'sct_a_' in img]

pipeline = ["center_of_mass", "translation"]

level_iters = [10]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]

# Going for each pairs of images:
for img in file_ids:

        sct_a_path = [file for file in input_files if 'sct_a_' + img in file][0]
        sct_c_path = [file for file in input_files if 'sct_c_' + img in file][0]
        
        sct_a, sct_a_affine = load_nifti(input_path + sct_a_path)
        sct_c, sct_c_affine = load_nifti(input_path + sct_c_path)

        xformed_img, reg_affine = affine_registration(
        sct_a,
        sct_c,
        moving_affine=sct_a_affine,
        static_affine=sct_c_affine,
        nbins=32,
        metric='MI',
        pipeline=pipeline,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors)

        save_nifti(f'{output_path}/sct_ta_{img}.nii.gz', xformed_img, reg_affine)
        save_nifti(f'{output_path}/sct_a_{img}.nii.gz', sct_a, sct_a_affine)
        save_nifti(f'{output_path}/sct_c_{img}.nii.gz', sct_c, sct_c_affine)

        print(f'Transforming img {img}')


