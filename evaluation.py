# Loading libraries
import PIL.Image
import numpy as np
import PIL
import argparse
import os
from skimage.metrics import structural_similarity
from math import log10, sqrt 

# Loading data
parser = argparse.ArgumentParser()
parser.add_argument('--results', default='results/paired_images', type=str, help='directory of results')
opt = parser.parse_args()

# Reading images
real_A_paths = os.listdir(opt.results + '/real_A/')
real_B_paths = os.listdir(opt.results + '/real_B/')
fake_A_paths = os.listdir(opt.results + '/fake_A/')
fake_B_paths = os.listdir(opt.results + '/fake_B/')

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

# Set the metrics to zero
mae = 0
ssim = 0
psnr = 0

# Getting the metrics for images set B
for idx, path in enumerate(fake_B_paths):

    # Obtain the proper paths
    path_b = path
    path_a = path.replace('fake', 'real')
    
    # Read images
    img_a = PIL.Image.open(opt.results + '/real_B/' + path_a)
    img_b = PIL.Image.open(opt.results + '/fake_B/' + path_b)

    # Pass to numpy arrays
    img_a = np.array(img_a).astype(np.float32)
    img_b = np.array(img_b).astype(np.float32)

    # We need to convert img_a to HU
    img_a = ((img_a* (2000 - (-800)))/ 255) + (-800)
    img_a = np.clip(img_a,a_min= -800, a_max= 2000)

    # Now, we need to convert img_b to HU
    img_b = ((img_b* (2000 - (-800)))/ 255) + (-800)
    img_b = np.clip(img_b,a_min= -800, a_max= 2000)

    s = 0 

    # Calculate metrics
    mae_val, ssim_val, psnr_val = getting_metrics(img_a,img_b)
    # Adding values
    mae += mae_val
    ssim += ssim_val
    psnr += psnr_val

print('MAE: ', mae/len(real_A_paths))
print('SSIM: ', ssim/len(real_A_paths))
print('PSNR: ', psnr/len(real_A_paths))









