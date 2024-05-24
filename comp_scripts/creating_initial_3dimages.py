# Import libraries
import nibabel as nib
import numpy as np
from PIL import Image
import os
import argparse
import shutil

# Extract the numeric part of images of unpaired images
def extract_numeric_part_unpaired(file_name):
    return int(file_name.split('_')[1])


path_3d_images_initial = '../../data/data-3d-unsupervised-fixed/'
path_2d_images_initial = '../../data/data-2d-unsupervised/'

# Remove output folder if already exists
if os.path.exists(path_3d_images_initial):
    shutil.rmtree(path_3d_images_initial)
# Create output folder
os.makedirs(path_3d_images_initial)

for type_ in [
              'train_maskA',
              'train_A',
              'train_B',
              'train_maskB',
              'val_A',
              'val_maskA', 
              'val_B',
              'val_maskB',
              'test_A',
              'test_maskA',
              'test_B',
              'test_maskB']:

    # Define main subdirectories
    folder = path_2d_images_initial + '/' + type_ + '/'

    # Obtain the images
    images = os.listdir(folder)

    # Create output folder
    os.makedirs(path_3d_images_initial + '/' + type_)

    # Getting base names
    base_images = np.unique([img[0:3] for img in images])

    # Create voxels 1 by 1
    for voxel_idx in base_images:

        slicing_image = [img for img in images if img[0:3] == voxel_idx]
        # Sort the list using the custom sorting key
        sorted_slicing_names = sorted(slicing_image, key=extract_numeric_part_unpaired)

        # Check how many images I have per voxel
        images_voxel = len(sorted_slicing_names)

        # Concatenate 2D Slices
        for idx, img in enumerate(sorted_slicing_names):

            # Open a PNG image
            image = Image.open(path_2d_images_initial + '/' + type_ + '/' + img)

            # Convert the image to a numpy array
            image_array = np.array(image).astype(np.float32)
            image_array = image_array.reshape(1,224,224)

            if '_A' in type_:
                s = 0 

            image_array = image_array / 255

            # Incorporate slices in the template
            if idx == 0:
                image3d = image_array
            else:
                image3d = np.concatenate((image3d,image_array), axis = 0)

        # Converting to original values 
        if '_A' in type_:
            image3d = image3d * (2048 - 0) + 0
        elif '_B' in type_:
            image3d = image3d * (2000 - (-800)) + (-800)    

        # Create a NIfTI image
        nifti_img = nib.Nifti1Image(image3d, affine=np.diag([1, 1, 1, 1]))  # Assuming an identity affine matrix

        # Save the NIfTI image to a file
        print('saving image ', path_3d_images_initial  + type_ + '/' + voxel_idx)
        nib.save(nifti_img, path_3d_images_initial  + type_ + '/' + voxel_idx +'.nii.gz')
    


