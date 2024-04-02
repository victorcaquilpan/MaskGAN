# Import libraries
import nibabel as nib
import numpy as np
from PIL import Image
import os
import argparse
import shutil

# Call the parser
parser = argparse.ArgumentParser()
parser.add_argument('--intermediate_results_folder', default='twostage_intermediate_results/semi-supervised-cleaned-paired-new-mask/', help='path to intermediate results (1rst stage)')
parser.add_argument('--intermediate_voxels_folder', default='../../data/two_stage_approach/intermediate_voxels/semi-supervised-cleaned-paired-imgs/', help='path to leave input of 2nd stage')
parser.add_argument('--size_input', default=224, help='length of sides of input image')
args = parser.parse_args()

# Extract the numeric part of images of unpaired images
def extract_numeric_part_unpaired(file_name):
    return int(file_name.split('_')[1])

# Extract the numeric part of images of paired images
def extract_numeric_part_paired(file_name):
    return int(file_name.split('_')[2])

# Remove output folder if already exists
if os.path.exists(args.intermediate_voxels_folder):
    shutil.rmtree(args.intermediate_voxels_folder)
# Create output folder
os.makedirs(args.intermediate_voxels_folder)

for phase in ['train', 'val', 'test']:

    # Define main subdirectories
    sct_folder = args.intermediate_results_folder + phase +'/'

    # Obtain the images
    sct_images = os.listdir(sct_folder)

    # Remove output folder if already exists
    if os.path.exists(args.intermediate_voxels_folder + phase):
        shutil.rmtree(args.intermediate_voxels_folder + phase)
    # Create output folder
    os.makedirs(args.intermediate_voxels_folder + phase)

    # Group images
    if phase == 'train':
        voxels_pairs_idx = np.unique([img[0:10] for img in sct_images if 'paired' in img])
    voxels_idx = np.unique([img[0:3] for img in sct_images if 'paired' not in img])

    # Merge idxs
    voxels_idx = np.concatenate((voxels_pairs_idx,voxels_idx))

    # Create voxels 1 by 1
    for voxel_idx in voxels_idx:

        # Concatenate images
        if 'paired' in voxel_idx:
            slicing_image = [img for img in sct_images if img[0:10] == voxel_idx]
            # Sort the list using the custom sorting key
            sorted_slicing_names = sorted(slicing_image, key=extract_numeric_part_paired)
        else:
            slicing_image = [img for img in sct_images if img[0:3] == voxel_idx]
            # Sort the list using the custom sorting key
            sorted_slicing_names = sorted(slicing_image, key=extract_numeric_part_unpaired)

        # Create a template array, which is cubic
        #image3d= np.zeros((args.size_input,args.size_input,args.size_input))

        # Check how many images I have per voxel
        images_voxel = len(sorted_slicing_names)

        # To ensure to insert the image in the middle of the template, we need to identify from which index, we need to include the images
        index_insert_brain = (args.size_input - images_voxel) // 2 

        # Concatenate 2D Slices
        for idx, img in enumerate(sorted_slicing_names):

            # Open a PNG image
            image_path = sct_folder + img
            image = Image.open(image_path)

            # Convert the image to a numpy array
            image_array = np.array(image)
            image_array = image_array.reshape(1,args.size_input,args.size_input)

            # Incorporate slices in the template
            if idx == 0:
                image3d = image_array
            else:
                image3d = np.concatenate((image3d,image_array), axis = 0)

            #image3d[index_insert_brain+idx, :, :] =  image_array

        # Create a NIfTI image
        nifti_img = nib.Nifti1Image(image3d, affine=np.diag([1, 1, 1, 1]))  # Assuming an identity affine matrix

        # Save the NIfTI image to a file
        print('saving image ', phase + '/' + voxel_idx)
        nib.save(nifti_img, args.intermediate_voxels_folder + phase + '/' + voxel_idx +'.nii.gz')
        


