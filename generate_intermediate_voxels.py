# Import libraries
import nibabel as nib
import numpy as np
from PIL import Image
import os
import argparse
import shutil

# Call the parser
parser = argparse.ArgumentParser()
parser.add_argument('--intermediate_2d_images', default='../../data/intermediate_2d_images/maskgan_firststage_adult/', help='path to intermediate results (1rst stage)')
parser.add_argument('--intermediate_3d_images', default='../../data/intermediate_3d_images/maskgan_firststage_adult/', help='path to leave input of 2nd stage')
parser.add_argument('--size_input', default=224, help='length of sides of input image')
args = parser.parse_args()

# Extract the numeric part of images of unpaired images
def extract_numeric_part_unpaired(file_name):
    return int(file_name.split('_')[1])

# Remove output folder if already exists
if os.path.exists(args.intermediate_3d_images):
    shutil.rmtree(args.intermediate_3d_images)
# Create output folder
os.makedirs(args.intermediate_3d_images)

for phase in ['train', 'val', 'test']:
    # Create output folder
    os.makedirs(args.intermediate_3d_images + phase)
    
    for type in ['real_A', 'real_B', 'fake_A', 'fake_B']:

        # Define main subdirectories
        image_folder = args.intermediate_2d_images + phase + '/' + type + '/'

        # Obtain the images
        images = os.listdir(image_folder)

        # Create output folder
        os.makedirs(args.intermediate_3d_images + phase +'/' + type)

        # Identify the voxels id
        voxels_idx = np.unique([img[0:3] for img in images])
            
        # Create voxels 1 by 1
        for voxel_idx in voxels_idx:

            # Concatenate images
            slicing_image = [img for img in images if img[0:3] == voxel_idx]
            # Sort the list using the custom sorting key
            sorted_slicing_names = sorted(slicing_image, key=extract_numeric_part_unpaired)
            # Check how many images I have per voxel
            images_voxel = len(sorted_slicing_names)

            # Concatenate 2D Slices
            for idx, img in enumerate(sorted_slicing_names):

                # Open a PNG image
                image_path = image_folder + img
                image = Image.open(image_path)

                # Convert the image to a numpy array
                image_array = np.array(image).astype(np.float32)

                # Conversion to original values
                if 'B' in type:
                    image_array = ((image_array* (2000 - (-800)))/ 255) + (-800)
                    image_array = np.clip(image_array,a_min= -800, a_max= 2000)
                    image_array = image_array.reshape(1,args.size_input,args.size_input)
                elif 'A' in type:
                    image_array = ((image_array* (2000 - (0)))/ 255) + (0)
                    image_array = np.clip(image_array,a_min= 0, a_max= 2000)
                    image_array = image_array.reshape(1,args.size_input,args.size_input)

                # Incorporate slices in the template
                if idx == 0:
                    image3d = image_array
                else:
                    image3d = np.concatenate((image3d,image_array), axis = 0)

            # Rotate to get the proper orientation
            image3d = np.rot90(image3d, axes = (2, 1))
            image3d = np.flip(image3d, axis = 0)

            # Create a NIfTI image
            nifti_img = nib.Nifti1Image(image3d, affine=np.diag([1, 1, 1, 1]))  # Assuming an identity affine matrix

            # Save the NIfTI image to a file
            print('saving image ', phase + '/' + type + '/' + voxel_idx)
            nib.save(nifti_img, args.intermediate_3d_images + phase + '/'+ type + '/' + voxel_idx +'.nii.gz')
        


