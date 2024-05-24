import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
from torchvision import io
from monai.transforms import Rand2DElasticd
import pandas as pd

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + f'_{opt.Aclass}')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + f'_{opt.Bclass}')  # create a path '/path/to/data/trainB'
        self.dir_maskA = os.path.join(opt.dataroot, opt.phase + f'_mask{opt.Aclass}')  # create a path '/path/to/data/trainA'
        self.dir_maskB = os.path.join(opt.dataroot, opt.phase + f'_mask{opt.Bclass}')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        # Define if you want to include paired images
        self.include_paired_images = opt.include_paired_images
        if not self.include_paired_images:
            self.A_paths = [img for img in self.A_paths if 'paired'not in img]
            self.B_paths = [img for img in self.B_paths if 'paired'not in img]

        self.maskA_paths = sorted(make_dataset(self.dir_maskA, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.maskB_paths = sorted(make_dataset(self.dir_maskB, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = opt.direction == 'BtoA'
        if opt.half:
            self.A_size, self.B_size = self.A_size//2, self.B_size//2
            self.A_paths, self.B_paths = self.A_paths[:self.A_size], self.B_paths[self.B_size:]
        
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(self.input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(self.output_nc == 1))

        # Check how many 3D images we are considering
        A_paths_main_img = [img.split("/")[-1] for img in self.A_paths]
        A_paths_main_img = [img.split("_")[0] for img in A_paths_main_img]
        self.number_3d_images_A = len(np.unique(A_paths_main_img))
        # Same for set B
        B_paths_main_img = [img.split("/")[-1] for img in self.B_paths]
        B_paths_main_img = [img.split("_")[0] for img in B_paths_main_img]        
        self.number_3d_images_B = len(np.unique(B_paths_main_img))

        # Save relative position of each img
        self.relative_pos_A = [int(img.split(".")[-2].split("_")[-1]) for img in self.A_paths]
        self.relative_pos_B = [int(img.split(".")[-2].split("_")[-1]) for img in self.B_paths]

        # Extract the name of images
        self.image_names_A = [name.split('/')[-1].replace('.png','') for name in self.A_paths]
        self.image_names_B = [name.split('/')[-1].replace('.png','') for name in self.B_paths]

        # Create an auxiliar list to determine if an image is paired or not
        if not self.include_paired_images:
            self.paired_imgs_A = ['paired' in img for img in self.image_names_A]
            self.paired_imgs_B = ['paired' in img for img in self.image_names_B]

        # List values paired/unpaired for A set
        if not self.include_paired_images:
            self.idx_paired_A = [idx for idx, paired in enumerate(self.paired_imgs_A) if paired == True]
            self.idx_unpaired_A = [idx for idx, paired in enumerate(self.paired_imgs_A) if paired != True]

        # Extract the name of the base image
        self.base_names_A = [img.split("_")[1] if 'paired' in img else img.split("_")[0] for img in self.image_names_A]
        self.base_names_B = [img.split("_")[1] if 'paired' in img else img.split("_")[0] for img in self.image_names_B]

        # Set a margin to use images from a similar relative position
        if self.opt.phase == 'train':
            self.position_based_range = opt.position_based_range # This is a percentage
            # Define the age range    
            self.range_months = opt.range_months

        # To force testing over train set
        try:
            self.force_testing = opt.force_testing
        except:
            self.force_testing = False

        if self.opt.phase != 'test' and not self.force_testing and self.opt.feature_images_file_path != '':
            # Loading the CSV file
            self.feature_images = pd.read_csv(opt.feature_images_file_path)
            # Remove duplicate rows
            self.feature_images = self.feature_images.drop_duplicates()
            # Getting the age of each CT and MRI
            self.ages_images_A = []
            self.ages_images_B = []
            for img in self.base_names_A:
                age = self.feature_images.loc[(self.feature_images['new_name'] == float(img)) & (self.feature_images['Modality'] == "MR"), "PatientAgeMonths"].values
                if age.size > 0:
                    self.ages_images_A.append(age[0])
            for img in self.base_names_B:
                age = self.feature_images.loc[(self.feature_images['new_name'] == float(img)) & (self.feature_images['Modality'] == "CT"), "PatientAgeMonths"].values
                if age.size > 0:
                    self.ages_images_B.append(age[0])
            
        # Define MONAI transforms to augment paired images
        self.transform_paired = Rand2DElasticd(keys = ['mri','ct', 'mri_mask', 'ct_mask'],
            prob=0.5,
            spacing=(100, 100),
            magnitude_range=(1, 2),
            rotate_range=(0.2),
            scale_range=(0.2, 0.2),
            padding_mode="zeros")
        
        # Store the phase
        self.phase = opt.phase
        
        
    def __getitem__(self, index):

        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # If we have paired data in our set, we want to be sure that we are picking up a ratio of 50:50. Otherwise, don't consider this condition
        if self.opt.phase != 'test' and not self.force_testing and len(self.idx_paired_A) > 0  and self.include_paired_images:
            # We define a ratio 50:50 paired:unpaired
            paired = random.choice((True, False))
            if paired:
                index_A = random.choice(self.idx_paired_A)
            else:
                index_A = random.choice(self.idx_unpaired_A)
        else:
            paired = False
            index_A = index % self.A_size
        # Define image from A set
        A_path = self.A_paths[index_A]
        
        # Define B image
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            # Check the relative position of the image (Position based selection PBS)
            A_path_spplited = A_path.split(".")
            A_relative_position = A_path_spplited[-2].split("_")[-1]
            # Convert to a number
            A_relative_position = float(A_relative_position)
            # Get the age of that image
            if self.opt.feature_images_file_path != '':
                A_months = self.ages_images_A[index_A]
            # Obtain the images in a similar range (Position based selection)
            potential_indexes = [index for index, value in enumerate(self.relative_pos_B) if (A_relative_position-self.position_based_range) <= value <= (A_relative_position + self.position_based_range)]
            # Select images which are in a similar age
            if self.opt.feature_images_file_path != '':
                potential_indexes_months = [index for index, value in enumerate(self.ages_images_B) if (A_months-self.range_months) <= value <= (A_months + self.range_months)]
            # Considering inclusion of paired dataset, we need to select images from the paired CT scan
            #paired_img = self.paired_imgs_A[index_A]
            if paired:
                base_img = self.base_names_A[index_A]
                potential_indexes_paired = [idx for idx, img in enumerate(self.base_names_B) if base_img == img]
                potential_indexes = list(set(potential_indexes) & set(potential_indexes_paired))

            # Define position of B image
            if self.opt.feature_images_file_path != '':
                potential_indexes = list(set(potential_indexes) & set(potential_indexes_months))
            index_position = random.randint(0, len(potential_indexes) - 1)
            index_B = potential_indexes[index_position]
            
        # Selecting the B image
        B_path = self.B_paths[index_B]
        # Loading masks
        maskA_path = self.maskA_paths[index_A]
        maskB_path = self.maskB_paths[index_B]

        A_img = io.read_image(A_path)
        B_img = io.read_image(B_path)
    
        A_mask = io.read_image(maskA_path)
        B_mask = io.read_image(maskB_path)

        # apply data augmentation to paired images
        if paired:
            #mri_augmented_base = self.transform_paired({'mri': torch.tensor(mri), 'ct': torch.tensor(ct), 'mri_mask' : torch.tensor(mri_mask), 'ct_mask': torch.tensor(ct_mask)})
            paired_images = self.transform_paired({'mri': A_img, 'ct': B_img, 'mri_mask' : A_mask, 'ct_mask': B_mask})
            A_img = paired_images['mri']
            A_mask = paired_images['mri_mask']
            B_img = paired_images['ct']
            B_mask = paired_images['ct_mask']

        # apply image transformation to standarize data
        A, A_mask = self.transform_A(A_img, A_mask)
        B, B_mask = self.transform_B(B_img, B_mask)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path,
        'A_mask': A_mask, 'B_mask': B_mask}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if self.phase == 'train':
            #return max(self.A_size, self.B_size)
            return self.A_size 
        else:
            return self.A_size