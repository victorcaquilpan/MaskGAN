"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_intermediate_images_twostages
from util import html
import glob
import shutil
import torch

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # Remove output folder if already exists
    if os.path.exists(f'{opt.results_dir}/{opt.name}'):
        shutil.rmtree(f'{opt.results_dir}/{opt.name}')
    # Create output folder
    os.makedirs(f'{opt.results_dir}/{opt.name}')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    print(web_dir)
    real_B_path = f'{opt.results_dir}/{opt.name}/real_B'
    real_A_path = f'{opt.results_dir}/{opt.name}/real_A'
    fake_B_path = f'{opt.results_dir}/{opt.name}/fake_B'
    fake_A_path = f'{opt.results_dir}/{opt.name}/fake_A'
    out_mae = f'{opt.results_dir}/{opt.name}/MAE'
    os.makedirs(real_B_path, exist_ok=True)
    os.makedirs(real_A_path, exist_ok=True)
    os.makedirs(fake_B_path, exist_ok=True)
    os.makedirs(fake_A_path, exist_ok=True)
    os.makedirs(out_mae, exist_ok=True)

    # Define the model for evaluation purposes
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if opt.num_test > 0 and i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        visuals['MAE'] = torch.abs(visuals['real_B'] - visuals['fake_B'])

        # Get image paths
        if opt.dataset_mode == 'unaligned_chunks':
            img_path_A = data['A_paths'][0]     
            img_path_B = data['B_paths'][0]
        elif opt.dataset_mode == 'unaligned':
            img_path_A = data['A_paths']   
            img_path_B = data['B_paths']
        img_path = model.get_image_paths()     # get image paths
        # if i % 5 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th image... %s' % (i, img_path))

        for idx_batch in range(0,visuals['real_A'].shape[0]):
                visuals_real_A = {'real_A' : visuals['real_A'][idx_batch,:,:].unsqueeze(1)}
                visuals_real_B = {'real_B' : visuals['real_B'][idx_batch,:,:].unsqueeze(1)}
                visuals_fake_A = {'fake_A': visuals['fake_A'][idx_batch,:,:].unsqueeze(1)}
                visuals_fake_B = {'fake_B': visuals['fake_B'][idx_batch,:,:].unsqueeze(1)}
                save_intermediate_images_twostages(webpage, visuals_real_A, img_path_A[idx_batch], aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
                save_intermediate_images_twostages(webpage, visuals_real_B, img_path_B[idx_batch], aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
                save_intermediate_images_twostages(webpage, visuals_fake_A, img_path_A[idx_batch], aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
                save_intermediate_images_twostages(webpage, visuals_fake_B, img_path_B[idx_batch], aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()  # save the HTML
    print("Move images")
    ### Save results
    for filepath in glob.glob(f'{web_dir}/images/*fake_A*'):
        filename = os.path.basename(filepath).replace("_fake_A","")
        shutil.copy(filepath, f'{fake_A_path}/{filename}')
    for filepath in glob.glob(f'{web_dir}/images/*real_A*'):
        filename = os.path.basename(filepath).replace("_real_A","")
        shutil.copy(filepath, f'{real_A_path}/{filename}')
    for filepath in glob.glob(f'{web_dir}/images/*fake_B*'):
        filename = os.path.basename(filepath).replace("_fake_B","")
        shutil.copy(filepath, f'{fake_B_path}/{filename}')
    for filepath in glob.glob(f'{web_dir}/images/*real_B*'):
        filename = os.path.basename(filepath).replace("_real_B","")
        shutil.copy(filepath, f'{real_B_path}/{filename}')
    # Remove temporal folder
    shutil.rmtree(web_dir)
        



