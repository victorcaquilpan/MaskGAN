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
from util.visualizer import save_images, save_intermediate_images_twostages
from util import html
import glob
import shutil
import torch

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.results_dir = 'twostage_intermediate_results'

    # Create results folder if there is not already exist
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)

    # Create or overwrite specific results folder
    if os.path.exists(opt.results_dir + '/' + opt.name):
        shutil.rmtree(opt.results_dir + '/' + opt.name)
    # Create output folder
    os.makedirs(opt.results_dir + '/' + opt.name)

    # Define the inference for all the phases (train, val and test)
    for phase in ['train','val', 'test']:    
        # We are considering serial batches
        opt.phase = phase
        opt.serial_batches = True
        # To force testing over train, val and test sets
        opt.force_testing = True
        opt.include_paired_images = False
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        # create a website
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % ('temporal', opt.epoch))  # define the website directory
        if os.path.exists(web_dir):
            shutil.rmtree(web_dir)
        os.makedirs(web_dir, exist_ok=True)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        print(web_dir)
        #out_ct = f'{opt.results_dir}/{opt.name}/{opt.phase}/real_B'
        #out_mri = f'{opt.results_dir}/{opt.name}/{opt.phase}/real_A'
        out_fake_ct = f'{opt.results_dir}/{opt.name}/{opt.phase}'
        #out_fake_mri = f'{opt.results_dir}/{opt.name}/{opt.phase}/fake_A'
        #out_mae = f'{opt.results_dir}/{opt.name}/{opt.phase}/MAE'
        #os.makedirs(out_ct, exist_ok=True)
        #os.makedirs(out_mri, exist_ok=True)
        if os.path.exists(out_fake_ct):
            shutil.rmtree(out_fake_ct)
        os.makedirs(out_fake_ct, exist_ok=True)
        os.makedirs(out_fake_ct + '/real_B', exist_ok=True)
        os.makedirs(out_fake_ct + '/fake_B', exist_ok=True)
        #os.makedirs(out_fake_mri, exist_ok=True)
        #os.makedirs(out_mae, exist_ok=True)
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        if opt.eval:
            model.eval()
        for i, data in enumerate(dataset):
            if opt.num_test > 0 and i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            visuals_real = {'real_B' : visuals['real_B']}
            visuals_fake = {'fake_B': visuals['fake_B']}
            #visuals = {'fake_B': visuals['fake_B']}
            #visuals = visuals['fake_B']
            ## Fix background CT
            #mask = visuals['mask_A']
            #visuals['real_B'][mask == 1] = -1
            #visuals['MAE'] = torch.abs(visuals['real_B'] - visuals['fake_B'])
            
            # Get image paths
            img_path_A = data['A_paths']     
            img_path_B = data['B_paths']
            
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path_A))
            save_intermediate_images_twostages(webpage, visuals_fake, img_path_A, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            save_intermediate_images_twostages(webpage, visuals_real, img_path_B, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        webpage.save()  # save the HTML
        print("Move images")
        ### Save results
        for filepath in glob.glob(f'{web_dir}/images/*fake_B*'):
            filename = os.path.basename(filepath)
            shutil.copy(filepath, f'{out_fake_ct}/fake_B/{filename}')
        for filepath in glob.glob(f'{web_dir}/images/*real_B*'):
            filename = os.path.basename(filepath)
            shutil.copy(filepath, f'{out_fake_ct}/real_B/{filename}')
        # Remove temporal folder
        shutil.rmtree(web_dir)
        

