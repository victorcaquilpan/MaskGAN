"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.visual_validation import validation # type: ignore
import torch
import wandb
import copy
import time

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    if opt.report_wandb:
        wandb.init(project="maskgan2D", name=opt.name) 
    train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    val_opt = copy.deepcopy(opt)
    val_opt.phase = 'val'
    val_opt.serial_batches = True
    val_dataset = create_dataset(val_opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    best = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.lambda_shape = (epoch/(opt.niter + opt.niter_decay + 1))*model.opt.lambda_shape

        model.train()
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()

            if opt.stage == 'first':
                total_iters += opt.batch_size 
                epoch_iter += opt.batch_size

            # Transform the input to treat each slice separately
            elif opt.stage == 'second':
                total_iters += opt.batch_size * opt.n_slices 
                epoch_iter += opt.batch_size * opt.n_slices

                data['A'] = torch.reshape(data['A'],(data['A'].shape[0]* data['A'].shape[1],data['A'].shape[2], data['A'].shape[3])).unsqueeze(1)
                data['B'] = torch.reshape(data['B'],(data['B'].shape[0]* data['B'].shape[1],data['B'].shape[2], data['B'].shape[3])).unsqueeze(1)
                data['A_mask'] = torch.reshape(data['A_mask'],(data['A_mask'].shape[0]* data['A_mask'].shape[1],data['A_mask'].shape[2], data['A_mask'].shape[3])).unsqueeze(1)
                data['B_mask'] = torch.reshape(data['B_mask'],(data['B_mask'].shape[0]* data['B_mask'].shape[1],data['B_mask'].shape[2], data['B_mask'].shape[3])).unsqueeze(1)

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                # Create a dictionary for wandb  
                if opt.report_wandb:        
                    metrics_train = {f'train/{key}': value for key, value in losses.items()} 
                    metrics_train['train/lr'] = model.get_learning_rate()
                    wandb.log(metrics_train) 

                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs

            perf = validation(val_dataset, model, val_opt)
            print('saving the model at the end of epoch %d, iters %d, MAE %d' % (epoch, total_iters, perf[0]))
            if opt.report_wandb:
                metrics_val = {"val/MAE_real": perf[0], "val/MSE_real": perf[1], "val/SSIM_real" : perf[2], "val/PSNR_real": perf[3], 
                               "val/MAE_fake": perf[4], "val/MSE_fake": perf[5], "val/SSIM_fake" : perf[6], "val/PSNR_fake": perf[7]} 
                wandb.log(metrics_val)         
            model.save_networks('latest')
            model.save_networks(epoch)

            if perf[6] > best:
                print(f"Best Model with SSIM={best} in epoch {epoch}")
                model.save_networks('best')
                best = perf[6]

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        # Update learning rates at the end of every epoch.
        model.update_learning_rate()                    
