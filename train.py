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
#from torchmetrics import StructuralSimilarityIndexMeasure
import torch
import wandb
import copy
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import PIL
import numpy as np

def validate(val_set, model):
    #ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    # Getting MAE
    metric_mae = torch.nn.L1Loss()
    # Getting MSE
    metric_mse = torch.nn.MSELoss()
    # Getting SSIM
    metric_ssim = StructuralSimilarityIndexMeasure()
    # Getting PSNR
    metric_psnr = PeakSignalNoiseRatio()

    model.eval()
    # Set zero
    mae = 0.0
    mse = 0.0
    ssim = 0.0
    psnr = 0.0
    for i, data in enumerate(val_set):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()   # calculate loss functions, get gradients, update network weights
        visuals = model.get_current_visuals()
        mri = visuals['real_A']
        real = visuals['real_B']
        pred = visuals['fake_B']

        # Converting values from floating (-1, 1) to 8-bytes integer (0,255)
        # Assuming normalization was done using mean = 0.5, std = 0.5
#        mri = mri * 0.5 + 0.5
#        mri = (mri * 255)
#        real = real * 0.5 + 0.5
#        real = (real * 255)
#        pred = pred * 0.5 + 0.5
#        pred = (pred * 255)

        mae += metric_mae(pred.cpu(), real.cpu())
        mse += metric_mse(pred.cpu(), real.cpu())
        ssim += metric_ssim(pred.cpu(), real.cpu())
        psnr += metric_psnr(pred.cpu(), real.cpu())

        # if i == 4:
        #     # Save real
        #     image_wandb_real = real.cpu().numpy()
        #     image_wandb_real = np.concatenate(image_wandb_real, axis=1)
        #     image_wandb_real = ((image_wandb_real + 1) * 127.5).astype(np.uint8)22
        #     image_wandb_real = PIL.Image.fromarray(np.squeeze(image_wandb_real))
        #     image_wandb_real = image_wandb_real.convert("L")

        #     # Save prediction
        #     image_wandb_pred = pred.cpu().numpy()
        #     image_wandb_pred = np.concatenate(image_wandb_pred, axis=1)
        #     image_wandb_pred = ((image_wandb_pred + 1) * 127.5).astype(np.uint8)
        #     image_wandb_pred = PIL.Image.fromarray(np.squeeze(image_wandb_pred))
        #     image_wandb_pred = image_wandb_pred.convert("L")
        #     if wandb: 
        #         wandb.log({"val/examples": [wandb.Image(image_wandb_real, caption="real"),wandb.Image(image_wandb_pred, caption="prediction")]})

        if i == 4: 
            image_wandb_mri = mri.cpu().numpy()
            image_wandb_mri = np.concatenate(image_wandb_mri, axis=1)
            # Save real
            image_wandb_real = real.cpu().numpy()
            image_wandb_real = np.concatenate(image_wandb_real, axis=1)
            # Save prediction
            image_wandb_pred = pred.cpu().numpy()
            image_wandb_pred = np.concatenate(image_wandb_pred, axis=1)
        elif i == 8 or i == 20:
            # Save real
            image_wandb_mri_2 = mri.cpu().numpy()
            image_wandb_mri_2 = np.concatenate(image_wandb_mri_2, axis=1)
            image_wandb_mri = np.concatenate((image_wandb_mri, image_wandb_mri_2), axis=2)
            image_wandb_real_2 = real.cpu().numpy()
            image_wandb_real_2 = np.concatenate(image_wandb_real_2, axis=1)
            image_wandb_real = np.concatenate((image_wandb_real, image_wandb_real_2), axis=2)
            # Save prediction
            image_wandb_pred_2 = pred.cpu().numpy()
            image_wandb_pred_2 = np.concatenate(image_wandb_pred_2, axis=1)
            image_wandb_pred = np.concatenate((image_wandb_pred, image_wandb_pred_2), axis=2)
        elif i == 24:
            image_wandb_mri_2 = mri.cpu().numpy()
            image_wandb_mri_2 = np.concatenate(image_wandb_mri_2, axis=1)
            image_wandb_mri = np.concatenate((image_wandb_mri, image_wandb_mri_2), axis=2)
            image_wandb_mri = ((image_wandb_mri + 1) * 127.5).astype(np.uint8)
            image_wandb_mri = PIL.Image.fromarray(np.squeeze(image_wandb_mri))
            image_wandb_mri = image_wandb_mri.convert("L")
            image_wandb_real_2 = real.cpu().numpy()
            image_wandb_real_2 = np.concatenate(image_wandb_real_2, axis=1)
            image_wandb_real = np.concatenate((image_wandb_real, image_wandb_real_2), axis=2)
            image_wandb_real = ((image_wandb_real + 1) * 127.5).astype(np.uint8)
            image_wandb_real = PIL.Image.fromarray(np.squeeze(image_wandb_real))
            image_wandb_real = image_wandb_real.convert("L")
            # Save prediction
            image_wandb_pred_2 = pred.cpu().numpy()
            image_wandb_pred_2 = np.concatenate(image_wandb_pred_2, axis=1)
            image_wandb_pred = np.concatenate((image_wandb_pred, image_wandb_pred_2), axis=2)
            image_wandb_pred = ((image_wandb_pred + 1) * 127.5).astype(np.uint8)
            image_wandb_pred = PIL.Image.fromarray(np.squeeze(image_wandb_pred))
            image_wandb_pred = image_wandb_pred.convert("L")
            if wandb_run: 
                wandb.log({"val/examples": [wandb.Image(image_wandb_mri, caption="MRI"),wandb.Image(image_wandb_real, caption="CT"),wandb.Image(image_wandb_pred, caption="sCT")]})

    return (mae/len(val_set)).item(), (mse/len(val_set)).item(), (ssim/len(val_set)).item(), (psnr/len(val_set)).item()

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    wandb_run = True
    if wandb_run:
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
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                # Create a dictionary for wandb  
                if wandb_run:        
                    metrics_train = {f'train/{key}': value for key, value in losses.items()} 
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
            perf = validate(val_dataset, model)
            print('saving the model at the end of epoch %d, iters %d, MAE %d' % (epoch, total_iters, perf[0]))
            if wandb_run: 
                metrics_val = {"val/MAE": perf[0], "val/MSE": perf[1], "val/SSIM" : perf[2], "val/PSNR": perf[3]} 
                wandb.log(metrics_val)         
            model.save_networks('latest')
            model.save_networks(epoch)

            if perf[2] > best:
                print(f"Best Model with SSIM={best}")
                model.save_networks('best')
                best = perf[2]

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
