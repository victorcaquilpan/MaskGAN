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

def validate(val_set, model, opt):
    #ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    # Getting MAE
    metric_mae = torch.nn.L1Loss()
    # Getting MSE
    metric_mse = torch.nn.MSELoss()
    # Getting SSIM
    metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    # Getting PSNR
    metric_psnr = PeakSignalNoiseRatio()

    model.eval()
    # Set zero
    mae_sct1 = 0
    mse_sct1 = 0
    ssim_sct1 = 0
    psnr_sct1 = 0

    mae_sct2 = 0
    mse_sct2 = 0
    ssim_sct2 = 0
    psnr_sct2 = 0

    # Check number of total batches for validation set
    val_slices = len(val_set.dataloader)
    range_between_batches = val_slices // 3

    for i, data in enumerate(val_set):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()   # calculate loss functions, get gradients, update network weights
        visuals = model.get_current_visuals()
        sct = visuals['real_A']
        real = visuals['real_B']
        pred = visuals['fake_B']

        mae_sct1 += metric_mae(sct.cpu(), real.cpu())
        mse_sct1 += metric_mse(sct.cpu(), real.cpu())
        ssim_sct1 += metric_ssim(sct.cpu(), real.cpu())
        psnr_sct1 += metric_psnr(sct.cpu(), real.cpu())

        mae_sct2 += metric_mae(pred.cpu(), real.cpu())
        mse_sct2 += metric_mse(pred.cpu(), real.cpu())
        ssim_sct2 += metric_ssim(pred.cpu(), real.cpu())
        psnr_sct2 += metric_psnr(pred.cpu(), real.cpu())

        if i == range_between_batches*0: 
            image_wandb_mri = sct.cpu().numpy()
            if image_wandb_mri.shape[1] > 4:
                image_wandb_mri = image_wandb_mri[0,10:14,:,:]
            image_wandb_mri = np.concatenate(image_wandb_mri, axis=0)
            # Save real
            image_wandb_real = real.cpu().numpy()
            if image_wandb_real.shape[1] > 4:
                image_wandb_real = image_wandb_real[0,10:14,:,:]
            image_wandb_real = np.concatenate(image_wandb_real, axis=0)
            # Save prediction
            image_wandb_pred = pred.cpu().numpy()
            if image_wandb_pred.shape[1] > 4:
                image_wandb_pred = image_wandb_pred[0,10:14,:,:]
            image_wandb_pred = np.concatenate(image_wandb_pred, axis=0)
        elif i == range_between_batches*1 or i == range_between_batches*2:
            # Save real
            image_wandb_mri_2 = sct.cpu().numpy()
            if image_wandb_mri_2.shape[1] > 4:
                image_wandb_mri_2 = image_wandb_mri_2[0,0:4,:,:]
            image_wandb_mri_2 = np.concatenate(image_wandb_mri_2, axis=0)
            image_wandb_mri = np.concatenate((image_wandb_mri, image_wandb_mri_2), axis=1)
            image_wandb_real_2 = real.cpu().numpy()
            if image_wandb_real_2.shape[1] > 4:
                image_wandb_real_2 = image_wandb_real_2[0,0:4,:,:]
            image_wandb_real_2 = np.concatenate(image_wandb_real_2, axis=0)
            image_wandb_real = np.concatenate((image_wandb_real, image_wandb_real_2), axis=1)
            # Save prediction
            image_wandb_pred_2 = pred.cpu().numpy()
            if image_wandb_pred_2.shape[1] > 4:
                image_wandb_pred_2 = image_wandb_pred_2[0,0:4,:,:]
            image_wandb_pred_2 = np.concatenate(image_wandb_pred_2, axis=0)
            image_wandb_pred = np.concatenate((image_wandb_pred, image_wandb_pred_2), axis=1)
        if i == range_between_batches*2:
            image_wandb_mri_2 = sct.cpu().numpy()
            if image_wandb_mri_2.shape[1] > 4:
                image_wandb_mri_2 = image_wandb_mri_2[0,10:14,:,:]
            image_wandb_mri_2 = np.concatenate(image_wandb_mri_2, axis=0)
            image_wandb_mri = np.concatenate((image_wandb_mri, image_wandb_mri_2), axis=1)
            image_wandb_mri = ((image_wandb_mri + 1) * 127.5).astype(np.uint8)
            image_wandb_mri = PIL.Image.fromarray(np.squeeze(image_wandb_mri))
            image_wandb_mri = image_wandb_mri.convert("L")
            image_wandb_real_2 = real.cpu().numpy()
            if image_wandb_real_2.shape[1] > 4:
                image_wandb_real_2 = image_wandb_real_2[0,10:14,:,:]
            image_wandb_real_2 = np.concatenate(image_wandb_real_2, axis=0)
            image_wandb_real = np.concatenate((image_wandb_real, image_wandb_real_2), axis=1)
            image_wandb_real = ((image_wandb_real + 1) * 127.5).astype(np.uint8)
            image_wandb_real = PIL.Image.fromarray(np.squeeze(image_wandb_real))
            image_wandb_real = image_wandb_real.convert("L")
            # Save prediction
            image_wandb_pred_2 = pred.cpu().numpy()
            if image_wandb_pred_2.shape[1] > 4:
                image_wandb_pred_2 = image_wandb_pred_2[0,10:14,:,:]
            image_wandb_pred_2 = np.concatenate(image_wandb_pred_2, axis=0)
            image_wandb_pred = np.concatenate((image_wandb_pred, image_wandb_pred_2), axis=1)
            image_wandb_pred = ((image_wandb_pred + 1) * 127.5).astype(np.uint8)
            image_wandb_pred = PIL.Image.fromarray(np.squeeze(image_wandb_pred))
            image_wandb_pred = image_wandb_pred.convert("L")
            
            if opt.report_wandb: 
                wandb.log({"val/examples": [wandb.Image(image_wandb_mri, caption="sCT1"),wandb.Image(image_wandb_real, caption="CT"),wandb.Image(image_wandb_pred, caption="sCT2")]})

    return (mae_sct1/len(val_set)).item(), (mse_sct1/len(val_set)).item(), (ssim_sct1/len(val_set)).item(), (psnr_sct1/len(val_set)).item(), (mae_sct2/len(val_set)).item(), (mse_sct2/len(val_set)).item(), (ssim_sct2/len(val_set)).item(), (psnr_sct2/len(val_set)).item()

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    if opt.report_wandb:
        wandb.init(project="maskgan2D", name=opt.name) 

    # Include paramter to avoid force testing
    opt.force_testing = False
    train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    val_opt = copy.deepcopy(opt)
    val_opt.phase = 'val'
    val_opt.serial_batches = True
    # Include paramter to avoid force testing
    val_opt.force_testing = False
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

            perf = validate(val_dataset, model,val_opt)
            print('saving the model at the end of epoch %d, iters %d, MAE %d' % (epoch, total_iters, perf[0]))
            if opt.report_wandb:
                metrics_val = {"val/MAE_sCT1": perf[0], "val/MSE_sCT1": perf[1], "val/SSIM_sCT1" : perf[2], "val/PSNR_sCT1": perf[3],
                               "val/MAE_sCT2": perf[4], "val/MSE_sCT2": perf[5], "val/SSIM_sCT2" : perf[6], "val/PSNR_sCT2": perf[7]} 
                wandb.log(metrics_val)         
            model.save_networks('latest')
            model.save_networks(epoch)

            if perf[2] > best:
                print(f"Best Model with SSIM={best}")
                model.save_networks('best')
                best = perf[2]

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.