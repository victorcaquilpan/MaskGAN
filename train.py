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


def validate(val_set, model, inferer, result_dir, epoch):
    # Getting MAE
    metric_mae = torch.nn.L1Loss()
    # Getting MSE
    metric_mse = torch.nn.MSELoss()
    #ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    # Getting SSIM
    metric_ssim = monai.metrics.SSIMMetric(spatial_dims=3)
    # Getting PSNR
    metric_psnr = PeakSignalNoiseRatio()

    model.eval()
    mae = 0
    mse = 0
    ssim = 0
    psnr = 0
    cnt = 0
    for i, data in enumerate(val_set):  # inner loop within one epoch
        with torch.no_grad():
            pred = inferer(data['A'].cuda(), network=model.netG_A)
        real = data['B']
        mae += metric_mae(pred.cpu(), real)
        mse += metric_mse(pred.cpu(), real)
        ssim += metric_ssim(pred.cpu(), real)
        psnr += metric_psnr(pred.cpu(), real)
        
        if i == 0:
            save_nifti(pred, result_dir, f'{epoch}_val_fake.nii')
        cnt += 1
    return (mae/cnt).item(), (mse/cnt).item(), (ssim/cnt).item(), (psnr/cnt).item()
  

if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = TrainOptions().parse()
    wandb_opt = vars(opt)
    
    # Initialize WANDB setting
    wandb.init(project="3d-cyclegan", name=wandb_opt['name'], config = wandb_opt)

    # -----  Transformation and Augmentation process for the data  -----
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
    # trainTransforms = [
    #             NiftiDataset.Resample(opt.new_resolution, opt.resample),
    #             NiftiDataset.Augmentation(),
    #             #NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
    #             NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
    #             ]
    
    # train_set = NifitDataSet(opt.data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=True, train=True)
    train_set = UnalignedDataset(opt)
    print('length train list:', len(train_set))
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)  # Here are then fed to the network with a defined batch size

    result_dir = os.path.join(opt.checkpoints_dir, opt.name, 'visuals')
    val_opt = copy.deepcopy(opt)
    val_opt.phase = 'val'
    val_opt.serial_batches = True
    val_dataset = UnalignedDataset(val_opt)  # create a dataset given opt.dataset_mode and other options
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=opt.workers, pin_memory=True)  # Here are then fed to the network with a defined batch size
    inferer = SlidingWindowInferer(roi_size=opt.patch_size, sw_batch_size=4, mode="gaussian")
    # -----------------------------------------------------
    model = create_model(opt)  # creation of the model
    model.setup(opt)
    if opt.continue_train:
        model.load_networks(opt.which_epoch)
    visualizer = Visualizer(opt)
    total_steps = 0
    print("Start training")
    best = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                # Create a dictionary for wandb                                          # ADDED
                metrics_train = {f'train/{key}': value for key, value in losses.items()} # ADDED
                wandb.log(metrics_train)                                                 # ADDED


                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                visuals = model.get_current_visuals()
                visualizer.save_results(visuals, epoch)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')
                # visuals = model.get_current_visuals()
                # visualizer.save_results(visuals, epoch)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            perf = validate(val_loader, model, inferer, result_dir, epoch)
            metrics_val = {"val/MAE": perf[0], "val/MSE": perf[1], "val/SSIM" : perf[2], "val/PSNR": perf[3]} # ADDED
            wandb.log(metrics_val)          # ADDED
            model.save_networks('latest')
            model.save_networks(epoch)
        
            #wandb.log({'val_ssim': perf})
            model.save_networks('latest')
            if perf[2] > best:
                print(f"Best Model with SSIM={perf[2]}")
                model.save_networks('best')
                best = perf[2]

            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
