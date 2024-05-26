
import PIL
import torch
import numpy as np
import wandb
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

def val_visualizations_over_batches(real_a, real_b, fake_b):
    """We define a function to visualizate val images of sets
        real_A, real_B and fake_B
    """
    # Save real A
    real_a = real_a.cpu().numpy()
    real_a = real_a[4:8,:,:,:]
    real_a = np.concatenate(real_a, axis=1)
    # Save real B
    real_b = real_b.cpu().numpy()
    real_b = real_b[4:8,:,:,:]
    real_b = np.concatenate(real_b, axis=1)
    # Save prediction
    fake_b = fake_b.cpu().numpy()
    fake_b = fake_b[4:8,:,:,:]
    fake_b = np.concatenate(fake_b, axis=1)

    return real_a, real_b, fake_b

def val_visualizations_over_chunks(real_a, real_b, fake_b):
    """Another auxiliar function to show visualizations over chunks
    """
    # Save real A
    real_a = real_a.cpu().numpy()
    real_a = real_a[0,4:8,:,:]
    real_a = np.concatenate(real_a, axis=0)
    # Save real B
    real_b = real_b.cpu().numpy()
    real_b = real_b[0,4:8,:,:]
    real_b = np.concatenate(real_b, axis=0)
    # Save fake B
    fake_b = fake_b.cpu().numpy()
    fake_b = fake_b[0,4:8,:,:]
    fake_b = np.concatenate(fake_b, axis=0)

    return real_a, real_b, fake_b

def validation(val_set, model, opt):
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
    mae_real = 0.0
    mse_real = 0.0
    ssim_real = 0.0
    psnr_real = 0.0
    mae_fake = 0.0
    mse_fake = 0.0
    ssim_fake = 0.0
    psnr_fake = 0.0

    # Check number of total batches for validation set
    batches = len(val_set.dataloader)
    # Select values which represent 20th, 40th, 60th, and 80th percentiles
    percentile_values = np.percentile(np.arange(0,batches + 1), [20,40,60,80]).astype(int)

    for i, data in enumerate(val_set):  # inner loop within one epoch

        # Transform the input to treat each slice separately
        if opt.stage == 'second':
            data['A'] = torch.reshape(data['A'],(data['A'].shape[0]* data['A'].shape[1],data['A'].shape[2], data['A'].shape[3])).unsqueeze(1)
            data['B'] = torch.reshape(data['B'],(data['B'].shape[0]* data['B'].shape[1],data['B'].shape[2], data['B'].shape[3])).unsqueeze(1)
            data['A_mask'] = torch.reshape(data['A_mask'],(data['A_mask'].shape[0]* data['A_mask'].shape[1],data['A_mask'].shape[2], data['A_mask'].shape[3])).unsqueeze(1)
            data['B_mask'] = torch.reshape(data['B_mask'],(data['B_mask'].shape[0]* data['B_mask'].shape[1],data['B_mask'].shape[2], data['B_mask'].shape[3])).unsqueeze(1)

        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()   # calculate loss functions, get gradients, update network weights
        visuals = model.get_current_visuals()
        real_A = visuals['real_A']
        real_B = visuals['real_B']
        fake_B = visuals['fake_B']

        # Get metrics comparing real A with real B
        mae_real += metric_mae(real_A.cpu(), real_B.cpu())
        mse_real += metric_mse(real_A.cpu(), real_B.cpu())
        ssim_real += metric_ssim(real_A.cpu(), real_B.cpu())
        psnr_real += metric_psnr(real_A.cpu(), real_B.cpu())
        # Get metrics comparing fake B with real B
        mae_fake += metric_mae(fake_B.cpu(), real_B.cpu())
        mse_fake += metric_mse(fake_B.cpu(), real_B.cpu())
        ssim_fake += metric_ssim(fake_B.cpu(), real_B.cpu())
        psnr_fake += metric_psnr(fake_B.cpu(), real_B.cpu())

        # Create visualizations
        if opt.report_wandb: 
            if i == percentile_values[0]: 
                imgA_wb, imgB_wb, fakeB_wb = val_visualizations_over_batches(real_A,real_B,fake_B)
            elif i == percentile_values[1] or i == percentile_values[2]:
                imgA2_wb, imgB2_wb, fakeB2_wb = val_visualizations_over_batches(real_A,real_B,fake_B)
                imgA_wb = np.concatenate((imgA_wb, imgA2_wb), axis=2)
                imgB_wb = np.concatenate((imgB_wb, imgB2_wb), axis=2)
                fakeB_wb = np.concatenate((fakeB_wb, fakeB2_wb), axis=2)
            elif i == percentile_values[3]:
                imgA2_wb, imgB2_wb, fakeB2_wb = val_visualizations_over_batches(real_A,real_B,fake_B)
                imgA_wb = np.concatenate((imgA_wb, imgA2_wb), axis=2)
                imgA_wb = ((imgA_wb + 1) * 127.5).astype(np.uint8)
                imgA_wb = PIL.Image.fromarray(np.squeeze(imgA_wb))
                imgA_wb = imgA_wb.convert("L")
                imgB_wb = np.concatenate((imgB_wb, imgB2_wb), axis=2)
                imgB_wb = ((imgB_wb + 1) * 127.5).astype(np.uint8)
                imgB_wb = PIL.Image.fromarray(np.squeeze(imgB_wb))
                imgB_wb = imgB_wb.convert("L")
                fakeB_wb = np.concatenate((fakeB_wb, fakeB2_wb), axis=2)
                fakeB_wb = ((fakeB_wb + 1) * 127.5).astype(np.uint8)
                fakeB_wb = PIL.Image.fromarray(np.squeeze(fakeB_wb))
                fakeB_wb = fakeB_wb.convert("L")
          
    # Send data to Wandb
    if opt.report_wandb: 
        wandb.log({"val/examples": [wandb.Image(imgA_wb, caption="realA"),wandb.Image(imgB_wb, caption="realB"),wandb.Image(fakeB_wb, caption="fakeB")]})                                     
                                         
    return (mae_real/batches).cpu().numpy(), (mse_real/batches).cpu().numpy(), (ssim_real/batches).cpu().numpy(), (psnr_real/batches).cpu().numpy(), \
           (mae_fake/batches).cpu().numpy(), (mse_fake/batches).cpu().numpy(), (ssim_fake/batches).cpu().numpy(), (psnr_fake/batches).cpu().numpy()
