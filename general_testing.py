import subprocess

name = 'maskgan_secondtstage_adult'
dataset = "../../data/intermediate_2d_images/maskgan_firststage_adult_coronal/"
stage = 'second'
model = 'mask_gan'
netG = 'att'

for epoch in range(35,42):

    # # Define the script and its arguments
    scripts_with_args = [
        
        ("test.py", [
            "--dataroot",
                    dataset,
                    "--gpu_ids",
                    "0",
                    "--model",
                    model,
                    "--name",
                    name,
                    "--netG",
                    netG,
                    "--dataset_mode",
                    "unaligned",
                    "--preprocess",
                    "none",
                    "--no_flip",
                    "--batch_size", "96",
                    "--n_slices", "1",
                    "--stage", stage,
                    "--epoch", str(epoch)
        ]),
    ("voxel_creation.py", [
            "--results_folder", f"{name}/",
            "--final_voxels_folder", f"{name}/",
            "--size_input", "224"
        ]),
    
        ("evaluation.py", [
            "--results_folder", f"{name}/",
            "--final_voxels_folder", f"{name}/",
            "--stage", stage

        ])]

    # Execute each script with arguments
    print(f'Results with epoch {epoch}')

    for script_and_argument in scripts_with_args:
        if len(script_and_argument) == 2:
            script, args = script_and_argument
            subprocess.run(["python", script] + args)
        else:
            subprocess.run(["python", script_and_argument])

