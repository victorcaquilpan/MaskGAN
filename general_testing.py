import subprocess

name = 'cyclegan'

for epoch in range(1,19):

    # # Define the script and its arguments
    scripts_with_args = [("test.py", [
            "--dataroot",
                    "../../data/two_stage_approach/slicing_unsupervised_coronal/",
                    "--gpu_ids",
                    "1",
                    "--model",
                    "cycle_gan",
                    "--name",
                    "cyclegan",
                    "--netG",
                    "unet-256",
                    "--dataset_mode",
                    "unaligned",
                    "--preprocess",
                    "none",
                    "--no_flip",
                    "--input_nc", "1",
                    "--output_nc", "1",
                    "--batch_size", "16",
                    "--n_slices", "1",
                    "--stage", "first",
                    "--epoch", "best"
        ]),
    ("voxel_creation.py", [
            "--results_folder", f"{name}/",
            "--final_voxels_folder", f"{name}/",
            "--size_input", "224"
        ]),
    
        ("evaluation.py", [
            "--results_folder", f"{name}/",
            "--final_voxels_folder", f"{name}/"

        ])]

    # Execute each script with arguments
    print(f'Results with epoch {epoch}')

    for script_and_argument in scripts_with_args:
        if len(script_and_argument) == 2:
            script, args = script_and_argument
            subprocess.run(["python", script] + args)
        else:
            subprocess.run(["python", script_and_argument])

