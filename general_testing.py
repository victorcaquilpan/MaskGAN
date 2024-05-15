import subprocess

for epoch in range(7,10):

    # # Define the script and its arguments
    scripts_with_args = [("test.py", [
            "--dataroot", "../../data/two_stage_approach/slicing_unsupervised_coronal/",
            "--gpu_ids", "0",
            "--model", "mask_gan",
            "--name", "secondstage_solved",
            "--netG", "att",
            "--dataset_mode", "unaligned_chunks",
            "--preprocess", "none",
            "--input_nc", "1",
            "--output_nc", "1",
            "--no_flip",
            "--batch_size", "16",
            "--n_slices", "1",
            "--epoch", str(epoch),
        ]),
    ("voxel_creation.py", [
            "--results_folder", "secondstage_solved/",
            "--final_voxels_folder", "secondstage_solved/",
            "--size_input", "224"
        ]),
    
        ("final_evaluation.py", [
            "--results_folder", "secondstage_solved/",
            "--final_voxels_folder", "secondstage_solved/"

        ])]

    # Execute each script with arguments
    print(f'Results with epoch {epoch}')

    for script_and_argument in scripts_with_args:
        if len(script_and_argument) == 2:
            script, args = script_and_argument
            subprocess.run(["python", script] + args)
        else:
            subprocess.run(["python", script_and_argument])

