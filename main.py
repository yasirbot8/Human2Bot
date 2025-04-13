
import subprocess
from absl import app
import os

def main(_):

    # Define the command and its arguments
    command = [
        "python3",
        train.py,  # Use the correct path to the script
        "--num_tasks", "3",
        "--traj_length", "32",
        "--num_epochs", "150",
        "--log_dir", "my_final_trained/fast/task_gen/",
        "--out_size", "256",
        "--batch_size", "24",
        "--domain_aug",
        "--im_size", "120",
        "--seed", "1",
        "--lr", "5e-3",
        "--human_data_dir", "/data/0.5aug/",
        "--sim_dir", "data/human_demos/",
        "--human_tasks", "43", "44", "45",
        "--add_demos", "0",
    ]

    # Set the CUDA_VISIBLE_DEVICES environment variable
    env = {"CUDA_VISIBLE_DEVICES": "2"}

    # Run the command
    subprocess.run(command, check=True, env=env)

if __name__ == "__main__":
    app.run(main)
