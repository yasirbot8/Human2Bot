# coding=utf-8

import subprocess
from absl import app

def main(_):

    tasks = [5, 41]
    envs = ["env1", "env2"]
    seeds = [1, 2, 3]
    

    for task in tasks:
        for env in envs:
            for seed in seeds:
                if task == 5:
                    tl = 45
                    ph = 45
                    beta = 0.4
                if task == 41:
                    tl = 20
                    ph = 20
                    beta = 0.3
                    
                subprocess.run(
                    [
                        "python3",
                        "plan_h2b.py",
                        "--num_epochs", "100",
                        "--num_tasks", "3",
                        "--task_num", str(task),
                        "--seed", str(seed),
                        "--similarity", "1",
                        "--num_demos", "3",
                        "--traj_length", str(tl),
                        "--num_traj_per_epoch", "1",
                        "--phorizon", str(ph),
                        "--noise_beta", str(beta),
                        "--sample_sz", "70",
                        "--xml", env,
                        "--cem_iters", '2'
                ],
                check=True,
            )

if __name__ == "__main__":
    app.run(main)
