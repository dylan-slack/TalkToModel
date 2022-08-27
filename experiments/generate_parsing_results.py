"""Evaluates the models in batch. These results define the parsing table provided in the paper."""
import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description="generate parsing results table")
parser.add_argument("--id", type=str, required=True, help="a unique id to associate with the run")
parser.add_argument("--slurm", action="store_true", help="whether to run on slurm cluster")
parser.add_argument("--wandb", action="store_true", help="whether to use weights and biases")
args = parser.parse_args()

# The set of models for the sweep
models = [
    "EleutherAI/gpt-j-6B",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neo-1.3B",
    "nearest-neighbor",
    "t5-small",
    "t5-base",
    "t5-large",
][::-1]

# The datasets for the sweep
datasets = [
    "compas",
    "diabetes",
    "german"
]

# This setting is for whether to use guided decoding (True) or not (False)
gd_choices = [True, False]

for model in models:
    for ds in datasets:
        for gd in gd_choices:

            if gd and model == "nearest-neighbor":
                continue

            if gd:
                gd_text = "--gd"
            else:
                gd_text = ""

            model_name = model.replace("/", "_")

            if args.slurm:
                """
                This section defines the slurm job description. If you're not running these
                experiments on a slurm cluster, you will need to make sure slurm is not set.
                """
                job_template = f"""#!/bin/bash
                #SBATCH --job-name={model_name}-{ds}-{gd}-parsing-accuracy
                #SBATCH --output=parsing-accuracy-{model_name}-{ds}-{gd}.txt
                #SBATCH --time=21-00:00
                #SBATCH --partition=ava_s.p
                #SBATCH --nodelist=ava-s5
                #SBATCH --cpus-per-task=8
                #SBATCH --gpus=1
                #SBATCH --mem=35000MB
                
                srun python ./experiments/compute_parsing_accuracy.py --wandb --model '{model}' {gd_text} --dataset {ds} --id {args.id}
                """
                job_template = job_template.replace("    ", "")
            else:
                wf = ""
                if args.wandb:
                    wf = "--wandb"

                job_template = f"python experiments/compute_parsing_accuracy.py {wf} --model '{model}' {gd_text} --dataset {ds} --id {args.id}"\

            if args.slurm:
                job_file = os.path.join("slurm", f"{model.replace('/', '-')}-{ds}-{gd}-parsing-accuracy.sh")
                with open(job_file, "w") as file:
                    file.write(job_template)
                subprocess.run(f"chmod a+x {job_file}", shell=True, check=True)
                subprocess.run(f"sbatch {job_file}", shell=True, check=True)
            else:
                subprocess.run(job_template, shell=True, check=True)
