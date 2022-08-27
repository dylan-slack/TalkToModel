# Fine-tunes all models for the diabetes dataset
sbatch ./parsing/t5/slurm/sweeps/t5_base_sweep_0.2.sh
sbatch ./parsing/t5/slurm/sweeps/t5_base_sweep_0.4.sh
sbatch ./parsing/t5/slurm/sweeps/t5_base_sweep_0.6.sh
sbatch ./parsing/t5/slurm/sweeps/t5_base_sweep_0.8.sh
sbatch ./parsing/t5/slurm/sweeps/t5_base_sweep_1.0.sh
