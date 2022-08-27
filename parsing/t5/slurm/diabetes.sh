# Fine-tunes all models for the diabetes dataset
export DATASET="diabetes"
sbatch ./parsing/t5/slurm/finetune_t5_small.sh
sbatch ./parsing/t5/slurm/finetune_t5_base.sh
sbatch ./parsing/t5/slurm/finetune_t5_large.sh
