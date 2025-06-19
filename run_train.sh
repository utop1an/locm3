#!/bin/bash

# Define a single extraction type to run
# extractions=("p2" "p" "p2b" "pb")
extraction="p2"
scripts=("train1-5.slurm" "train6-8.slurm" "train9-10.slurm")

# Submit each script
for script in "${scripts[@]}"; do
    echo "Submitting job with --e=$extraction $script"
    sbatch --export=ALL,EVAL_TYPE=$extraction "$script"
done
