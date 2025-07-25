#! /bin/bash
#SBATCH --job-name=polocm_1
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=u6797221@anu.edu.au     # Where to send mail
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=120:00:00
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --output=1_%j_polocm.log
# Standard output and error log
pwd; hostname; date
echo "--- running ---"

# disable cplex log
export CPLEX_LOGFILE=/dev/null

for dod in 0.1 0.2 0.3 0.4 0.5; do
    echo "Running with --d=$dod"
    singularity exec -B ~/locm3/src/:/src \
        -B ~/locm3/data/:/data \
        -B ~/locm3/logs:/logs \
        -B ~/locm3/output/:/output \
        ~/locm3/polocm.sif python3 /src/experiment.py \
        --ct 2 --et 8 \
        --d $dod \
        --cplex /opt/cplex/cplex/bin/x86-64_linux/cplex \
        --e p2 --debug

    echo "Completed task with --d=$dod, --e=p2"

    sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,State,Elapsed >> 1_$SLURM_JOB_ID_polocm.log
done

echo "--- done ---"
date
