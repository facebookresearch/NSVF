#! /bin/bash
#SBATCH --output=/checkpoint/jgu/slurm_logs/slurm-%A.out
#SBATCH --error=/checkpoint/jgu/slurm_logs/slurm-%A.err
#SBATCH --job-name=neural_rendering_ms
#SBATCH --partition=priority
#SBATCH --comment="opensource code neurips submission"
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=480g
#SBATCH -C volta32gb
#SBATCH --cpus-per-task=8
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=4320

trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
     # SLURM_JOB_ID is a unique representation of the job, equivalent
     # to above
     scontrol requeue $SLURM_JOB_ID
   fi
}


# Install signal handler
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

echo $@

srun --label bash $@
