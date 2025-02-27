#!/bin/bash
#SBATCH --time=00:10:00      
#SBATCH --mem=5G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='pascal|volta|ampere' 
#SBATCH --output=Test_mcxlab_rfmusreplay_27022025_PH.out

## Load needed modules.
module load matlab/r2023b ## only available
module load gcc/11.4.0 ## versions 8-11 for matlab/r2023b; Simo told to us 12.3.0
module load cuda/12.2.1 ## only available

# Report modules.
nvcc --version
module list

## Run GPU accelerated executable(, note the --gres).
srun matlab -nodisplay -batch "demo_replay_FD_mus_PH();"
