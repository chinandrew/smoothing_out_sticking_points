#!/bin/bash
#SBATCH --mail-user=achin23@jhu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --time=144:00:00

module load conda
source activate env
source activate env

cd /users/achin/smoothing_out_sticking_points/logreg/

python run_logreg_simulation.py $method $iters $thin $SLURM_ARRAY_TASK_ID $stepsize  $steps $true_init $pslab $alpha  # num steps is array id


