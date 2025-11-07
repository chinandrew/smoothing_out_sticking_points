#!/bin/bash
#SBATCH --mail-user=achin23@jhu.edu
#SBATCH --mail-type=FAIL,END

module load conda
source activate env
source activate env

cd /users/achin/smoothing_out_sticking_points/simulation/

python run_simulation.py $method $n_iter $seed $n $p $nonzero $alpha $sigma2 $thin $num_blocks $p_slab $tau2 $t1 $t2
