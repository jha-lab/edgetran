#!/bin/bash

#SBATCH --job-name=auto_gp		    		# create a short name for your job
#SBATCH --nodes=1                  	    	# node count
#SBATCH --ntasks=1                  	    # total number of tasks across all nodes
#SBATCH --cpus-per-task=16                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G                          # memory per cpu-core (4G is default)
#SBATCH --time=144:00:00                    # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=stuli@princeton.edu

module purge
module load anaconda3/2020.7
conda activate cnnbench

cd ..

python run_grow_and_prune.py