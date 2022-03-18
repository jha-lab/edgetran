#!/bin/bash

#SBATCH --job-name=pretrain_multinode       # create a short name for your job
#SBATCH --nodes=2                           # node count
#SBATCH --ntasks-per-node=1                 # total number of tasks per node
#SBATCH --cpus-per-task=24                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G                          # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2                        # number of gpus per node
#SBATCH --time=144:00:00                    # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=stuli@princeton.edu

# Get the node list and the master address
node_list_string=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
node_list_arr=($node_list_string)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "NODELIST="${SLURM_NODELIST}
echo "MASTER_ADDR="$master_addr

module purge
module load anaconda3/2020.7
conda activate txf_design-space

cd ..

for i in {0..1}
do
    ssh ${node_list_arr[${i}]} "conda activate txf_design-space; \
        cd /scratch/gpfs/stuli/edge_txf/grow_and_prune/; \
        python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=${i} \
            --master_addr=${master_addr} --master_port=12345 pretrain_model.py --output_dir ../models/bert_mini/40f62e468f3458f8d4a5b49ba1413ce6/ \
            --steps 1000000 --learning_rate 1e-4 --local_rank ${i}" &
done

wait
