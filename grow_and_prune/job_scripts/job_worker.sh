#!/bin/bash

# Script to train a given model

# Author : Shikhar Tuli

cluster="della"
id="stuli"
model_dir=""
model_hash=""
steps=""
partition="gpu"

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
ENDC='\033[0m'

Help()
{
   # Display Help
   echo -e "${CYAN}Script to train a given model${ENDC}"
   echo
   echo -e "Syntax: source ${CYAN}./job_scripts/job_train.sh${ENDC} [${YELLOW}flags${ENDC}]"
   echo "Flags:"
   echo -e "${YELLOW}-c${ENDC} | ${YELLOW}--cluster${ENDC} [default = ${GREEN}\"tiger\"${ENDC}] \t\t Selected cluster - adroit, tiger or della"
   echo -e "${YELLOW}-p${ENDC} | ${YELLOW}--partition${ENDC} [default = ${GREEN}\"gpu\"${ENDC}] \t\t Selected partition if cluster is della"
   echo -e "${YELLOW}-i${ENDC} | ${YELLOW}--id${ENDC} [default = ${GREEN}\"stuli\"${ENDC}] \t\t\t Selected PU-NetID to email slurm updates"
   echo -e "${YELLOW}-d${ENDC} | ${YELLOW}--model_dir${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Directory of the FlexiBERT model"
   echo -e "${YELLOW}-m${ENDC} | ${YELLOW}--model_hash${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Hash of the FlexiBERT model"
   echo -e "${YELLOW}-s${ENDC} | ${YELLOW}--steps${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Number of steps for pre-training"
   echo -e "${YELLOW}-h${ENDC} | ${YELLOW}--help${ENDC} \t\t\t\t\t Call this help message"
   echo
}

while [[ $# -gt 0 ]]
do
case "$1" in
    -c | --cluster)
        shift
        cluster=$1
        shift
        ;;
    -p | --partition)
        shift
        partition=$1
        shift
        ;;
    -i | --id)
        shift
        id=$1
        shift
        ;;
    -d | --model_dir)
        shift
        model_dir=$1
        shift
        ;;
    -m | --model_hash)
        shift
        model_hash=$1
        shift
        ;;
    -s | --steps)
        shift
        steps=$1
        shift
        ;;
    -h| --help)
       Help
       return 1;
       ;;
    *)
       echo "Unrecognized flag $1"
       return 1;
       ;;
esac
done  

if [[ $cluster == "adroit" ]]
then
  cluster_gpu="gpu:tesla_v100:4"
elif [[ $cluster == "tiger" ]]
then
  cluster_gpu="gpu:4"
elif [[ $cluster == "della" ]]
then
  cluster_gpu="gpu:2"
else
    echo "Unrecognized cluster"
    return 1
fi

job_file="./job_${model_hash}.slurm"
mkdir -p "./job_scripts/auto_gp/"

cd "./job_scripts/auto_gp/"

# Create SLURM job script to train CNN-Accelerator pair
echo "#!/bin/bash" >> $job_file
echo "#SBATCH --job-name=pretrain_${model_hash}               # create a short name for your job" >> $job_file
if [[ $partition == "gpu-ee" ]]
then
    echo "#SBATCH --partition gpu-ee                         # partition" >> $job_file
fi
echo "#SBATCH --nodes=2                                      # node count" >> $job_file
echo "#SBATCH --ntasks-per-node=1                            # total number of tasks across all nodes" >> $job_file
echo "#SBATCH --cpus-per-task=24                             # cpu-cores per task (>1 if multi-threaded tasks)" >> $job_file
# echo "#SBATCH --cpus-per-task=4                              # cpu-cores per task (>1 if multi-threaded tasks)" >> $job_file
echo "#SBATCH --mem=128G                                     # memory per cpu-core (4G is default)" >> $job_file
# echo "#SBATCH --mem=32G                                      # memory per cpu-core (4G is default)" >> $job_file
echo "#SBATCH --gres=${cluster_gpu}                      # number of gpus per node" >> $job_file
# echo "#SBATCH --gres=gpu:1" >> $job_file
echo "#SBATCH --time=36:00:00                                # total run time limit (HH:MM:SS)" >> $job_file
# echo "#SBATCH --time=6:00:00                                 # total run time limit (HH:MM:SS)" >> $job_file
# echo "#SBATCH --mail-type=all                                # send email" >> $job_file
# echo "#SBATCH --mail-user=stuli@princeton.edu" >> $job_file
echo "" >> $job_file

echo "# Get the node list and the master address" >> $job_file
echo "node_list_string=\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\")" >> $job_file
echo "node_list_arr=(\$node_list_string)" >> $job_file
echo "master_addr=\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\" | head -n 1)" >> $job_file
echo "echo \"NODELIST=\"\${SLURM_NODELIST}" >> $job_file
echo "echo \"MASTER_ADDR=\"\$master_addr" >> $job_file
echo "" >> $job_file

echo "module purge" >> $job_file
echo "module load anaconda3/2020.7" >> $job_file
# echo "conda init bash" >> $job_file
echo "conda activate txf_design-space" >> $job_file
echo "" >> $job_file
echo "cd ../.." >> $job_file
echo "" >> $job_file

echo "for i in {0..1}" >> $job_file
echo "do" >> $job_file
echo -e "\tssh \${node_list_arr[\${i}]} \"conda activate txf_design-space; \\" >> $job_file
echo -e "\t\tcd /scratch/gpfs/stuli/edge_txf/grow_and_prune/; \\" >> $job_file
echo -e "\t\tpython -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=\${i} \\" >> $job_file
echo -e "\t\t--master_addr=\${master_addr} --master_port=12345 pretrain_model.py --output_dir ${model_dir} --steps ${steps} --local_rank \${i}\" &" >> $job_file
echo "done" >> $job_file
echo "" >> $job_file

echo "wait" >> $job_file

sbatch $job_file

cd ../../
