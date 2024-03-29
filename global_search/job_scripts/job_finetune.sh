#!/bin/bash

# Script to fine-tune a given model

# Author : Shikhar Tuli

cluster="della"
id="stuli"
model_hash=""
partition="gpu"

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
ENDC='\033[0m'

Help()
{
   # Display Help
   echo -e "${CYAN}Script to fine-tune a given model${ENDC}"
   echo
   echo -e "Syntax: source ${CYAN}./job_scripts/job_finetune.sh${ENDC} [${YELLOW}flags${ENDC}]"
   echo "Flags:"
   echo -e "${YELLOW}-c${ENDC} | ${YELLOW}--cluster${ENDC} [default = ${GREEN}\"della\"${ENDC}] \t\t Selected cluster - adroit, tiger or della"
   echo -e "${YELLOW}-p${ENDC} | ${YELLOW}--partition${ENDC} [default = ${GREEN}\"gpu\"${ENDC}] \t\t Selected partition if cluster is della"
   echo -e "${YELLOW}-i${ENDC} | ${YELLOW}--id${ENDC} [default = ${GREEN}\"stuli\"${ENDC}] \t\t\t Selected PU-NetID to email slurm updates"
   echo -e "${YELLOW}-m${ENDC} | ${YELLOW}--model_hash${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Hash of the pretrained model"
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
    -m | --model_hash)
        shift
        model_hash=$1
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
  cluster_gpu="gpu:tesla_v100:2"
elif [[ $cluster == "tiger" ]]
then
  cluster_gpu="gpu:2"
elif [[ $cluster == "della" ]]
then
  cluster_gpu="gpu:2"
else
    echo "Unrecognized cluster"
    return 1
fi

job_file="./job_${model_hash}.slurm"
mkdir -p "./job_scripts/glue/"

cd "./job_scripts/glue/"

# Create SLURM job script to train CNN-Accelerator pair
echo "#!/bin/bash" >> $job_file
echo "#SBATCH --job-name=glue_${model_hash}              # create a short name for your job" >> $job_file
if [[ $partition == "gpu-ee" ]]
then
    echo "#SBATCH --partition gpu-ee                         # partition" >> $job_file
fi
echo "#SBATCH --nodes=1                                      # node count" >> $job_file
echo "#SBATCH --ntasks-per-node=1                            # total number of tasks across all nodes" >> $job_file
echo "#SBATCH --cpus-per-task=16                             # cpu-cores per task (>1 if multi-threaded tasks)" >> $job_file
echo "#SBATCH --mem=32G                                      # memory per cpu-core (4G is default)" >> $job_file
echo "#SBATCH --gres=${cluster_gpu}                          # number of gpus per node" >> $job_file
echo "#SBATCH --time=144:00:00                               # total run time limit (HH:MM:SS)" >> $job_file
echo "#SBATCH --mail-type=all" >> $job_file
echo "" >> $job_file

echo "module purge" >> $job_file
echo "module load anaconda3/2020.7" >> $job_file
echo "conda activate txf_design-space" >> $job_file
echo "" >> $job_file
echo "cd ../.." >> $job_file
echo "" >> $job_file

echo "python finetune_model.py --pretrained_dir ../models/global_search/${model_hash} --autotune" >> $job_file

sbatch $job_file

cd ../..
