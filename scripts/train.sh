#! /bin/bash
#SBATCH -p h200
#SBATCH --time 1:00:00
#SBATCH --gres gpu:h200:1
#SBATCH --signal=USR1@600

#SBATCH --output=logs/%x-%j.out

source /export/scratch/ra63ral/miniconda3/etc/profile.d/conda.sh
conda activate pytorch2.8_cu128

. ./scripts/setup.sh

launch --compile
# launch run_name=test-fsdp experiment=fm_img gc_freq=1000 val_freq=1000 distributed.dp_shard=$SLURM_GPUS_ON_NODE