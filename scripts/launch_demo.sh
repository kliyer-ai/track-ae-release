#! /bin/bash
#SBATCH -p a100
#SBATCH --gres gpu:a100:1
#SBATCH --time 6:00:00



source /export/home/ra49veb/miniconda3/etc/profile.d/conda.sh
conda activate pytorch2.6

export GRADIO_TEMP_DIR="/export/home/ra49veb/dev/diffusion-trajectory-ae/outputs/gradio"

srun python -m scripts.demo --server_port 55551 