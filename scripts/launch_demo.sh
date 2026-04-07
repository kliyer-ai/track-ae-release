#! /bin/bash
#SBATCH -p h200
#SBATCH --gres gpu:h200:1
#SBATCH --time 6:00:00



source /export/home/ra49veb/miniconda3/etc/profile.d/conda.sh
conda activate pytorch2.6

export GRADIO_TEMP_DIR="/export/home/ra49veb/dev/diffusion-trajectory-ae/outputs/gradio"

srun python -m scripts.demo --server_port 55551 