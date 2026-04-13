## Action Prediction Evaluation on LIBERO

If you want to run the actual LIBERO simulation and reproduce success rates, follow the setup below.
Getting the LIBERO simulation to run can be tricky; these are the instructions used for this release and should work smoothly if followed as-is.
This code has been tested on NVIDIA A100 GPUs.

### LIBERO Setup

```bash
conda create -n libero_env python=3.10 -y
conda activate libero_env

# From project root:
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
cd LIBERO
pip install -r requirements.txt
pip install -e .
python benchmark_scripts/download_libero_datasets.py --download-dir .
cd ..

pip install -r requirements_libero.txt
python scripts/preproc_text_emb.py --dataset-root LIBERO/datasets
```

### Launch Evaluation

Assumes you already activated the environment and completed setup above.

```bash
cd <PROJECT_ROOT>
accelerate launch ./scripts/eval_libero_policy.py \
  --save_path <OUTPUT_DIR> \
  --ckpt_path <POLICY_CKPT_PATH> \
  --suite <libero_goal|libero_object|libero_spatial|libero_10|libero_90>
```

Notes:
- Required args are `--save_path`, `--ckpt_path`, and `--suite`.
- Defaults assume LIBERO is at `<PROJECT_ROOT>/LIBERO` with datasets at `LIBERO/datasets` and embeddings at `LIBERO/task_embedding_caches/task_emb_bert.npy`.
- If your layout differs, also set `--dataset_path`, `--task_emb_cache_path`, and `--libero_path`.
