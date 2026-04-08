### LIBERO Setup

```bash
conda create -n libero_env python=3.10 -y
conda activate libero_env

# From project root:
# /export/home/ra48gaq/code/track-ae-release
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
cd LIBERO
pip install -r requirements.txt
pip install -e .
python benchmark_scripts/download_libero_datasets.py --download-dir .
cd ..
python scripts/preproc_text_emb.py --dataset-root LIBERO/datasets
```
