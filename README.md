# Meta-Learned NextConfigTransformer for Constrained BBOB

This repository implements the full pipeline from the Bayesian Optimization practical assignment:

1. Build offline **teacher sequences** with a strong baseline (e.g., constrained EI with GP).
2. Train a **NextConfigTransformer** to imitate those sequences.
3. Benchmark the learned policy against **Random Search** and **qLogEI** on the COCO bbob-constrained suite.
4. Plot **training curves** and **benchmark performance**.

The main scripts are:

- `build_sequence.py` – builds and saves offline teacher sequences.
- `train_model.py` – trains the Transformer from the cached teacher sequences.
- `new_benchmark.py` – benchmarks Random, qLogEI and Transformer on COCO.
- `plot_graphs.py` – plots training loss and benchmark curves.

---

## Run Sequence

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Build teacher sequences (slow, CPU-heavy)
python build_sequence.py

# 2. Train Transformer models (dim=2,10)
python train_model.py --dims 2 10 --seq_dir teacher_seqs --log_dir training_logs

# 3. Benchmark Random vs qLogEI vs Transformer on COCO
python new_benchmark.py

# 4. Generate plots for the report/poster
python plot_graphs.py


