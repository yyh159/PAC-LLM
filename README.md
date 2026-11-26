# PAC-LLM

Predicting the Unpredictable: LLM-powered Long-term Chaotic Time Series Forecasting under Short-term Observations

## Structure

```text
data_provider/              data generation and dataloaders
model/                      PAC-LLM
prompt/                     dataset description prompts
scripts/                    run_all.sh
util/                       metrics
run.py                      shared train/evaluation logic
train.py                    train entry
evaluate.py                 evaluation entry
```

## Installation

```bash
pip install -r requirements.txt
```

## Experimental Environment

Experiments were conducted on an NVIDIA RTX 4080 SUPER-based virtual GPU with a 32 GB memory profile (Driver: 580.76.05, CUDA: 13.0).

## Data

The benchmark includes `Lorenz63`, `Rossler`, `Chua`, and `Lorenz96`.

```bash
python data_provider/generate.py --save_dir data_provider/data_dysts_benchmark
```

## Run

Train and evaluate one dataset:

```bash
bash scripts/lorenz63.sh
```

Run all datasets:

```bash
bash scripts/run_all.sh
```

You can also call the Python entry points directly:

```bash
python train.py --system Lorenz63 --feat_dim 3 --model_variant pac_llm
python evaluate.py --system Lorenz63 --feat_dim 3 --model_variant pac_llm
```

## Notes

Dataset descriptions are loaded from `prompt/{system}.txt`. Command-line arguments override the default values in `run.py`.
