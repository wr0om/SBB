# Silenced Biases: The Dark Side LLMs Learned to Refuse

**Rom Himelstein, Amit LeVi, Brit Youngmann, Yaniv Nemcovsky, Avi Mendelson**

This repository contains the code and instructions needed to reproduce the experiments from the paper
***Silenced Biases: The Dark Side LLMs Learned to Refuse***.

---

## Repository Structure

* `pipeline/` – Core experiment code, notebooks, and scripts
* `main_data/` – Dataset construction utilities
* `requirements.txt` – Python dependencies

---

## Getting Started

### 1. Clone the repository

```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Create a conda environment

```bash
conda create -n silenced-biases python=3.11.11
conda activate silenced-biases
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Demo

**`pipeline/demo.ipynb`**

A short demo showing how to:

* Define custom demographic categories and groups
* Define negative, positive, and neutral subjects
* Run and obtain answer frequencies and fairness metrics

This notebook is the recommended entry point for understanding the pipeline.

---

## Experiments

All experiments are located inside the `pipeline/` directory. The order below reflects their importance and role in the paper.

### 1. Main Experiments

* **`bias_multi_direction.py`** – Runs the main experiments with multiple bias directions
* **`bias_multi_direction_results.ipynb`** – Evaluates refusal frequencies

Run:

```bash
python bias_multi_direction.py --model_path {model}
```

### 2. `results_figs.ipynb`

* Generates all main result figures
* Includes statistical significance testing

### 3. `bias_direction.ipynb`

* Empirical validation that the refusal direction is unbiased
* Dataset similarity analysis to the refusal direction
* Corresponds to **Figure 2**

### 4. Jailbreak Experiments

* **`bias_jailbreak.py`** – Runs the jailbreak experiment
* **`bias_jailbreak_results.ipynb`** – Evaluates and visualizes results

Run:

```bash
python bias_jailbreak.py --model_path {model}
```

### 5. `motivation_fig.ipynb`

* PCA visualization of biased vs. unbiased responses
* Demonstrates that no new biases are introduced
* Corresponds to **Figure 3**

### 6. `ablation.ipynb`

* Results for the refusal steering method
* Direction ablation experiments (Appendix)

### 7. `appendix_examples.ipynb`

* Generates all answer-frequency tables shown in the Appendix

### 8. Dataset Construction

**`main_data/get_started.ipynb`**

Helps you:

* Define custom demographic categories
* Choose subjects and query variations
* Generate and save datasets to `main_data/quiz_bias/`

---

## Supported Models

The following models are supported out of the box:

```python
[
  'meta-llama/Llama-2-7b-chat-hf',
  'meta-llama/Llama-2-13b-chat-hf',
  'meta-llama/Meta-Llama-3-8B-Instruct',
  'meta-llama/Llama-3.1-8B-Instruct',
  'google/gemma-2b-it',
  'google/gemma-7b-it',
  'Qwen/Qwen-7B-Chat',
  'Qwen/Qwen-14B-Chat',
  'Qwen/Qwen2.5-7B-Instruct',
  'Qwen/Qwen2.5-14B-Instruct'
]
```

### Adding New Models

To add support for additional models:

1. Implement a new model class in `pipeline/model_utils/`
2. Register it in `pipeline/model_utils/model_factory.py`
