# Early academic risk prediction with course-module context on OULAD

Reproducibility repository for the manuscript **“Early prediction of academic risk considering the course module using OULAD data”** by A. M. Abuov and O. S. Salykova.

The repository contains the data-preparation code, clean year-based split, baseline models, five-seed neural experiments, component ablations, forecast-horizon analysis, calibration metrics, the explicit module-logit-bias control, and student-cluster bootstrap inference.

## Main experimental design

- Target: binary academic risk, `Fail` or `Withdrawn`.
- Observation window: exact days `0..41` for the six-week experiment.
- Train: `2013B` and `2013J`.
- Validation: `2014B`.
- Test: `2014J`.
- Student identifiers are removed from later partitions if they occur in an earlier partition.
- The `CCC` module is absent from the 2013 training presentations and is represented by a dedicated zero embedding.
- Neural results are reported across five random seeds.
- Uncertainty of PR-AUC differences is evaluated by 2,000 student-cluster bootstrap replicates with additional resampling of neural seeds.

## Reported results

| Model | PR-AUC on 2014J |
|---|---:|
| Logistic Regression | 0.6521 |
| HistGradientBoosting | 0.6441 |
| Full course-conditioned model | 0.6606 ± 0.0035 |
| Without course-module context | 0.6278 ± 0.0015 |
| Module code only in the hidden output classifier | 0.6592 ± 0.0033 |
| Scalar module-logit bias only | 0.6446 ± 0.0054 |

Student-cluster bootstrap 95% confidence intervals for the PR-AUC difference:

- full model minus Logistic Regression: `[0.0027, 0.0146]`;
- full model minus HistGradientBoosting: `[0.0092, 0.0234]`.

The neural model does not outperform all baselines on F1 or probability calibration. The results support a contribution of `code_module`, but they do not establish an independent benefit of temporal attention or the learned fusion gate.

## Repository structure

```text
.
├── data/oulad/raw/                 # place the original OULAD CSV files here
├── results/                        # aggregate results reported in the manuscript
├── oulad_data.py                   # exact weekly-window construction and shared utilities
├── run_oulad_multiseed.py          # five-seed model and neural ablations
├── run_clean_baseline_comparison.py
├── run_hgb_module_ablation.py
├── run_clean_horizons.py
├── run_paired_bootstrap.py         # produces predictions used by bootstrap
├── run_student_cluster_bootstrap.py
├── run_supervisor_highlight_experiments.py
├── requirements.txt
└── CITATION.cff
```

## Data

Download the anonymised **Open University Learning Analytics Dataset (OULAD)** from the [official Open University page](https://research.stem.open.ac.uk/ouanalyse/dataset/).

Extract the CSV files into:

```text
data/oulad/raw/
```

The experiments require at least:

```text
studentInfo.csv
studentRegistration.csv
studentVle.csv
```

The raw dataset is not redistributed in this repository.

Dataset citation:

> Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data, 4*, 170171. https://doi.org/10.1038/sdata.2017.171

## Environment

The reported runs used:

- Python 3.12.14
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
- PyTorch 2.5.1 CPU

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduction

Run commands from the repository root.

### 1. Five-seed main experiment and neural ablations

```bash
python run_oulad_multiseed.py
```

This command also constructs the exact six-week VLE windows if the cache is absent.

### 2. Baseline comparison

```bash
python run_clean_baseline_comparison.py
```

### 3. HistGradientBoosting ablation of `code_module`

```bash
python run_hgb_module_ablation.py
```

`HistGradientBoostingClassifier` is evaluated with `early_stopping=False`.

### 4. Exact 2-, 4-, 6-, and 8-week horizons

```bash
python run_clean_horizons.py 2
python run_clean_horizons.py 4
python run_clean_horizons.py 6
python run_clean_horizons.py 8
```

### 5. Scalar module-logit-bias control and results without CCC

```bash
python run_supervisor_highlight_experiments.py
```

### 6. Student-cluster bootstrap

First generate the aligned test predictions:

```bash
python run_paired_bootstrap.py
```

Then run the student-cluster bootstrap:

```bash
python run_student_cluster_bootstrap.py
```

## Precomputed outputs

The `results/` directory contains compact CSV and JSON summaries used in the manuscript. Raw OULAD data, trained model checkpoints, and large intermediate weekly-feature tables are intentionally excluded.

## Русская инструкция

1. Скачайте OULAD с официальной страницы.
2. Поместите исходные CSV в `data/oulad/raw/`.
3. Создайте виртуальное окружение и установите `requirements.txt`.
4. Запускайте команды из корня репозитория.
5. Для статьи используйте ссылку на корень опубликованного GitHub-репозитория, например:

```text
https://github.com/USERNAME/oulad-course-conditioned-risk-prediction
```

Рекомендуемая формулировка в статье:

> Скрипты подготовки данных, обучения, абляции и бутстреп-анализа доступны в открытом репозитории: https://github.com/USERNAME/oulad-course-conditioned-risk-prediction

## License and data terms

The source code is provided for research reproducibility. Before public release, the repository owner should select a code license in GitHub. OULAD remains subject to its original terms and is not included here.

