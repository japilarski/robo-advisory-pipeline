# [ENGLISH TITLE PLACEHOLDER]
# [POLISH TITLE PLACEHOLDER]

Implementation accompanying my master thesis.

## Project overview
This repository contains the end-to-end pipeline for data preparation, portfolio construction, backtesting, and visualization used in the thesis.

## Structure
- `block_0.py` — data loading and preprocessing
- `block_1.py` — expected returns models
- `block_2.py` — covariance estimation
- `block_3.py` — portfolio optimization
- `block_4.py` — backtest without costs/taxes
- `block_5.py` — backtest with costs/taxes
- `block_6.py` — metrics and charts pipeline
- `thesis_data.py` — export thesis tables to `data/thesis/csv`
- `thesis_plots.py` — export thesis charts to `data/thesis/pdf`

## Usage
Run the scripts in order from `block_0.py` to `block_6.py`. Use `thesis_data.py` and `thesis_plots.py` to export thesis-ready tables and figures.
