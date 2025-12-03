# Project Overview

This repository contains a custom GPT-style Transformer model designed for stock return prediction, inspired by and architecturally based on Karpathy’s GPT.
It represents an end-to-end experiment in building, training, and evaluating a numerical time-series GPT for financial forecasting.

The repository includes:

- **A_LLM_Data.csv** – Example dataset showing the exact structure and format used to train the model.

- **Test.py** – Evaluation script for testing the trained model on **unseen** tickers and generating performance metrics.

- **stocks_gpt.py** – Full implementation of the custom GPT-based time-series model, including preprocessing, sequence generation, and training logic.

---

## stocks_gpt.py

**stocks_gpt.py** contains the full implementation of my first custom GPT-style Transformer designed specifically for numerical stock time-series forecasting.
It follows the architectural principles of Karpathy’s GPT, but adapts them for multi-feature sequences instead of raw text.

This implementation is part of my first attempt at building a complete end-to-end GPT model from scratch, so it may still contain imperfections or inefficiencies. Nevertheless, it successfully processes a large collection of stock datasets and produces meaningful predictions on unseen companies.

### Training Dataset and Scale
The model was trained on:
- 793 individual company datasets,
- each containing approximately 100 weeks of historical features,
- resulting in roughly two million data points.

For a handcrafted dataset of this size (and without access to high-quality institutional financial feeds), this is a substantial amount of information. However, the data quality is not ideal—values vary widely across sources, and several features are noisy or incomplete.
In the future, I plan to revisit this project and build a cleaner and more reliable dataset.

### Model Performance

Despite the dataset limitations, the model converges to stable:
- Training loss ≈ 0.0016–0.0018
- Validation loss ≈ 0.0015–0.0016

These values are strong for weekly stock-return forecasting.
Financial time-series are notoriously noisy and difficult to predict, so achieving sub-0.0020 loss on unseen data indicates that the Transformer is successfully learning meaningful patterns rather than memorizing noise.
With a larger and higher-quality dataset, these results could likely be improved further.

---

## Test.py

**Test.py** is a standalone evaluation script designed to measure the model’s true generalization ability.
Instead of reusing training tickers, it loads a separate directory of unseen companies and runs the full preprocessing -> sequence building -> prediction -> metric computation.

This ensures that the results reflect the model’s real-world predictive performance, not just memorization of training data.

### What the script does
- Loads all *_LLM_Data.csv files from data/test/
- Applies the same normalization, feature engineering, and sequence windowing as during training
- Runs the trained GPTTimeSeriesModel in inference mode
- Computes multiple financial performance metrics:
	- Directional Accuracy (up/down correctness)
	- Pearson Correlation (linearity between predictions and reality)
	- Mean Absolute Error (average prediction error)
	- Ranking-based performance (top-vs-bottom predicted portfolios)


### Performance on Unseen Tickers (example run)
Using a test set of completely unseen tickers, the model produced the following results:
- Directional Accuracy: 56.12%
- Prediction–Real Correlation: 0.1887
- Mean Absolute Error: 0.05545

### Ranking-Based Portfolio Performance
When ranking stocks by predicted return, the model delivered:
- Top-10 predicted average return: +5.95%
- Bottom-10 predicted average return: –5.94%
- Long–Short Spread: ≈ 11.9%


These numbers are unusually high for weekly stock-return prediction and should be interpreted with **caution**.  
Even though the test companies were chosen at random and were never seen during training, the specific test period may coincide with strong market swings or sector moves, which can amplify apparent signal strength. It is also possible that there is still a subtle bug or methodological issue in the evaluation pipeline that I have not yet identified.

In other words, the portfolio results are promising and suggest that the model is capturing meaningful structure in the data, but they should be viewed as **exploratory**, not as definitive proof of real-world trading performance. Future work will include more robust backtesting, additional sanity checks, and independent verification of all calculations.

---

## Directional Accuracy

At first glance, a **56.12%** directional accuracy might look unimpressive — after all, it’s only slightly above 50%.
But in financial prediction, especially weekly stock returns, even a single percentage point of directional edge is considered significant.

Here’s why:

### 1. Random chance is 50%

Predicting whether a stock will go UP or DOWN next week is fundamentally difficult due to noise, news, sentiment, macro events, and market randomness.
A naïve model or random guess gives you:
-~50% accuracy

Even many professional quant funds operated with signals around 51%–53%, yet manage to produce enormous profits through **scaling**.
So 56.12% is NOT small. It's meaningfully above chance.

