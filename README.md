# Bayesian Analysis of Bellabeat Hourly Activity Patterns

## Overview

This project builds a hierarchical Bayesian model to understand how Bellabeat user personas differ in their hourly activity patterns. The analysis confirms that personas vary not only in how much they move, but when they move — revealing distinct circadian rhythms that can support personalised engagement strategies.

The model resolves several methodological issues in the original exploratory analysis and provides a statistically principled foundation for persona‑specific product features such as timed notifications, habit‑building nudges, and personalised activity insights.

## Key Findings

Personas show distinct and credible daily activity rhythms.

- A Gamma likelihood ensures all predictions remain positive and matches the right‑skewed nature of step‑count data.

- A Fourier basis replaces the unstable 24‑parameter lookup table, producing smooth, interpretable circadian curves.

- A hierarchical structure corrects pseudoreplication by modelling user‑level variation.

- Posterior predictive checks confirm the model reproduces the observed distribution.

- Probability‑of‑superiority analysis quantifies how likely one persona is to exceed another at each hour.

- Personas differ in peak activity time, ranging from ~13:30 to ~16:00 depending on group.

These results validate the exploratory analysis and provide a robust statistical basis for personalised engagement.

## Methodology

1. Data Preparation
- Merged daily and hourly step‑count datasets

- Attached persona labels

- Removed extreme outliers

- Constructed modelling arrays for personas, users, and hours

2. Bayesian Model
- Likelihood: Gamma with log‑link

- Circadian structure: First‑order Fourier series

- Hierarchy: User‑level random intercept

- Inference: NUTS sampling via PyMC

- Diagnostics: r̂ ≈ 1.00, high ESS, clean PPC

3. Behavioural Feature Extraction
- Amplitude (strength of daily rhythm)

- Peak hour (time of maximum activity)

- Posterior activity curves with HDIs

- Probability‑of‑superiority (hour‑by‑hour and daily‑mean)

## Results

Reconstructed Activity Curves
Smooth 24‑hour curves show clear differences in both amplitude and timing across personas. HDI bands widen appropriately for personas with fewer users, reflecting honest uncertainty.

Probability‑of‑Superiority Heatmaps
Hour‑by‑hour comparisons reveal when one persona is more active than another and how strongly those differences persist across the day.

Daily‑Mean Summary Matrix
Averaged across all hours, personas form a stable hierarchy:

- Cardio — highest activity

- Mixed — strong intermediate

- Baseline — moderate

- LowAct — lowest

## Persona Comparison Table

Summarises amplitude, peak hour, mean predicted activity, and pairwise probabilities.

Business Implications

The distinct hourly patterns suggest clear opportunities for personalised engagement:

- Morning‑active personas → early‑day nudges, hydration reminders

- Afternoon‑peak personas → midday motivation, step‑goal pacing

- Evening‑active personas → late‑day encouragement, recovery insights

- Low‑activity personas → simplified goals, gentle onboarding

Aligning notifications with each persona’s natural rhythm can increase relevance and improve daily engagement.

## Lessons Learned

- Model structure matters more than complexity.

- Likelihood choice must match the data.

- Hierarchical models prevent false confidence.

- Posterior predictive checks are essential.

- Interpretability drives product value.

## Repository Structure

Code
├── bellabeat_bayesian_analysis.ipynb
├── bellabeat_bayesian_analysis.md
├── README.md                   # This file
└── images/                     # Plots and visualisations

## Technologies Used

- Python

- PyMC

- ArviZ

- NumPy / Pandas

- Matplotlib / Seaborn

## How to Run

- Clone the repository

- Install dependencies

- Open the notebook in JupyterLab

- Run top‑to‑bottom

The notebook is fully reproducible and includes all modelling steps, diagnostics, and visualisations.

## Author

Andrew Moran

I focus on statistical modelling, behavioural insights, and clear, accessible technical communication.
