# Bayesian Analysis of Bellabeat Hourly Activity Patterns

## Overview

This project builds a hierarchical Bayesian model to understand how Bellabeat user personas differ in their hourly activity patterns. The model shows that personas vary not only in how much they move, but when they move — revealing clear circadian rhythms that support personalised engagement.

The analysis replaces ad‑hoc exploratory methods with a principled statistical model. The result is a clean, interpretable foundation for persona‑specific features such as timed notifications, habit‑building nudges, and personalised activity insights.

## Why Bayesian

Bayesian modelling is the right tool for this problem because:

- Hourly step counts are noisy and skewed. A Gamma likelihood handles this naturally.

- Users differ in baseline activity. A hierarchical structure captures this without overfitting.

- We care about uncertainty. Posterior distributions show how confident we are in each persona’s rhythm.

- Daily rhythms are smooth, not jagged. A Fourier basis gives interpretable 24‑hour curves without 24 separate parameters.

- We want probability‑based comparisons. Probability‑of‑superiority answers questions like “How likely is Persona A to be more active than Persona B at 3pm?”

In short: Bayesian methods give honest uncertainty, interpretable structure, and comparisons that make sense for real product decisions.

## Key Findings

1. Personas follow distinct 24‑hour activity rhythms
The model identifies four clear activity personas. These groups differ in both intensity and timing, not just total steps.

- Cardio Movers peak around 13:00–14:00, reaching ~1200 steps/hr.

- Mid‑activity users peak slightly later, with moderate mid‑day advantages over lower‑activity groups.

- Baseline users show moderate, smoother rhythms.

- LowAct users remain around 300–400 steps/hr with flatter curves and limited peak structure.

2. Hourly differences reveal patterns hidden by daily totals
Daily aggregates hide meaningful variation. Hour‑by‑hour comparisons show:

- Cardio Movers exceed LowAct users for most of the day (P > 0.95).

- Mid‑activity users exceed LowAct users during mid‑day hours (P ≈ 0.70–0.80).

- Personas differ more in when they move than in how much they move overall.

3. Most individual variation comes from baseline intensity
The hierarchical model shows that users differ more in overall activity level than in the shape of their daily cycle.
The random intercept captures most between‑user variability.

## Methodology

1. Data Preparation

- Merged daily and hourly step‑count datasets

- Attached persona labels

- Removed extreme outliers

- Built modelling arrays for personas, users, and hours

2. Bayesian Model

- Likelihood: Gamma with log‑link

- Circadian structure: First‑order Fourier series

- Hierarchy: User‑level random intercept

- Inference: NUTS sampling via PyMC

3. Convergence Diagnostics

The model converged cleanly:

- r̂ values at or near 1.00

- High effective sample sizes (ESS)

- Well‑mixed trace plots

- No divergences after tuning

- Energy plots showed no pathologies

These checks confirm that the posterior estimates are stable and trustworthy.

4. Behavioural Feature Extraction

- Amplitude (strength of daily rhythm)

- Peak hour (time of maximum activity)

- Posterior activity curves with HDIs

- Probability‑of‑superiority (hour‑by‑hour and daily‑mean)

## Results

### Reconstructed Activity Curves

Smooth 24‑hour curves show clear differences in amplitude and timing across personas.
HDI bands widen for personas with fewer users, reflecting honest uncertainty.

### Probability‑of‑Superiority Heatmaps

Hour‑by‑hour comparisons reveal when one persona is more active than another and how strongly those differences persist across the day.

### Daily‑Mean Summary

Averaged across all hours, personas form a stable hierarchy:

- Cardio — highest activity

- Mixed — strong intermediate

- Baseline — moderate

- LowAct — lowest

## Business Implications

Distinct hourly patterns create clear opportunities for personalised engagement:

- Morning‑active personas → early‑day nudges, hydration reminders

- Afternoon‑peak personas → midday motivation, step‑goal pacing

- Evening‑active personas → late‑day encouragement, recovery insights

- Low‑activity personas → simplified goals, gentle onboarding

- Aligning notifications with each persona’s natural rhythm can increase relevance and improve daily engagement.

## Lessons Learned

- Model structure matters more than complexity

- Likelihood choice must match the data

- Hierarchical models prevent false confidence

- Posterior predictive checks are essential

Interpretability drives product value

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
