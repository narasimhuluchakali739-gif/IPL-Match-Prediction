# IPL Match Outcome Prediction 🏏

An end-to-end Machine Learning pipeline designed to predict the outcomes of Indian Premier League (IPL) matches based on historical data, venue bias, and head-to-head statistics.

## Project Overview
This project processes ball-by-ball historical data to extract match-level summaries and engineers predictive features to forecast future fixtures. The model outputs calibrated probabilities rather than absolute predictions to account for the unpredictable nature of T20 cricket.

## Key Features Engineered
* **Head-to-Head Win Rate:** A rolling historical calculation of how often the team batting first successfully defends their total against their specific opponent.
* **Venue Bias:** A statistical representation of stadium pitch conditions, calculating the historical 'bat-first win percentage' for every specific ground (e.g., M. Chinnaswamy Stadium vs. MA Chidambaram Stadium).

## Machine Learning Pipeline
1. **Data Preprocessing:** Standardized legacy franchise names (e.g., *Delhi Daredevils -> Delhi Capitals*), handled missing values, and encoded categorical variables.
2. **Model Training:** Trained a **Random Forest Classifier** with an 80/20 train-validation split to prevent overfitting.
3. **Probability Calibration:** Applied **Isotonic Regression** to smooth the Random Forest's overconfident predictions, heavily reducing the Log Loss penalty.
4. **Test Set Alignment:** Dynamically mapped future, un-played fixtures to historical schema and safely imputed neutral 50/50 probabilities for unknown playoff matchups.

## Tech Stack
* **Python** * **Pandas** (Data manipulation and aggregations)
* **Scikit-Learn** (RandomForestClassifier, CalibratedClassifierCV, train_test_split)
* **Jupyter Notebook**
