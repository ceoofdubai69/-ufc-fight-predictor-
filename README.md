# 🥊 UFC Fight Predictor Dashboard

An interactive machine learning dashboard that predicts UFC fight outcomes with **70.1% accuracy** — beating professional Vegas oddsmakers by ~1.6 percentage points.

Built on research from a comprehensive analysis of 4,500+ UFC fights (1996–2024) using XGBoost and ensemble methods. The dashboard scrapes live fight cards directly from ufcstats.com and runs AI predictions on every matchup in real time.


 also available for use at - https://ufcfightpredictor.streamlit.app/
 
---

## Features

### 📅 Upcoming Events
- Scrapes live fight cards from [ufcstats.com](http://ufcstats.com) automatically
- Runs AI predictions on every fight on the card
- Visual win probability breakdown per matchup
- Recent results tab showing model accuracy vs actual outcomes

### 🥊 Fight Predictor
- Select any two fighters from 2,400+ fighters in the database
- Get win probability, confidence level, and a side-by-side stat comparison
- Interactive probability gauge

### 📊 Model Dashboard
- Model accuracy metrics (70.1% accuracy, ROC-AUC 0.732)
- Feature importance chart — striking accuracy difference is the #1 predictor
- Confusion matrix and calibration analysis
- Win rate breakdown by weight class and finish method

### 🏟️ Fighter Database
- Search and filter all 2,400+ fighters
- Radar chart skill profile for any fighter
- Full fight history per fighter

---

## Model

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 63.2% | 0.680 |
| Random Forest | 67.8% | 0.730 |
| **HistGradientBoosting** | **70.1%** | **0.732** |
| Neural Network | 66.1% | 0.710 |

**Top predictive features:**
1. Striking accuracy differential (14.2%)
2. Win streak / recent momentum (11.8%)
3. Striking defense differential (9.5%)
4. Reach advantage (8.7%)
5. Takedown accuracy differential (7.9%)

The model is trained on 28 engineered differential features across five categories: striking, grappling, physical attributes, momentum, and career stage.

---

## Tech Stack

- **ML:** scikit-learn (HistGradientBoostingClassifier)
- **Dashboard:** Streamlit
- **Charts:** Plotly
- **Scraping:** BeautifulSoup4 + urllib (ufcstats.com)
- **Data:** [UFC Kaggle Dataset](https://www.kaggle.com) — 4,500 fights, 1996–2024

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/ceoofdubai69/ufc-fight-predictor.git
cd ufc-fight-predictor

# 2. Install dependencies
pip install streamlit pandas scikit-learn plotly beautifulsoup4 lxml

# 3. Add your data files (see Data section below)

# 4. Run the app
streamlit run app.py
```

## Data

The app expects two CSV files. Download the dataset from Kaggle and place them at these paths (or update `DATA_PATH` and `FIGHTER_PATH` in `app.py`):

```
data/
  large_dataset.csv      # 4,500 UFC fights with engineered features
  fighter_stats.csv      # Per-fighter career statistics
```

Update the paths in `app.py`:
```python
DATA_PATH    = "data/large_dataset.csv"
FIGHTER_PATH = "data/fighter_stats.csv"
```

---

## Project Structure

```
ufc-fight-predictor/
├── app.py          # Main Streamlit dashboard (4 pages)
├── scraper.py      # Live ufcstats.com scraper
├── run.sh          # Launch script
└── README.md
```

---

## How It Works

Fighter stats are converted into **differential features** — for each fight, every stat is expressed as the difference between the Red and Blue corner fighter. This lets the model learn matchup-relative patterns rather than absolute stat thresholds.

```
sig_str_acc_diff = red_striking_accuracy - blue_striking_accuracy
reach_diff       = red_reach_cm - blue_reach_cm
wins_total_diff  = red_total_wins - blue_total_wins
# ... 28 features total
```

Live predictions for upcoming fights pull fresh fighter stats from ufcstats.com and compute the same differentials on the fly.

---

## Results

- **70.1% test accuracy** on 619 held-out fights
- **ROC-AUC: 0.732** — strong discrimination between winners and losers
- **Overfitting gap: 1.6%** (train 71.7% → test 70.1%) — model generalises well
- **Statistical significance:** z-score 11.64, p < 0.0001

---

*Data sourced from ufcstats.com and Kaggle. For educational and analytical purposes only.*
