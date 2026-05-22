# 🥊 UFC Fight Predictor Dashboard

**Built by [Saransh Raina](https://github.com/saranshraina)**

An interactive machine learning dashboard that predicts UFC fight outcomes with **72.8% accuracy** — beating professional Vegas oddsmakers by ~4.2 percentage points.

Live at → https://ufcfightpredictor.streamlit.app/

---

## Features

### 📅 Upcoming Events
- Scrapes live fight cards from [ufcstats.com](http://ufcstats.com) automatically
- Runs AI predictions on every fight on the card
- Hot Picks section for high-confidence matchups (≥70%)
- Probability bars per matchup + recent results with model accuracy tracking
- Auto-refreshes every 30 minutes, manual refresh button for fight night

### 🥊 Fight Predictor
- Select any two fighters from 2,400+ fighters in the database
- Head-to-head stat comparison bars (striking, grappling, physical)
- Win probability, confidence level, and dual-fighter radar chart overlay
- Interactive probability gauge

### 📊 Model Dashboard
- Model accuracy metrics (72.8% accuracy, ROC-AUC 0.805)
- Permutation-based feature importance
- Confusion matrix and win rate by weight class
- Finish method breakdown

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
| **HistGradientBoosting (tuned)** | **72.8%** | **0.805** |

**Top predictive features:**
1. Striking accuracy differential
2. Striking defense differential
3. Strikes landed per minute differential
4. Takedown accuracy differential
5. Reach advantage

The model uses 14 differential features (Red minus Blue) across striking, grappling, and physical attributes. Trained with an 80/20 **chronological split** and tuned via **Optuna** (80 trials, TimeSeriesSplit CV) to eliminate temporal leakage.

---

## Tech Stack

- **ML:** scikit-learn (HistGradientBoostingClassifier) · Optuna hyperparameter tuning
- **Dashboard:** Streamlit + streamlit-autorefresh
- **Charts:** Plotly
- **Scraping:** BeautifulSoup4 + urllib (ufcstats.com)
- **Data:** 6,393 UFC fights · 2,479 fighter profiles

---

## Setup

```bash
git clone https://github.com/saranshraina/-ufc-fight-predictor-.git
cd -ufc-fight-predictor-
pip install -r requirements.txt
streamlit run app.py
```

## Data

```
data/
  large_dataset.csv      # 6,393 UFC fights with engineered features
  fighter_stats.csv      # Per-fighter career statistics
```

---

## Project Structure

```
ufc-fight-predictor/
├── app.py          # Main Streamlit dashboard (4 pages)
├── scraper.py      # Live ufcstats.com scraper
├── requirements.txt
└── README.md
```

---

*Data sourced from ufcstats.com. For educational and analytical purposes only.*

*Made by Saransh Raina*
