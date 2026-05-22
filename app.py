import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import scraper
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "data/large_dataset.csv"
FIGHTER_PATH = "data/fighter_stats.csv"

FEATURE_COLS = [
    "sig_str_acc_total_diff", "td_acc_total_diff", "str_def_total_diff",
    "td_def_total_diff", "sub_avg_diff", "td_avg_diff",
    "SLpM_total_diff", "SApM_total_diff",
    "reach_diff", "height_diff", "age_diff", "weight_diff",
    "wins_total_diff", "losses_total_diff",
]

FEATURE_LABELS = {
    "sig_str_acc_total_diff": "Striking Accuracy",
    "td_acc_total_diff": "Takedown Accuracy",
    "str_def_total_diff": "Striking Defense",
    "td_def_total_diff": "Takedown Defense",
    "sub_avg_diff": "Submission Avg",
    "td_avg_diff": "Takedown Avg",
    "SLpM_total_diff": "Strikes Landed/Min",
    "SApM_total_diff": "Strikes Absorbed/Min",
    "reach_diff": "Reach (cm)",
    "height_diff": "Height (cm)",
    "age_diff": "Age (yrs)",
    "weight_diff": "Weight (kg)",
    "wins_total_diff": "Total Wins",
    "losses_total_diff": "Total Losses",
}

st.markdown("""
<style>
    /* ── Base ── */
    .main { background-color: #0e1117; }
    section[data-testid="stSidebar"] { background-color: #12141c; }

    /* ── Cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #e63946;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        height: 100%;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #e63946; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }

    .red-card {
        background: linear-gradient(135deg, #2a0a0a 0%, #1a0505 100%);
        border: 2px solid #e63946;
        border-radius: 14px;
        padding: 22px 18px;
        text-align: center;
    }
    .blue-card {
        background: linear-gradient(135deg, #0a0a2a 0%, #05051a 100%);
        border: 2px solid #4361ee;
        border-radius: 14px;
        padding: 22px 18px;
        text-align: center;
    }
    .win-banner {
        font-size: 1.5rem;
        font-weight: 800;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        margin: 16px 0;
        letter-spacing: 0.5px;
    }

    /* ── Fight cards ── */
    .fight-card {
        background: linear-gradient(135deg, #14161f 0%, #1a1c2e 100%);
        border: 1px solid #2a2d3e;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        transition: border-color 0.2s;
    }
    .fight-card:hover { border-color: #e63946; }

    .hot-pick-badge {
        display: inline-block;
        background: linear-gradient(90deg, #e63946, #ff6b35);
        color: white;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        margin-left: 8px;
        vertical-align: middle;
    }

    /* ── Prob bar ── */
    .prob-bar-wrap {
        display: flex;
        align-items: center;
        gap: 0;
        border-radius: 8px;
        overflow: hidden;
        height: 44px;
        margin: 10px 0;
    }
    .prob-bar-red {
        background: linear-gradient(90deg, #c1121f, #e63946);
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding: 0 12px;
        font-weight: 700;
        font-size: 1rem;
        color: white;
        transition: width 0.5s ease;
    }
    .prob-bar-blue {
        background: linear-gradient(90deg, #4361ee, #3a0ca3);
        display: flex;
        align-items: center;
        justify-content: flex-start;
        padding: 0 12px;
        font-weight: 700;
        font-size: 1rem;
        color: white;
        transition: width 0.5s ease;
    }

    /* ── Stat comparison bar ── */
    .stat-row {
        display: flex;
        align-items: center;
        margin: 6px 0;
        gap: 8px;
    }
    .stat-label { color: #aaa; font-size: 0.8rem; min-width: 120px; text-align: center; }
    .stat-bar-red {
        height: 18px;
        border-radius: 4px 0 0 4px;
        background: #e63946;
        min-width: 4px;
    }
    .stat-bar-blue {
        height: 18px;
        border-radius: 0 4px 4px 0;
        background: #4361ee;
        min-width: 4px;
    }
    .stat-val { font-size: 0.8rem; color: #ddd; min-width: 38px; }
    .stat-val-r { text-align: right; min-width: 38px; }

    /* ── Confidence badge ── */
    .conf-high { color: #2dc653; font-weight: 700; }
    .conf-med  { color: #f4a261; font-weight: 700; }
    .conf-low  { color: #aaa;    font-weight: 700; }

    /* ── Headings ── */
    h1 { color: #e63946 !important; }
    h2, h3 { color: #fff !important; }
    .stSelectbox label { color: #ccc !important; }

    /* ── Mobile ── */
    @media (max-width: 768px) {
        .metric-value { font-size: 1.5rem; }
        .win-banner { font-size: 1.1rem; padding: 12px; }
        .prob-bar-red, .prob-bar-blue { font-size: 0.85rem; padding: 0 8px; }
        .stat-label { min-width: 80px; font-size: 0.72rem; }
    }
</style>
""", unsafe_allow_html=True)


# ── Helper ───────────────────────────────────────────────────────────────────

def prob_bar_html(prob_red, name_red, name_blue):
    pr = f"{prob_red:.0%}"
    pb = f"{1 - prob_red:.0%}"
    wr = int(prob_red * 100)
    wb = 100 - wr
    return f"""
    <div style="margin:4px 0 2px 0;">
      <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#aaa;margin-bottom:4px;">
        <span style="color:#e63946;font-weight:600">🔴 {name_red}</span>
        <span style="color:#4361ee;font-weight:600">{name_blue} 🔵</span>
      </div>
      <div class="prob-bar-wrap">
        <div class="prob-bar-red" style="width:{wr}%">{pr}</div>
        <div class="prob-bar-blue" style="width:{wb}%">{pb}</div>
      </div>
    </div>"""


def stat_comparison_html(stats_list):
    """stats_list: list of (label, r_val, b_val, fmt, higher_is_better)"""
    rows = []
    for label, rv, bv, fmt, hib in stats_list:
        if rv is None or bv is None:
            continue
        total = abs(rv) + abs(bv)
        if total == 0:
            rw = bw = 50
        else:
            rw = int(abs(rv) / total * 100)
            bw = 100 - rw
        rv_str = fmt.format(rv)
        bv_str = fmt.format(bv)
        r_winner = (rv > bv) == hib
        r_bold = "font-weight:700;color:#e63946;" if r_winner else "color:#ccc;"
        b_bold = "font-weight:700;color:#4361ee;" if not r_winner else "color:#ccc;"
        rows.append(f"""
        <div class="stat-row">
          <div class="stat-val stat-val-r" style="{r_bold}">{rv_str}</div>
          <div class="stat-bar-red" style="width:{rw}%;flex:{rw}"></div>
          <div class="stat-label">{label}</div>
          <div class="stat-bar-blue" style="width:{bw}%;flex:{bw}"></div>
          <div class="stat-val" style="{b_bold}">{bv_str}</div>
        </div>""")
    return "<div>" + "".join(rows) + "</div>"


def conf_badge(confidence):
    if confidence > 0.72:
        return f'<span class="conf-high">HIGH CONFIDENCE ({confidence:.0%})</span>'
    elif confidence > 0.62:
        return f'<span class="conf-med">MODERATE ({confidence:.0%})</span>'
    return f'<span class="conf-low">CLOSE FIGHT ({confidence:.0%})</span>'


# ── Data / Model ─────────────────────────────────────────────────────────────

@st.cache_data
def load_and_train():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLS + ["winner"])
    df["target"] = (df["winner"] == "Red").astype(int)

    # Data is reverse-chronological; reverse for proper time-based split
    df = df.iloc[::-1].reset_index(drop=True)

    X = df[FEATURE_COLS]
    y = df["target"]

    # Chronological 80/20 split — no future leakage into test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Optuna-tuned params (80-trial TimeSeriesSplit search)
    model = HistGradientBoostingClassifier(
        max_iter=556,
        max_depth=3,
        learning_rate=0.0161,
        l2_regularization=1.744,
        min_samples_leaf=20,
        max_leaf_nodes=45,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)

    # Permutation-based feature importance
    baseline = roc_auc_score(y_test, y_proba)
    perm_imps = []
    rng = np.random.default_rng(0)
    for feat in FEATURE_COLS:
        X_perm = X_test.copy()
        X_perm[feat] = rng.permutation(X_perm[feat].values)
        score_perm = roc_auc_score(y_test, model.predict_proba(X_perm)[:, 1])
        perm_imps.append(max(0, baseline - score_perm))
    total = sum(perm_imps) or 1
    importances = pd.DataFrame({
        "feature":    [FEATURE_LABELS[f] for f in FEATURE_COLS],
        "importance": [v / total for v in perm_imps],
    }).sort_values("importance", ascending=True)

    return model, df, acc, auc, cm, importances


@st.cache_data
def load_fighters():
    return pd.read_csv(FIGHTER_PATH)


@st.cache_data(ttl=3600)
def fetch_upcoming_events():
    return scraper.get_upcoming_events()


@st.cache_data(ttl=3600)
def fetch_recent_events():
    return scraper.get_recent_completed_events(n=3)


@st.cache_data(ttl=3600)
def fetch_event_fights(url):
    return scraper.get_event_fights(url)


@st.cache_data(ttl=3600)
def fetch_completed_results(url):
    return scraper.get_completed_event_results(url)


@st.cache_data(ttl=86400)
def fetch_live_fighter(name, url):
    local = fighters_df[fighters_df["name"].str.lower() == name.lower()]
    if not local.empty:
        return local.iloc[0].to_dict()
    if url:
        live = scraper.scrape_fighter_stats(url)
        if live:
            return live
    found_url = scraper.find_fighter_url(name)
    if found_url:
        return scraper.scrape_fighter_stats(found_url)
    return {}


def predict_matchup(red_stats, blue_stats):
    def g(d, key, default=0.0):
        v = d.get(key, default)
        try:
            return float(v) if v is not None else default
        except Exception:
            return default

    feat = {
        "sig_str_acc_total_diff": g(red_stats, "sig_str_acc") - g(blue_stats, "sig_str_acc"),
        "td_acc_total_diff":      g(red_stats, "td_acc")      - g(blue_stats, "td_acc"),
        "str_def_total_diff":     g(red_stats, "str_def")     - g(blue_stats, "str_def"),
        "td_def_total_diff":      g(red_stats, "td_def")      - g(blue_stats, "td_def"),
        "sub_avg_diff":           g(red_stats, "sub_avg")     - g(blue_stats, "sub_avg"),
        "td_avg_diff":            g(red_stats, "td_avg")      - g(blue_stats, "td_avg"),
        "SLpM_total_diff":        g(red_stats, "SLpM")        - g(blue_stats, "SLpM"),
        "SApM_total_diff":        g(red_stats, "SApM")        - g(blue_stats, "SApM"),
        "reach_diff":             g(red_stats, "reach")       - g(blue_stats, "reach"),
        "height_diff":            g(red_stats, "height")      - g(blue_stats, "height"),
        "age_diff":               g(red_stats, "age")         - g(blue_stats, "age"),
        "weight_diff":            g(red_stats, "weight")      - g(blue_stats, "weight"),
        "wins_total_diff":        g(red_stats, "wins")        - g(blue_stats, "wins"),
        "losses_total_diff":      g(red_stats, "losses")      - g(blue_stats, "losses"),
    }
    X = pd.DataFrame([feat])[FEATURE_COLS]
    prob_red = model.predict_proba(X)[0, 1]
    return prob_red, 1 - prob_red


model, df, accuracy, roc_auc, cm, importances = load_and_train()
fighters_df = load_fighters()
fighter_names = sorted(fighters_df["name"].dropna().unique().tolist())

vegas_baseline = 0.685
vs_vegas = accuracy - vegas_baseline

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥊 UFC Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Upcoming Events", "Fight Predictor", "Model Dashboard", "Fighter Database"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        f"**Accuracy:** {accuracy:.1%}  \n"
        f"**ROC-AUC:** {roc_auc:.3f}  \n"
        f"**vs Vegas:** +{vs_vegas:.1%}  \n"
        f"**Fights trained:** {len(df):,}"
    )
    st.markdown("---")
    st.caption("Model: HistGradientBoosting · Optuna-tuned · Time-series CV")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — UPCOMING EVENTS
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Upcoming Events":
    st.title("📅 Upcoming UFC Events")
    st.markdown("Live fight cards from **ufcstats.com** with AI win predictions.")

    tab_upcoming, tab_recent = st.tabs(["🔜 Upcoming Cards", "🏁 Recent Results"])

    with tab_upcoming:
        with st.spinner("Fetching from ufcstats.com..."):
            try:
                events = fetch_upcoming_events()
            except Exception as e:
                st.error(f"Could not reach ufcstats.com: {e}")
                events = []

        if not events:
            st.info("No upcoming events found.")
        else:
            event_names = [e["name"] for e in events]
            selected_event = st.selectbox("Select Event", event_names)
            ev = next(e for e in events if e["name"] == selected_event)

            c1, c2 = st.columns(2)
            c1.markdown(f"**Date:** {ev['date']}")
            c2.markdown(f"**Location:** {ev['location']}")

            st.markdown("")
            run_btn = st.button("⚡ Run AI Predictions", type="primary", use_container_width=True)

            if run_btn:
                with st.spinner("Fetching fighter stats and running predictions..."):
                    fights = fetch_event_fights(ev["url"])

                if not fights:
                    st.warning("Could not load fight card.")
                else:
                    results = []
                    prog = st.progress(0)
                    for i, fight in enumerate(fights):
                        r_stats = fetch_live_fighter(fight["r_fighter"], fight.get("r_url", ""))
                        b_stats = fetch_live_fighter(fight["b_fighter"], fight.get("b_url", ""))
                        prob_r, prob_b = predict_matchup(r_stats, b_stats)
                        confidence = max(prob_r, prob_b)
                        pick = fight["r_fighter"] if prob_r > 0.5 else fight["b_fighter"]
                        results.append({
                            "fight": fight,
                            "prob_r": prob_r,
                            "prob_b": prob_b,
                            "confidence": confidence,
                            "pick": pick,
                        })
                        prog.progress((i + 1) / len(fights))

                    prog.empty()

                    # Hot picks summary
                    hot = [r for r in results if r["confidence"] >= 0.70]
                    if hot:
                        st.markdown("### 🔥 Hot Picks (≥70% confidence)")
                        hp_cols = st.columns(min(len(hot), 3))
                        for col, r in zip(hp_cols, hot[:3]):
                            with col:
                                color = "#e63946" if r["prob_r"] > 0.5 else "#4361ee"
                                st.markdown(
                                    f'<div style="background:{color}22;border:2px solid {color};'
                                    f'border-radius:10px;padding:12px;text-align:center;">'
                                    f'<div style="font-size:0.75rem;color:#aaa;">'
                                    f'{r["fight"]["weight_class"]}</div>'
                                    f'<div style="font-weight:700;font-size:1rem;color:{color};">'
                                    f'{r["pick"]}</div>'
                                    f'<div style="font-size:1.3rem;font-weight:800;color:white;">'
                                    f'{r["confidence"]:.0%}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    st.markdown(f"### {selected_event} — All {len(fights)} Fights")

                    for r in results:
                        fight = r["fight"]
                        is_hot = r["confidence"] >= 0.70
                        title_tag = " 🥇" if fight["is_title"] else ""
                        hot_tag = '<span class="hot-pick-badge">HOT PICK</span>' if is_hot else ""
                        st.markdown(
                            f'<div class="fight-card">'
                            f'<div style="font-size:0.78rem;color:#888;margin-bottom:6px;">'
                            f'{fight["weight_class"]}{title_tag}{hot_tag}</div>'
                            + prob_bar_html(r["prob_r"], fight["r_fighter"], fight["b_fighter"]) +
                            f'<div style="font-size:0.78rem;color:#aaa;margin-top:6px;">'
                            f'Predicted winner: <strong style="color:{"#e63946" if r["prob_r"]>0.5 else "#4361ee"};">'
                            f'{r["pick"]}</strong> · {conf_badge(r["confidence"])}'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )

    with tab_recent:
        with st.spinner("Fetching recent results..."):
            try:
                recent = fetch_recent_events()
            except Exception as e:
                st.error(f"Could not reach ufcstats.com: {e}")
                recent = []

        if not recent:
            st.info("No recent events found.")
        else:
            for ev in recent:
                st.markdown(f"### {ev['name']}")
                st.markdown(f"*{ev['date']} · {ev['location']}*")
                with st.spinner("Loading results..."):
                    fights = fetch_completed_results(ev["url"])
                if fights:
                    rows = []
                    for f in fights:
                        r_stats = fetch_live_fighter(f["r_fighter"], "")
                        b_stats = fetch_live_fighter(f["b_fighter"], "")
                        prob_r, prob_b = predict_matchup(r_stats, b_stats)
                        model_pick = f["r_fighter"] if prob_r > 0.5 else f["b_fighter"]
                        actual_winner = f.get("winner", "")
                        correct = "✅" if actual_winner and model_pick == actual_winner else (
                            "❌" if actual_winner else "—")
                        rows.append({
                            "🔴 Red": f["r_fighter"],
                            "🔵 Blue": f["b_fighter"],
                            "Actual Winner": actual_winner or "—",
                            "Model Pick": model_pick,
                            "Correct": correct,
                            "Red %": f"{prob_r:.0%}",
                            "Blue %": f"{prob_b:.0%}",
                            "Method": f["method"],
                            "Round": f["round"],
                            "Weight Class": f["weight_class"],
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — FIGHT PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Fight Predictor":
    st.title("🥊 Fight Predictor")
    st.markdown("Pick two fighters and get an AI-powered win probability based on career stats.")

    col_r, col_vs, col_b = st.columns([5, 1, 5])
    with col_r:
        st.markdown("### 🔴 Red Corner")
        red_name = st.selectbox(
            "Select Fighter", fighter_names, key="red",
            index=fighter_names.index("Amanda Ribas") if "Amanda Ribas" in fighter_names else 0,
        )
    with col_vs:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;color:#555;'>VS</h2>", unsafe_allow_html=True)
    with col_b:
        st.markdown("### 🔵 Blue Corner")
        blue_name = st.selectbox(
            "Select Fighter", fighter_names, key="blue",
            index=fighter_names.index("Rose Namajunas") if "Rose Namajunas" in fighter_names else 1,
        )

    st.markdown("")
    predict_btn = st.button("⚡ Predict Fight", use_container_width=True, type="primary")

    if predict_btn:
        if red_name == blue_name:
            st.error("Please select two different fighters.")
        else:
            red  = fighters_df[fighters_df["name"] == red_name].iloc[0]
            blue = fighters_df[fighters_df["name"] == blue_name].iloc[0]

            def g(row, col):
                try:
                    v = row.get(col, 0)
                    return float(v) if v is not None and str(v) not in ("nan", "") else 0.0
                except Exception:
                    return 0.0

            feat_vals = {
                "sig_str_acc_total_diff": g(red, "sig_str_acc") - g(blue, "sig_str_acc"),
                "td_acc_total_diff":      g(red, "td_acc")      - g(blue, "td_acc"),
                "str_def_total_diff":     g(red, "str_def")     - g(blue, "str_def"),
                "td_def_total_diff":      g(red, "td_def")      - g(blue, "td_def"),
                "sub_avg_diff":           g(red, "sub_avg")     - g(blue, "sub_avg"),
                "td_avg_diff":            g(red, "td_avg")      - g(blue, "td_avg"),
                "SLpM_total_diff":        g(red, "SLpM")        - g(blue, "SLpM"),
                "SApM_total_diff":        g(red, "SApM")        - g(blue, "SApM"),
                "reach_diff":             g(red, "reach")       - g(blue, "reach"),
                "height_diff":            g(red, "height")      - g(blue, "height"),
                "age_diff":              g(red, "age")           - g(blue, "age"),
                "weight_diff":            g(red, "weight")      - g(blue, "weight"),
                "wins_total_diff":        g(red, "wins")        - g(blue, "wins"),
                "losses_total_diff":      g(red, "losses")      - g(blue, "losses"),
            }

            X_pred   = pd.DataFrame([feat_vals])[FEATURE_COLS]
            prob_red  = model.predict_proba(X_pred)[0, 1]
            prob_blue = 1 - prob_red
            winner    = red_name if prob_red > 0.5 else blue_name
            confidence = max(prob_red, prob_blue)

            st.markdown("---")

            # Winner banner
            win_color = "#e63946" if prob_red > 0.5 else "#4361ee"
            st.markdown(
                f'<div class="win-banner" style="background:{win_color}22;border:2px solid {win_color};">'
                f'🏆 {winner.upper()} WINS &nbsp;·&nbsp; {conf_badge(confidence)}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Probability bar
            st.markdown(prob_bar_html(prob_red, red_name, blue_name), unsafe_allow_html=True)

            st.markdown("")

            # Probability cards
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f'<div class="red-card">'
                    f'<div style="color:#e63946;font-weight:700;font-size:1.05rem;">{red_name}</div>'
                    f'<div style="font-size:2.8rem;font-weight:800;color:#e63946;margin:8px 0;">{prob_red:.1%}</div>'
                    f'<div style="color:#aaa;font-size:0.85rem;">Win Probability</div>'
                    f'<div style="color:#888;font-size:0.78rem;margin-top:4px;">'
                    f'{int(g(red,"wins"))}W – {int(g(red,"losses"))}L</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="blue-card">'
                    f'<div style="color:#4361ee;font-weight:700;font-size:1.05rem;">{blue_name}</div>'
                    f'<div style="font-size:2.8rem;font-weight:800;color:#4361ee;margin:8px 0;">{prob_blue:.1%}</div>'
                    f'<div style="color:#aaa;font-size:0.85rem;">Win Probability</div>'
                    f'<div style="color:#888;font-size:0.78rem;margin-top:4px;">'
                    f'{int(g(blue,"wins"))}W – {int(g(blue,"losses"))}L</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")
            st.markdown("### Head-to-Head Stats")

            pct = "{:.0%}"
            num = "{:.2f}"
            int_ = "{:.0f}"

            stats_list = [
                ("Striking Acc",   g(red,"sig_str_acc"), g(blue,"sig_str_acc"), pct, True),
                ("Striking Def",   g(red,"str_def"),     g(blue,"str_def"),     pct, True),
                ("TD Accuracy",    g(red,"td_acc"),      g(blue,"td_acc"),      pct, True),
                ("TD Defense",     g(red,"td_def"),      g(blue,"td_def"),      pct, True),
                ("Str/Min",        g(red,"SLpM"),        g(blue,"SLpM"),        num, True),
                ("Abs/Min",        g(red,"SApM"),        g(blue,"SApM"),        num, False),
                ("Sub Avg",        g(red,"sub_avg"),     g(blue,"sub_avg"),     num, True),
                ("TD Avg",         g(red,"td_avg"),      g(blue,"td_avg"),      num, True),
                ("Reach (cm)",     g(red,"reach"),       g(blue,"reach"),       int_, True),
                ("Age",            g(red,"age"),         g(blue,"age"),         int_, False),
            ]
            st.markdown(
                f'<div style="font-size:0.78rem;color:#888;display:flex;'
                f'justify-content:space-between;margin-bottom:4px;">'
                f'<span style="color:#e63946;">🔴 {red_name}</span>'
                f'<span style="color:#4361ee;">{blue_name} 🔵</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(stat_comparison_html(stats_list), unsafe_allow_html=True)

            st.markdown("")

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_red * 100,
                title={"text": f"{red_name} Win %", "font": {"color": "white", "size": 14}},
                number={"suffix": "%", "font": {"color": "#e63946", "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "gray"},
                    "bar": {"color": "#e63946"},
                    "bgcolor": "#1a1a2e",
                    "steps": [
                        {"range": [0, 50],  "color": "#16213e"},
                        {"range": [50, 100], "color": "#1a0505"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0e1117", font_color="white",
                height=260, margin=dict(t=60, b=10, l=30, r=30),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Radar comparison
            st.markdown("### Fighter Radar Comparison")
            radar_cats = ["Str Acc", "Str Def", "TD Acc", "TD Def", "Sub Avg", "SLpM"]
            max_vals = [1, 1, 1, 1,
                        max(fighters_df["sub_avg"].max(), 0.01),
                        max(fighters_df["SLpM"].max(), 0.01)]
            r_raw = [g(red,"sig_str_acc"), g(red,"str_def"), g(red,"td_acc"),
                     g(red,"td_def"), g(red,"sub_avg"), g(red,"SLpM")]
            b_raw = [g(blue,"sig_str_acc"), g(blue,"str_def"), g(blue,"td_acc"),
                     g(blue,"td_def"), g(blue,"sub_avg"), g(blue,"SLpM")]
            r_norm = [min(v/m, 1) if m > 0 else 0 for v, m in zip(r_raw, max_vals)]
            b_norm = [min(v/m, 1) if m > 0 else 0 for v, m in zip(b_raw, max_vals)]
            cats_c = radar_cats + [radar_cats[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=r_norm + [r_norm[0]], theta=cats_c,
                fill="toself", fillcolor="rgba(230,57,70,0.2)",
                line=dict(color="#e63946", width=2),
                name=red_name,
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=b_norm + [b_norm[0]], theta=cats_c,
                fill="toself", fillcolor="rgba(67,97,238,0.2)",
                line=dict(color="#4361ee", width=2),
                name=blue_name,
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#16213e",
                    radialaxis=dict(visible=True, range=[0,1], showticklabels=False, gridcolor="#333"),
                    angularaxis=dict(gridcolor="#333"),
                ),
                paper_bgcolor="#0e1117", font_color="white",
                height=380, margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(bgcolor="#0e1117", font=dict(color="white")),
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Dashboard":
    st.title("📊 Model Dashboard")
    st.markdown("HistGradientBoosting · Optuna-tuned · Time-series validated · 6,393 UFC fights (1994–2024)")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Test Accuracy",    f"{accuracy:.1%}"),
        ("ROC-AUC",          f"{roc_auc:.3f}"),
        ("vs Vegas Odds",    f"+{vs_vegas:.1%}"),
        ("Fights Trained",   f"{len(df):,}"),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], kpis):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    col_imp, col_cm = st.columns([3, 2])

    with col_imp:
        st.markdown("### Feature Importance")
        fig_imp = go.Figure(go.Bar(
            x=importances["importance"],
            y=importances["feature"],
            orientation="h",
            marker=dict(
                color=importances["importance"],
                colorscale=[[0, "#16213e"], [1, "#e63946"]],
                showscale=False,
            ),
            text=[f"{v:.1%}" for v in importances["importance"]],
            textposition="outside",
        ))
        fig_imp.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="white", height=480,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            margin=dict(l=10, r=60, t=20, b=20),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_cm:
        st.markdown("### Confusion Matrix")
        labels = ["Blue Wins", "Red Wins"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels, y=labels,
            color_continuous_scale=[[0, "#0e1117"], [1, "#e63946"]],
            text_auto=True,
        )
        fig_cm.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="white", height=320,
            margin=dict(l=10, r=10, t=20, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("### Model Info")
        st.markdown("""
| Param | Value |
|-------|-------|
| max_iter | 556 |
| max_depth | 3 |
| learning_rate | 0.016 |
| l2_regularization | 1.74 |
| min_samples_leaf | 20 |
| max_leaf_nodes | 45 |
| CV method | TimeSeriesSplit |
""")

    st.markdown("### Win Rate by Weight Class")
    wc_stats = (
        df.groupby("weight_class")["target"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Red Win Rate", "count": "Fights"})
        .query("Fights >= 20")
        .sort_values("Red Win Rate", ascending=False)
    )
    fig_wc = px.bar(
        wc_stats, x="weight_class", y="Red Win Rate",
        color="Red Win Rate",
        color_continuous_scale=[[0, "#4361ee"], [0.5, "#888"], [1, "#e63946"]],
        text=[f"{v:.0%}" for v in wc_stats["Red Win Rate"]],
        labels={"weight_class": "Weight Class", "Red Win Rate": "Red Win Rate"},
    )
    fig_wc.update_traces(textposition="outside")
    fig_wc.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
    fig_wc.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="white", height=360, showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(tickangle=-35),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_wc, use_container_width=True)

    st.markdown("### Finish Methods")
    method_counts = df["method"].value_counts().reset_index()
    method_counts.columns = ["Method", "Count"]
    fig_method = px.pie(
        method_counts, values="Count", names="Method",
        color_discrete_sequence=["#e63946", "#4361ee", "#f4a261", "#2a9d8f", "#e9c46a"],
        hole=0.45,
    )
    fig_method.update_layout(
        paper_bgcolor="#0e1117", font_color="white",
        height=340, margin=dict(l=10, r=10, t=20, b=20),
        legend=dict(bgcolor="#0e1117"),
    )
    st.plotly_chart(fig_method, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FIGHTER DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Fighter Database":
    st.title("🏟️ Fighter Database")
    st.markdown("Browse and compare fighters in the dataset.")

    col_s, col_st = st.columns([3, 1])
    with col_s:
        search = st.text_input("Search fighter", placeholder="e.g. McGregor")
    with col_st:
        stance_filter = st.selectbox("Stance", ["All"] + sorted(fighters_df["stance"].dropna().unique().tolist()))

    filtered = fighters_df.copy()
    if search:
        filtered = filtered[filtered["name"].str.contains(search, case=False, na=False)]
    if stance_filter != "All":
        filtered = filtered[filtered["stance"] == stance_filter]

    display_cols = ["name","wins","losses","height","weight","reach",
                    "stance","age","SLpM","sig_str_acc","str_def","td_acc","td_def","sub_avg"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.markdown(f"**{len(filtered):,} fighters**")
    st.dataframe(
        filtered[display_cols].sort_values("wins", ascending=False),
        hide_index=True, use_container_width=True, height=420,
    )

    st.markdown("---")
    st.markdown("### Fighter Profile")

    profile_name = st.selectbox("Select a fighter for detailed stats", fighter_names)
    if profile_name:
        p = fighters_df[fighters_df["name"] == profile_name].iloc[0]

        def gp(col):
            try:
                v = p.get(col, 0)
                return float(v) if v is not None and str(v) not in ("nan","") else 0.0
            except Exception:
                return 0.0

        pc1, pc2, pc3, pc4 = st.columns(4)
        stats_top = [
            ("Record",  f"{int(gp('wins'))}W – {int(gp('losses'))}L"),
            ("Reach",   f"{gp('reach'):.0f} cm"),
            ("Age",     f"{int(gp('age'))}"),
            ("Stance",  str(p.get("stance", "N/A"))),
        ]
        for col, (lbl, val) in zip([pc1, pc2, pc3, pc4], stats_top):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="font-size:1.4rem;">{val}</div>'
                    f'<div class="metric-label">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("")

        radar_cats = ["Str Acc", "Str Def", "TD Acc", "TD Def", "Sub Avg", "SLpM"]
        max_vals = [1, 1, 1, 1,
                    max(fighters_df["sub_avg"].max(), 0.01),
                    max(fighters_df["SLpM"].max(), 0.01)]
        raw_vals = [gp("sig_str_acc"), gp("str_def"), gp("td_acc"),
                    gp("td_def"), gp("sub_avg"), gp("SLpM")]
        norm_vals = [min(v/m, 1) if m > 0 else 0 for v, m in zip(raw_vals, max_vals)]
        cats_c    = radar_cats + [radar_cats[0]]
        norm_c    = norm_vals  + [norm_vals[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=norm_c, theta=cats_c,
            fill="toself", fillcolor="rgba(230,57,70,0.25)",
            line=dict(color="#e63946", width=2),
            name=profile_name,
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#16213e",
                radialaxis=dict(visible=True, range=[0,1], showticklabels=False, gridcolor="#333"),
                angularaxis=dict(gridcolor="#333"),
            ),
            paper_bgcolor="#0e1117", font_color="white",
            height=360, margin=dict(l=30, r=30, t=30, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        fight_history = df[(df["r_fighter"] == profile_name) | (df["b_fighter"] == profile_name)].copy()
        if not fight_history.empty:
            fight_history["Result"] = fight_history.apply(
                lambda row: "Win" if row["winner"] == (
                    "Red" if row["r_fighter"] == profile_name else "Blue") else "Loss",
                axis=1,
            )
            fight_history["Opponent"] = fight_history.apply(
                lambda row: row["b_fighter"] if row["r_fighter"] == profile_name else row["r_fighter"],
                axis=1,
            )
            st.markdown(f"### Fight History ({len(fight_history)} fights in dataset)")
            st.dataframe(
                fight_history[["event_name","Opponent","Result","method","weight_class"]]
                .rename(columns={"event_name":"Event","method":"Method","weight_class":"Weight Class"}),
                hide_index=True, use_container_width=True,
            )
