import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
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

DATA_PATH  = "data/large_dataset.csv"
CLEAN_PATH = "data/clean_dataset.csv"
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
    "td_acc_total_diff":      "Takedown Accuracy",
    "str_def_total_diff":     "Striking Defense",
    "td_def_total_diff":      "Takedown Defense",
    "sub_avg_diff":           "Submission Avg",
    "td_avg_diff":            "Takedown Avg",
    "SLpM_total_diff":        "Strikes Landed/Min",
    "SApM_total_diff":        "Strikes Absorbed/Min",
    "reach_diff":             "Reach (cm)",
    "height_diff":            "Height (cm)",
    "age_diff":               "Age (yrs)",
    "weight_diff":            "Weight (kg)",
    "wins_total_diff":        "Total Wins",
    "losses_total_diff":      "Total Losses",
}

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    section[data-testid="stSidebar"] { background-color: #12141c; }

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

    .corner-card-red {
        background: linear-gradient(160deg, #1f0507 0%, #2a0a0a 100%);
        border: 2px solid #e63946;
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 0 24px rgba(230,57,70,0.15);
        transition: box-shadow 0.3s;
    }
    .corner-card-red:hover { box-shadow: 0 0 36px rgba(230,57,70,0.3); }

    .corner-card-blue {
        background: linear-gradient(160deg, #05051f 0%, #0a0a2a 100%);
        border: 2px solid #4361ee;
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 0 24px rgba(67,97,238,0.15);
        transition: box-shadow 0.3s;
    }
    .corner-card-blue:hover { box-shadow: 0 0 36px rgba(67,97,238,0.3); }

    .vs-badge {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        font-size: 1.8rem;
        font-weight: 900;
        color: #444;
        letter-spacing: 2px;
    }

    .win-banner {
        font-size: 1.5rem;
        font-weight: 800;
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        margin: 16px 0;
        letter-spacing: 0.5px;
        animation: fadeIn 0.4s ease-in;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }

    .fight-card {
        background: linear-gradient(135deg, #14161f 0%, #1a1c2e 100%);
        border: 1px solid #2a2d3e;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .fight-card:hover { border-color: #e63946; box-shadow: 0 2px 16px rgba(230,57,70,0.1); }

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

    .prob-bar-wrap {
        display: flex;
        border-radius: 10px;
        overflow: hidden;
        height: 48px;
        margin: 10px 0;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.4);
    }
    .prob-bar-red {
        background: linear-gradient(90deg, #c1121f, #e63946);
        display: flex; align-items: center; justify-content: flex-end;
        padding: 0 14px; font-weight: 800; font-size: 1rem; color: white;
    }
    .prob-bar-blue {
        background: linear-gradient(90deg, #4361ee, #3a0ca3);
        display: flex; align-items: center; justify-content: flex-start;
        padding: 0 14px; font-weight: 800; font-size: 1rem; color: white;
    }

    .stat-row { display: flex; align-items: center; margin: 7px 0; gap: 8px; }
    .stat-label { color: #aaa; font-size: 0.8rem; min-width: 110px; text-align: center; flex-shrink: 0; }
    .stat-bar-red  { height: 20px; border-radius: 4px 0 0 4px; background: #e63946; min-width: 4px; }
    .stat-bar-blue { height: 20px; border-radius: 0 4px 4px 0; background: #4361ee; min-width: 4px; }
    .stat-val      { font-size: 0.82rem; color: #ddd; min-width: 42px; }
    .stat-val-r    { text-align: right; }

    .form-dot-w { display:inline-block; width:22px; height:22px; border-radius:50%;
        background:#2dc653; color:white; font-size:0.7rem; font-weight:700;
        text-align:center; line-height:22px; margin:1px; }
    .form-dot-l { display:inline-block; width:22px; height:22px; border-radius:50%;
        background:#e63946; color:white; font-size:0.7rem; font-weight:700;
        text-align:center; line-height:22px; margin:1px; }

    .conf-high { color: #2dc653; font-weight: 700; }
    .conf-med  { color: #f4a261; font-weight: 700; }
    .conf-low  { color: #aaa;    font-weight: 700; }

    h1 { color: #e63946 !important; }
    h2, h3 { color: #fff !important; }
    .stSelectbox label, .stTextInput label { color: #ccc !important; }

    @media (max-width: 768px) {
        .metric-value { font-size: 1.5rem; }
        .win-banner   { font-size: 1.1rem; padding: 14px; }
        .prob-bar-red, .prob-bar-blue { font-size: 0.85rem; padding: 0 8px; }
        .stat-label   { min-width: 80px; font-size: 0.72rem; }
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def prob_bar_html(prob_red, name_red, name_blue):
    wr, wb = int(prob_red * 100), int((1 - prob_red) * 100)
    return f"""
    <div style="margin:4px 0 2px 0;">
      <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:5px;">
        <span style="color:#e63946;font-weight:700">🔴 {name_red}</span>
        <span style="color:#4361ee;font-weight:700">{name_blue} 🔵</span>
      </div>
      <div class="prob-bar-wrap">
        <div class="prob-bar-red"  style="flex:{wr}">{wr}%</div>
        <div class="prob-bar-blue" style="flex:{wb}">{wb}%</div>
      </div>
    </div>"""


def stat_comparison_html(stats_list):
    rows = []
    for label, rv, bv, fmt, hib in stats_list:
        if rv is None or bv is None:
            continue
        total = abs(rv) + abs(bv)
        rw = int(abs(rv) / total * 100) if total else 50
        bw = 100 - rw
        r_win = (rv > bv) == hib
        rs = "font-weight:700;color:#e63946;" if r_win  else "color:#ccc;"
        bs = "font-weight:700;color:#4361ee;" if not r_win else "color:#ccc;"
        rows.append(f"""
        <div class="stat-row">
          <div class="stat-val stat-val-r" style="{rs}">{fmt.format(rv)}</div>
          <div class="stat-bar-red"  style="flex:{rw}"></div>
          <div class="stat-label">{label}</div>
          <div class="stat-bar-blue" style="flex:{bw}"></div>
          <div class="stat-val" style="{bs}">{fmt.format(bv)}</div>
        </div>""")
    return "<div>" + "".join(rows) + "</div>"


def conf_badge(c):
    if c > 0.72: return f'<span class="conf-high">HIGH ({c:.0%})</span>'
    if c > 0.62: return f'<span class="conf-med">MODERATE ({c:.0%})</span>'
    return f'<span class="conf-low">CLOSE ({c:.0%})</span>'


def method_category(method):
    m = str(method).lower()
    if "ko" in m or "tko" in m or "doctor" in m:
        return "KO/TKO"
    if "sub" in m:
        return "Submission"
    if "decision" in m:
        return "Decision"
    return "Other"


def g(row, col):
    try:
        v = row.get(col, 0) if hasattr(row, "get") else getattr(row, col, 0)
        return float(v) if v is not None and str(v) not in ("nan", "") else 0.0
    except Exception:
        return 0.0


# ── Data / Model ─────────────────────────────────────────────────────────────

@st.cache_data
def load_and_train():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLS + ["winner"])
    df["target"] = (df["winner"] == "Red").astype(int)
    df = df.iloc[::-1].reset_index(drop=True)

    X = df[FEATURE_COLS]
    y = df["target"]
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = HistGradientBoostingClassifier(
        max_iter=556, max_depth=3, learning_rate=0.0161,
        l2_regularization=1.744, min_samples_leaf=20,
        max_leaf_nodes=45, random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)

    baseline = roc_auc_score(y_test, y_proba)
    rng = np.random.default_rng(0)
    perm_imps = []
    for feat in FEATURE_COLS:
        Xp = X_test.copy()
        Xp[feat] = rng.permutation(Xp[feat].values)
        perm_imps.append(max(0, baseline - roc_auc_score(y_test, model.predict_proba(Xp)[:, 1])))
    total = sum(perm_imps) or 1
    importances = pd.DataFrame({
        "feature":    [FEATURE_LABELS[f] for f in FEATURE_COLS],
        "importance": [v / total for v in perm_imps],
    }).sort_values("importance", ascending=True)

    return model, df, acc, auc, cm, importances


@st.cache_data
def load_fighters():
    return pd.read_csv(FIGHTER_PATH)


@st.cache_data
def load_fight_history():
    """Load clean dataset with dates and build per-fighter history."""
    if not os.path.exists(CLEAN_PATH):
        return pd.DataFrame(), set()
    clean = pd.read_csv(CLEAN_PATH, parse_dates=["event_date"])
    clean = clean.sort_values("event_date").reset_index(drop=True)

    # Active fighters: fought since 2022
    red_last  = clean.groupby("r_fighter")["event_date"].max()
    blue_last = clean.groupby("b_fighter")["event_date"].max()
    last_fight = pd.concat([red_last, blue_last]).groupby(level=0).max()
    active = set(last_fight[last_fight >= "2022-01-01"].index)

    return clean, active


@st.cache_data(ttl=21600)
def fetch_all_events():        return scraper.get_all_events()

@st.cache_data(ttl=21600)
def fetch_upcoming_events():   return scraper.get_upcoming_events()

@st.cache_data(ttl=21600)
def fetch_recent_events():     return scraper.get_recent_completed_events(n=20)

@st.cache_data(ttl=21600)
def fetch_event_fights(url):   return scraper.get_event_fights(url)

@st.cache_data(ttl=21600)
def fetch_completed_results(url): return scraper.get_completed_event_results(url)

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
    feat = {
        "sig_str_acc_total_diff": g(red_stats,"sig_str_acc") - g(blue_stats,"sig_str_acc"),
        "td_acc_total_diff":      g(red_stats,"td_acc")      - g(blue_stats,"td_acc"),
        "str_def_total_diff":     g(red_stats,"str_def")     - g(blue_stats,"str_def"),
        "td_def_total_diff":      g(red_stats,"td_def")      - g(blue_stats,"td_def"),
        "sub_avg_diff":           g(red_stats,"sub_avg")     - g(blue_stats,"sub_avg"),
        "td_avg_diff":            g(red_stats,"td_avg")      - g(blue_stats,"td_avg"),
        "SLpM_total_diff":        g(red_stats,"SLpM")        - g(blue_stats,"SLpM"),
        "SApM_total_diff":        g(red_stats,"SApM")        - g(blue_stats,"SApM"),
        "reach_diff":             g(red_stats,"reach")       - g(blue_stats,"reach"),
        "height_diff":            g(red_stats,"height")      - g(blue_stats,"height"),
        "age_diff":               g(red_stats,"age")         - g(blue_stats,"age"),
        "weight_diff":            g(red_stats,"weight")      - g(blue_stats,"weight"),
        "wins_total_diff":        g(red_stats,"wins")        - g(blue_stats,"wins"),
        "losses_total_diff":      g(red_stats,"losses")      - g(blue_stats,"losses"),
    }
    prob_red = model.predict_proba(pd.DataFrame([feat])[FEATURE_COLS])[0, 1]
    return prob_red, 1 - prob_red


def get_fighter_history(name, clean_df, n=None):
    """Return a fighter's fights from clean_dataset, newest first."""
    mask = (clean_df["r_fighter"] == name) | (clean_df["b_fighter"] == name)
    hist = clean_df[mask].copy()
    hist["won"] = hist.apply(
        lambda r: r["winner"] == ("Red" if r["r_fighter"] == name else "Blue"), axis=1
    )
    hist["opponent"] = hist.apply(
        lambda r: r["b_fighter"] if r["r_fighter"] == name else r["r_fighter"], axis=1
    )
    hist = hist.sort_values("event_date", ascending=False)
    return hist.head(n) if n else hist


def method_donut(fight_hist, name):
    wins = fight_hist[fight_hist["won"]]
    if wins.empty:
        return None
    wins = wins.copy()
    wins["category"] = wins["method"].apply(method_category)
    counts = wins["category"].value_counts().reset_index()
    counts.columns = ["Method", "Count"]
    fig = px.pie(
        counts, values="Count", names="Method", hole=0.5,
        color="Method",
        color_discrete_map={
            "KO/TKO":     "#e63946",
            "Submission": "#4361ee",
            "Decision":   "#888",
            "Other":      "#2a9d8f",
        },
    )
    fig.update_layout(
        paper_bgcolor="#0e1117", font_color="white",
        height=260, margin=dict(l=10,r=10,t=30,b=10),
        legend=dict(bgcolor="#0e1117", font=dict(size=11)),
        title=dict(text=f"{name} — Win Methods", font=dict(color="white", size=13)),
    )
    return fig


# ── Bootstrap ─────────────────────────────────────────────────────────────────

model, df, accuracy, roc_auc, cm, importances = load_and_train()
fighters_df  = load_fighters()
clean_df, active_fighters = load_fight_history()
all_fighter_names    = sorted(fighters_df["name"].dropna().unique().tolist())
active_fighter_names = sorted(f for f in all_fighter_names if f in active_fighters)
vegas_baseline = 0.685
vs_vegas = accuracy - vegas_baseline


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥊 UFC Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Upcoming Events", "Fight Predictor", "Fighter Lookup", "Model Dashboard", "Fighter Database"],
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
    st.markdown("---")
    st.caption("Made by **Saransh Raina**")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — UPCOMING EVENTS
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Upcoming Events":
    st_autorefresh(interval=1_800_000, key="live_refresh")

    st.title("📅 Upcoming UFC Events")
    col_title, col_refresh = st.columns([5, 1])
    with col_title:
        st.markdown("Live fight cards from **ufcstats.com** with AI win predictions.")
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            fetch_upcoming_events.clear()
            fetch_recent_events.clear()
            fetch_event_fights.clear()
            fetch_completed_results.clear()
            fetch_live_fighter.clear()
            st.rerun()
    st.caption(f"Last updated: {datetime.now().strftime('%b %d, %Y · %I:%M %p')} · auto-refreshes every 30 min")

    tab_upcoming, tab_parlay, tab_recent, tab_calendar = st.tabs(["🔜 Upcoming Cards", "🎰 Parlay Builder", "🏁 Recent Results", "📋 Full Calendar"])

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
            selected_event = st.selectbox("Select Event", [e["name"] for e in events])
            ev = next(e for e in events if e["name"] == selected_event)
            c1, c2 = st.columns(2)
            c1.markdown(f"**Date:** {ev['date']}")
            c2.markdown(f"**Location:** {ev['location']}")

            st.markdown("")
            if st.button("⚡ Run AI Predictions", type="primary", use_container_width=True):
                with st.spinner("Fetching fighter stats and running predictions..."):
                    fights = fetch_event_fights(ev["url"])

                if not fights:
                    st.warning("Could not load fight card.")
                else:
                    results = []
                    prog = st.progress(0)
                    for i, fight in enumerate(fights):
                        r_stats = fetch_live_fighter(fight["r_fighter"], fight.get("r_url",""))
                        b_stats = fetch_live_fighter(fight["b_fighter"], fight.get("b_url",""))
                        prob_r, prob_b = predict_matchup(r_stats, b_stats)
                        confidence = max(prob_r, prob_b)
                        results.append({
                            "fight": fight, "prob_r": prob_r, "prob_b": prob_b,
                            "confidence": confidence,
                            "pick": fight["r_fighter"] if prob_r > 0.5 else fight["b_fighter"],
                        })
                        prog.progress((i + 1) / len(fights))
                    prog.empty()

                    hot = [r for r in results if r["confidence"] >= 0.70]
                    if hot:
                        st.markdown("### 🔥 Hot Picks (≥70% confidence)")
                        hp_cols = st.columns(min(len(hot), 3))
                        for col, r in zip(hp_cols, hot[:3]):
                            color = "#e63946" if r["prob_r"] > 0.5 else "#4361ee"
                            with col:
                                st.markdown(
                                    f'<div style="background:{color}22;border:2px solid {color};'
                                    f'border-radius:10px;padding:14px;text-align:center;">'
                                    f'<div style="font-size:0.75rem;color:#aaa;">{r["fight"]["weight_class"]}</div>'
                                    f'<div style="font-weight:700;font-size:1rem;color:{color};">{r["pick"]}</div>'
                                    f'<div style="font-size:1.4rem;font-weight:800;color:white;">{r["confidence"]:.0%}</div>'
                                    f'</div>', unsafe_allow_html=True,
                                )

                    st.markdown(f"### {selected_event} — {len(fights)} Fights")
                    for r in results:
                        fight = r["fight"]
                        is_hot   = r["confidence"] >= 0.70
                        title_tag = " 🥇" if fight["is_title"] else ""
                        hot_tag   = '<span class="hot-pick-badge">HOT PICK</span>' if is_hot else ""
                        st.markdown(
                            f'<div class="fight-card">'
                            f'<div style="font-size:0.78rem;color:#888;margin-bottom:6px;">'
                            f'{fight["weight_class"]}{title_tag}{hot_tag}</div>'
                            + prob_bar_html(r["prob_r"], fight["r_fighter"], fight["b_fighter"]) +
                            f'<div style="font-size:0.78rem;color:#aaa;margin-top:6px;">'
                            f'Pick: <strong style="color:{"#e63946" if r["prob_r"]>0.5 else "#4361ee"};">'
                            f'{r["pick"]}</strong> · {conf_badge(r["confidence"])}'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )

    with tab_parlay:
        st.markdown("### 🎰 Parlay Builder")
        st.markdown("Pick 2 or 3 fights from the current card and we'll calculate your combined parlay probability.")

        with st.spinner("Loading fight card..."):
            try:
                parlay_events = fetch_upcoming_events()
            except Exception:
                parlay_events = []

        if not parlay_events:
            st.info("No upcoming events found. Check back closer to fight week.")
        else:
            p_event_names = [e["name"] for e in parlay_events]
            p_selected = st.selectbox("Select Event", p_event_names, key="parlay_event")
            p_ev = next(e for e in parlay_events if e["name"] == p_selected)

            with st.spinner("Loading fight card..."):
                p_fights = fetch_event_fights(p_ev["url"])

            if not p_fights:
                st.warning("Could not load fight card.")
            else:
                fight_labels = [f"{f['r_fighter']} vs {f['b_fighter']} ({f['weight_class']})" for f in p_fights]
                num_legs = st.radio("Legs", [2, 3], horizontal=True, key="parlay_legs")

                leg_selections = []
                colors = ["#e63946", "#4361ee", "#2a9d8f"]
                cols = st.columns(num_legs)
                for i, col in enumerate(cols):
                    with col:
                        color = colors[i]
                        st.markdown(
                            f'<div style="border:2px solid {color};border-radius:10px;padding:10px 14px;margin-bottom:8px;">'
                            f'<div style="color:{color};font-weight:700;font-size:0.85rem;">LEG {i+1}</div></div>',
                            unsafe_allow_html=True,
                        )
                        fight_pick = st.selectbox("Fight", ["— select —"] + fight_labels, key=f"pleg_fight{i}")
                        if fight_pick != "— select —":
                            idx = fight_labels.index(fight_pick)
                            fight = p_fights[idx]
                            winner_pick = st.radio(
                                "Pick", [fight["r_fighter"], fight["b_fighter"]],
                                key=f"pleg_pick{i}", horizontal=True,
                            )
                            leg_selections.append({"fight": fight, "pick": winner_pick, "color": color, "idx": i+1})
                        else:
                            leg_selections.append(None)

                st.markdown("")
                if st.button("⚡ Calculate Parlay", type="primary", use_container_width=True):
                    valid_legs = [l for l in leg_selections if l is not None]
                    if len(valid_legs) < 2:
                        st.error("Select at least 2 fights.")
                    else:
                        combined_prob = 1.0
                        leg_results = []
                        for leg in valid_legs:
                            fight = leg["fight"]
                            r_stats = fetch_live_fighter(fight["r_fighter"], fight.get("r_url",""))
                            b_stats = fetch_live_fighter(fight["b_fighter"], fight.get("b_url",""))
                            prob_r, prob_b = predict_matchup(r_stats, b_stats)
                            picked_prob = prob_r if leg["pick"] == fight["r_fighter"] else prob_b
                            combined_prob *= picked_prob
                            leg_results.append({
                                "idx": leg["idx"], "color": leg["color"],
                                "r": fight["r_fighter"], "b": fight["b_fighter"],
                                "prob_r": prob_r, "prob_b": prob_b,
                                "pick": leg["pick"], "picked_prob": picked_prob,
                            })

                        st.markdown("---")
                        for res in leg_results:
                            st.markdown(
                                f'<div class="fight-card">'
                                f'<div style="color:{res["color"]};font-weight:700;font-size:0.8rem;margin-bottom:6px;">LEG {res["idx"]}</div>'
                                + prob_bar_html(res["prob_r"], res["r"], res["b"]) +
                                f'<div style="font-size:0.8rem;color:#aaa;margin-top:6px;">'
                                f'Your pick: <strong style="color:{res["color"]};">{res["pick"]}</strong>'
                                f' · Probability: <strong style="color:white;">{res["picked_prob"]:.1%}</strong>'
                                f'</div></div>',
                                unsafe_allow_html=True,
                            )

                        implied = 1 / combined_prob if combined_prob > 0 else 0
                        amer = int((implied - 1) * 100) if implied >= 2 else int(-100 / max(implied - 1, 0.01))
                        conf_color = "#2dc653" if combined_prob > 0.40 else ("#f4a261" if combined_prob > 0.22 else "#e63946")
                        st.markdown(
                            f'<div style="background:{conf_color}18;border:2px solid {conf_color};'
                            f'border-radius:14px;padding:24px;text-align:center;margin-top:10px;">'
                            f'<div style="font-size:0.82rem;color:#aaa;margin-bottom:4px;">{len(valid_legs)}-LEG PARLAY PROBABILITY</div>'
                            f'<div style="font-size:3.5rem;font-weight:900;color:{conf_color};">{combined_prob:.1%}</div>'
                            f'<div style="font-size:0.95rem;color:#ccc;margin-top:8px;">'
                            f'Implied: <strong>{implied:.2f}x</strong> &nbsp;·&nbsp; '
                            f'American: <strong>{"+" if amer > 0 else ""}{amer}</strong></div>'
                            f'<div style="font-size:0.75rem;color:#555;margin-top:10px;">AI probabilities only — not financial advice.</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    with tab_calendar:
        st.markdown("### 📋 UFC Event Calendar — Last 6 Months & Next 6 Months")
        with st.spinner("Loading event record..."):
            try:
                all_evs = fetch_all_events()
            except Exception as e:
                st.error(f"Could not load events: {e}")
                all_evs = []

        if not all_evs:
            st.info("No events found.")
        else:
            upcoming_evs  = [e for e in all_evs if not e["completed"]]
            completed_evs = [e for e in all_evs if e["completed"]]
            completed_evs.sort(key=lambda e: e["date"], reverse=True)

            if upcoming_evs:
                st.markdown("#### Upcoming")
                for ev in upcoming_evs:
                    bouts = ev.get("bouts", [])
                    main  = bouts[-1] if bouts else None
                    main_str = f"{main['r_fighter']} vs {main['b_fighter']}" if main else ""
                    st.markdown(
                        f'<div class="fight-card" style="margin-bottom:8px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<div>'
                        f'<span style="color:white;font-weight:700;">{ev["name"]}</span><br>'
                        f'<span style="color:#aaa;font-size:0.8rem;">{ev["date"]} · {ev["location"]}</span><br>'
                        f'<span style="color:#888;font-size:0.78rem;">{main_str} · {len(bouts)} bouts</span>'
                        f'</div>'
                        f'<span style="background:#2dc65322;color:#2dc653;border:1px solid #2dc653;'
                        f'border-radius:20px;padding:3px 12px;font-size:0.75rem;font-weight:700;">UPCOMING</span>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("#### Past Results (last 6 months)")
            for ev in completed_evs:
                bouts   = ev.get("bouts", [])
                main    = bouts[-1] if bouts else None
                winner  = main.get("winner","") if main else ""
                loser   = main.get("b_fighter","") if main and winner == main.get("r_fighter") else (main.get("r_fighter","") if main else "")
                result_str = f"{winner} def. {loser} · {main.get('method','')}" if winner else ""
                with st.expander(f"{ev['name']} — {ev['date']}"):
                    st.markdown(f"**Location:** {ev['location']}")
                    if result_str:
                        st.markdown(f"**Main event:** {result_str}")
                    if bouts:
                        rows = []
                        for b in bouts:
                            rows.append({
                                "Fighter A": b["r_fighter"],
                                "Fighter B": b["b_fighter"],
                                "Winner": b.get("winner","—") or "—",
                                "Method": b.get("method","—") or "—",
                                "Class": b.get("weight_class",""),
                            })
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

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
                        model_pick   = f["r_fighter"] if prob_r > 0.5 else f["b_fighter"]
                        actual_winner = f.get("winner","")
                        correct = "✅" if actual_winner and model_pick == actual_winner else ("❌" if actual_winner else "—")
                        rows.append({
                            "🔴 Red": f["r_fighter"], "🔵 Blue": f["b_fighter"],
                            "Actual Winner": actual_winner or "—", "Model Pick": model_pick,
                            "✓": correct, "Red %": f"{prob_r:.0%}", "Blue %": f"{prob_b:.0%}",
                            "Method": f["method"], "Rd": f["round"], "Class": f["weight_class"],
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — FIGHT PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Fight Predictor":
    st.title("🥊 Fight Predictor")
    st.markdown("Select two fighters and get an AI-powered win probability.")

    show_all = st.checkbox("Show inactive / retired fighters", value=False)
    names_to_use = all_fighter_names if show_all else active_fighter_names
    inactive_note = "" if show_all else f"Showing {len(active_fighter_names)} active fighters (fought since 2022)"
    if inactive_note:
        st.caption(inactive_note)

    st.markdown("")
    col_r, col_vs, col_b = st.columns([5, 1, 5])

    with col_r:
        st.markdown(
            '<div class="corner-card-red"><div style="color:#e63946;font-size:0.9rem;font-weight:700;'
            'letter-spacing:1px;margin-bottom:12px;">🔴 RED CORNER</div></div>',
            unsafe_allow_html=True,
        )
        default_r = names_to_use.index("Islam Makhachev") if "Islam Makhachev" in names_to_use else 0
        red_name = st.selectbox("Red fighter", names_to_use, index=default_r, key="red",
                                label_visibility="collapsed")

    with col_vs:
        st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown(
            '<div class="corner-card-blue"><div style="color:#4361ee;font-size:0.9rem;font-weight:700;'
            'letter-spacing:1px;margin-bottom:12px;">🔵 BLUE CORNER</div></div>',
            unsafe_allow_html=True,
        )
        default_b = names_to_use.index("Alexander Volkanovski") if "Alexander Volkanovski" in names_to_use else min(1, len(names_to_use)-1)
        blue_name = st.selectbox("Blue fighter", names_to_use, index=default_b, key="blue",
                                 label_visibility="collapsed")

    st.markdown("")
    predict_btn = st.button("⚡ Predict Fight", use_container_width=True, type="primary")

    if predict_btn:
        if red_name == blue_name:
            st.error("Select two different fighters.")
        else:
            red  = fighters_df[fighters_df["name"] == red_name].iloc[0]
            blue = fighters_df[fighters_df["name"] == blue_name].iloc[0]

            feat_vals = {
                "sig_str_acc_total_diff": g(red,"sig_str_acc") - g(blue,"sig_str_acc"),
                "td_acc_total_diff":      g(red,"td_acc")      - g(blue,"td_acc"),
                "str_def_total_diff":     g(red,"str_def")     - g(blue,"str_def"),
                "td_def_total_diff":      g(red,"td_def")      - g(blue,"td_def"),
                "sub_avg_diff":           g(red,"sub_avg")     - g(blue,"sub_avg"),
                "td_avg_diff":            g(red,"td_avg")      - g(blue,"td_avg"),
                "SLpM_total_diff":        g(red,"SLpM")        - g(blue,"SLpM"),
                "SApM_total_diff":        g(red,"SApM")        - g(blue,"SApM"),
                "reach_diff":             g(red,"reach")       - g(blue,"reach"),
                "height_diff":            g(red,"height")      - g(blue,"height"),
                "age_diff":               g(red,"age")         - g(blue,"age"),
                "weight_diff":            g(red,"weight")      - g(blue,"weight"),
                "wins_total_diff":        g(red,"wins")        - g(blue,"wins"),
                "losses_total_diff":      g(red,"losses")      - g(blue,"losses"),
            }
            prob_red  = model.predict_proba(pd.DataFrame([feat_vals])[FEATURE_COLS])[0, 1]
            prob_blue = 1 - prob_red
            winner    = red_name if prob_red > 0.5 else blue_name
            confidence = max(prob_red, prob_blue)

            st.markdown("---")
            win_color = "#e63946" if prob_red > 0.5 else "#4361ee"
            st.markdown(
                f'<div class="win-banner" style="background:{win_color}22;border:2px solid {win_color};">'
                f'🏆 {winner.upper()} WINS &nbsp;·&nbsp; {conf_badge(confidence)}'
                f'</div>', unsafe_allow_html=True,
            )

            st.markdown(prob_bar_html(prob_red, red_name, blue_name), unsafe_allow_html=True)
            st.markdown("")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f'<div class="corner-card-red">'
                    f'<div style="color:#e63946;font-weight:700;">{red_name}</div>'
                    f'<div style="font-size:3rem;font-weight:900;color:#e63946;margin:8px 0;">{prob_red:.1%}</div>'
                    f'<div style="color:#aaa;font-size:0.85rem;">Win Probability</div>'
                    f'<div style="color:#888;font-size:0.8rem;margin-top:6px;">'
                    f'{int(g(red,"wins"))}W – {int(g(red,"losses"))}L</div>'
                    f'</div>', unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="corner-card-blue">'
                    f'<div style="color:#4361ee;font-weight:700;">{blue_name}</div>'
                    f'<div style="font-size:3rem;font-weight:900;color:#4361ee;margin:8px 0;">{prob_blue:.1%}</div>'
                    f'<div style="color:#aaa;font-size:0.85rem;">Win Probability</div>'
                    f'<div style="color:#888;font-size:0.8rem;margin-top:6px;">'
                    f'{int(g(blue,"wins"))}W – {int(g(blue,"losses"))}L</div>'
                    f'</div>', unsafe_allow_html=True,
                )

            st.markdown("")
            st.markdown("### Head-to-Head Stats")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:6px;">'
                f'<span style="color:#e63946;font-weight:700;">🔴 {red_name}</span>'
                f'<span style="color:#4361ee;font-weight:700;">{blue_name} 🔵</span></div>',
                unsafe_allow_html=True,
            )
            stats_list = [
                ("Striking Acc",  g(red,"sig_str_acc"), g(blue,"sig_str_acc"), "{:.0%}", True),
                ("Striking Def",  g(red,"str_def"),     g(blue,"str_def"),     "{:.0%}", True),
                ("TD Accuracy",   g(red,"td_acc"),      g(blue,"td_acc"),      "{:.0%}", True),
                ("TD Defense",    g(red,"td_def"),      g(blue,"td_def"),      "{:.0%}", True),
                ("Str/Min",       g(red,"SLpM"),        g(blue,"SLpM"),        "{:.2f}", True),
                ("Abs/Min",       g(red,"SApM"),        g(blue,"SApM"),        "{:.2f}", False),
                ("Sub Avg",       g(red,"sub_avg"),     g(blue,"sub_avg"),     "{:.2f}", True),
                ("TD Avg",        g(red,"td_avg"),      g(blue,"td_avg"),      "{:.2f}", True),
                ("Reach (cm)",    g(red,"reach"),       g(blue,"reach"),       "{:.0f}", True),
                ("Age",           g(red,"age"),         g(blue,"age"),         "{:.0f}", False),
            ]
            st.markdown(stat_comparison_html(stats_list), unsafe_allow_html=True)
            st.markdown("")

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob_red * 100,
                title={"text": f"{red_name} Win %", "font": {"color": "white", "size": 13}},
                number={"suffix": "%", "font": {"color": "#e63946", "size": 36}},
                gauge={
                    "axis": {"range": [0,100], "tickcolor": "gray"},
                    "bar": {"color": "#e63946"}, "bgcolor": "#1a1a2e",
                    "steps": [{"range":[0,50],"color":"#16213e"},{"range":[50,100],"color":"#1a0505"}],
                    "threshold": {"line":{"color":"white","width":3},"thickness":0.8,"value":50},
                },
            ))
            fig_gauge.update_layout(paper_bgcolor="#0e1117", font_color="white",
                                    height=260, margin=dict(t=60,b=10,l=30,r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Dual radar
            st.markdown("### Style Comparison")
            radar_cats = ["Str Acc", "Str Def", "TD Acc", "TD Def", "Sub Avg", "SLpM"]
            max_v = [1, 1, 1, 1, max(fighters_df["sub_avg"].max(), 0.01), max(fighters_df["SLpM"].max(), 0.01)]
            r_raw = [g(red,"sig_str_acc"),g(red,"str_def"),g(red,"td_acc"),g(red,"td_def"),g(red,"sub_avg"),g(red,"SLpM")]
            b_raw = [g(blue,"sig_str_acc"),g(blue,"str_def"),g(blue,"td_acc"),g(blue,"td_def"),g(blue,"sub_avg"),g(blue,"SLpM")]
            r_n = [min(v/m,1) if m>0 else 0 for v,m in zip(r_raw,max_v)]
            b_n = [min(v/m,1) if m>0 else 0 for v,m in zip(b_raw,max_v)]
            cats_c = radar_cats + [radar_cats[0]]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=r_n+[r_n[0]], theta=cats_c, fill="toself",
                fillcolor="rgba(230,57,70,0.2)", line=dict(color="#e63946",width=2), name=red_name))
            fig_radar.add_trace(go.Scatterpolar(r=b_n+[b_n[0]], theta=cats_c, fill="toself",
                fillcolor="rgba(67,97,238,0.2)", line=dict(color="#4361ee",width=2), name=blue_name))
            fig_radar.update_layout(
                polar=dict(bgcolor="#16213e",
                    radialaxis=dict(visible=True,range=[0,1],showticklabels=False,gridcolor="#333"),
                    angularaxis=dict(gridcolor="#333")),
                paper_bgcolor="#0e1117", font_color="white",
                height=380, margin=dict(l=40,r=40,t=40,b=40),
                legend=dict(bgcolor="#0e1117",font=dict(color="white")),
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FIGHTER LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Fighter Lookup":
    st.title("🔍 Fighter Lookup")
    st.markdown("Search one or two fighters to see their record, win methods, and recent form.")

    show_all_lu = st.checkbox("Include inactive / retired fighters", value=False, key="lu_all")
    names_lu = all_fighter_names if show_all_lu else active_fighter_names

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f1 = st.selectbox("Fighter 1", ["— select —"] + names_lu, key="lu_f1")
    with col_f2:
        f2 = st.selectbox("Fighter 2 (optional)", ["— none —"] + names_lu, key="lu_f2")

    def render_fighter_card(name, color, side_label):
        if clean_df.empty:
            st.warning("Fight history not available.")
            return
        row = fighters_df[fighters_df["name"] == name]
        if row.empty:
            st.warning(f"{name} not found in database.")
            return
        row = row.iloc[0]
        hist = get_fighter_history(name, clean_df)

        wins   = int(hist["won"].sum())
        losses = int((~hist["won"]).sum())
        total  = wins + losses

        # Recent form dots (last 8)
        recent8 = hist.head(8)
        dots = "".join(
            f'<span class="form-dot-w">W</span>' if w else f'<span class="form-dot-l">L</span>'
            for w in recent8["won"]
        )
        streak_label = ""
        if not hist.empty:
            streak_val = 1
            first_res = hist.iloc[0]["won"]
            for _, r in hist.iloc[1:].iterrows():
                if r["won"] == first_res:
                    streak_val += 1
                else:
                    break
            streak_label = f'{"🔥 " + str(streak_val) + "-fight WIN streak" if first_res else "❄️ " + str(streak_val) + "-fight loss streak"}'

        st.markdown(
            f'<div style="background:{"#1f0507" if color=="#e63946" else "#05051f"};'
            f'border:2px solid {color};border-radius:16px;padding:20px;margin-bottom:16px;">'
            f'<div style="font-size:0.8rem;color:#888;font-weight:600;letter-spacing:1px;">{side_label}</div>'
            f'<div style="font-size:1.4rem;font-weight:800;color:{color};margin:4px 0;">{name}</div>'
            f'<div style="font-size:2rem;font-weight:900;color:white;">{wins}-{losses}</div>'
            f'<div style="font-size:0.82rem;color:#aaa;">W – L  ({total} fights in dataset)</div>'
            f'{"<div style=margin-top:8px;font-size:0.82rem;color:#f4a261;>" + streak_label + "</div>" if streak_label else ""}'
            f'<div style="margin-top:10px;">{dots}</div>'
            f'<div style="font-size:0.72rem;color:#666;margin-top:4px;">← most recent</div>'
            f'</div>', unsafe_allow_html=True,
        )

        # Win method donut
        fig_donut = method_donut(hist, name)
        if fig_donut:
            st.plotly_chart(fig_donut, use_container_width=True)

        # Recent fights table
        st.markdown(f"**Last {min(10, len(hist))} fights**")
        recent_display = hist.head(10)[["event_date","opponent","won","method","weight_class"]].copy()
        recent_display["event_date"] = recent_display["event_date"].dt.strftime("%b %Y")
        recent_display["Result"] = recent_display["won"].map({True: "✅ Win", False: "❌ Loss"})
        recent_display = recent_display.rename(columns={
            "event_date": "Date", "opponent": "Opponent",
            "method": "Method", "weight_class": "Class",
        })[["Date","Opponent","Result","Method","Class"]]
        st.dataframe(recent_display, hide_index=True, use_container_width=True)

    if f1 != "— select —" and f2 != "— none —" and f1 != f2:
        col_l, col_r = st.columns(2)
        with col_l:
            render_fighter_card(f1, "#e63946", "FIGHTER 1")
        with col_r:
            render_fighter_card(f2, "#4361ee", "FIGHTER 2")

        # Quick predict button
        st.markdown("---")
        if st.button(f"⚡ Predict {f1} vs {f2}", type="primary", use_container_width=True):
            r_stats = fighters_df[fighters_df["name"]==f1].iloc[0].to_dict() if not fighters_df[fighters_df["name"]==f1].empty else {}
            b_stats = fighters_df[fighters_df["name"]==f2].iloc[0].to_dict() if not fighters_df[fighters_df["name"]==f2].empty else {}
            if r_stats and b_stats:
                pr, pb = predict_matchup(r_stats, b_stats)
                st.markdown(prob_bar_html(pr, f1, f2), unsafe_allow_html=True)
                winner = f1 if pr > 0.5 else f2
                wc = "#e63946" if pr > 0.5 else "#4361ee"
                st.markdown(
                    f'<div class="win-banner" style="background:{wc}22;border:2px solid {wc};">'
                    f'🏆 {winner.upper()} WINS · {conf_badge(max(pr,pb))}</div>',
                    unsafe_allow_html=True,
                )

    elif f1 != "— select —":
        render_fighter_card(f1, "#e63946", "FIGHTER")

    else:
        st.info("Select a fighter above to see their profile.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Dashboard":
    st.title("📊 Model Dashboard")
    st.markdown("HistGradientBoosting · Optuna-tuned · Time-series validated · 6,393 UFC fights (1994–2024)")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, (label, val) in zip([c1,c2,c3,c4], [
        ("Test Accuracy", f"{accuracy:.1%}"),
        ("ROC-AUC",       f"{roc_auc:.3f}"),
        ("vs Vegas Odds", f"+{vs_vegas:.1%}"),
        ("Fights Trained",f"{len(df):,}"),
    ]):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    col_imp, col_cm = st.columns([3, 2])

    with col_imp:
        st.markdown("### Feature Importance")
        fig_imp = go.Figure(go.Bar(
            x=importances["importance"], y=importances["feature"], orientation="h",
            marker=dict(color=importances["importance"],
                        colorscale=[[0,"#16213e"],[1,"#e63946"]], showscale=False),
            text=[f"{v:.1%}" for v in importances["importance"]], textposition="outside",
        ))
        fig_imp.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
            height=480, xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            yaxis=dict(showgrid=False), margin=dict(l=10,r=60,t=20,b=20))
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_cm:
        st.markdown("### Confusion Matrix")
        labels = ["Blue Wins","Red Wins"]
        fig_cm = px.imshow(cm, labels=dict(x="Predicted",y="Actual",color="Count"),
            x=labels, y=labels,
            color_continuous_scale=[[0,"#0e1117"],[1,"#e63946"]], text_auto=True)
        fig_cm.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
            height=320, margin=dict(l=10,r=10,t=20,b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("### Model Params")
        st.markdown("""
| Param | Value |
|-------|-------|
| max_iter | 556 |
| max_depth | 3 |
| learning_rate | 0.016 |
| l2_regularization | 1.74 |
| min_samples_leaf | 20 |
| max_leaf_nodes | 45 |
| CV | TimeSeriesSplit |
""")

    st.markdown("### Win Rate by Weight Class")
    wc_stats = (df.groupby("weight_class")["target"].agg(["mean","count"]).reset_index()
        .rename(columns={"mean":"Red Win Rate","count":"Fights"}).query("Fights >= 20")
        .sort_values("Red Win Rate", ascending=False))
    fig_wc = px.bar(wc_stats, x="weight_class", y="Red Win Rate", color="Red Win Rate",
        color_continuous_scale=[[0,"#4361ee"],[0.5,"#888"],[1,"#e63946"]],
        text=[f"{v:.0%}" for v in wc_stats["Red Win Rate"]],
        labels={"weight_class":"Weight Class","Red Win Rate":"Red Win Rate"})
    fig_wc.update_traces(textposition="outside")
    fig_wc.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
    fig_wc.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
        height=360, showlegend=False, coloraxis_showscale=False,
        xaxis=dict(tickangle=-35), margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_wc, use_container_width=True)

    st.markdown("### Finish Methods")
    mc = df["method"].value_counts().reset_index()
    mc.columns = ["Method","Count"]
    fig_m = px.pie(mc, values="Count", names="Method", hole=0.45,
        color_discrete_sequence=["#e63946","#4361ee","#f4a261","#2a9d8f","#e9c46a"])
    fig_m.update_layout(paper_bgcolor="#0e1117", font_color="white",
        height=340, margin=dict(l=10,r=10,t=20,b=20), legend=dict(bgcolor="#0e1117"))
    st.plotly_chart(fig_m, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FIGHTER DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Fighter Database":
    st.title("🏟️ Fighter Database")
    st.markdown("Browse, filter, and profile all fighters in the dataset.")

    show_all_db = st.checkbox("Show inactive / retired fighters", value=False, key="db_all")
    base_pool = fighters_df if show_all_db else fighters_df[fighters_df["name"].isin(active_fighters)]

    col_s, col_st = st.columns([3, 1])
    with col_s:
        search = st.text_input("Search fighter", placeholder="e.g. McGregor")
    with col_st:
        stance_filter = st.selectbox("Stance", ["All"] + sorted(fighters_df["stance"].dropna().unique().tolist()))

    filtered = base_pool.copy()
    if search:
        filtered = filtered[filtered["name"].str.contains(search, case=False, na=False)]
    if stance_filter != "All":
        filtered = filtered[filtered["stance"] == stance_filter]

    display_cols = ["name","wins","losses","height","weight","reach",
                    "stance","age","SLpM","sig_str_acc","str_def","td_acc","td_def","sub_avg"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.markdown(f"**{len(filtered):,} fighters**")
    st.dataframe(filtered[display_cols].sort_values("wins", ascending=False),
                 hide_index=True, use_container_width=True, height=420)

    st.markdown("---")
    st.markdown("### Fighter Profile")

    pool_names = sorted(base_pool["name"].dropna().unique().tolist())
    profile_name = st.selectbox("Select a fighter", pool_names)
    if profile_name:
        p = fighters_df[fighters_df["name"] == profile_name].iloc[0]

        pc1, pc2, pc3, pc4 = st.columns(4)
        for col, (lbl, val) in zip([pc1,pc2,pc3,pc4], [
            ("Record",  f"{int(g(p,'wins'))}W – {int(g(p,'losses'))}L"),
            ("Reach",   f"{g(p,'reach'):.0f} cm"),
            ("Age",     f"{int(g(p,'age'))}"),
            ("Stance",  str(p.get("stance","N/A"))),
        ]):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="font-size:1.4rem;">{val}</div>'
                    f'<div class="metric-label">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("")
        radar_cats = ["Str Acc","Str Def","TD Acc","TD Def","Sub Avg","SLpM"]
        max_v = [1,1,1,1,max(fighters_df["sub_avg"].max(),0.01),max(fighters_df["SLpM"].max(),0.01)]
        raw_v = [g(p,"sig_str_acc"),g(p,"str_def"),g(p,"td_acc"),g(p,"td_def"),g(p,"sub_avg"),g(p,"SLpM")]
        norm_v = [min(v/m,1) if m>0 else 0 for v,m in zip(raw_v,max_v)]
        cats_c = radar_cats+[radar_cats[0]]; norm_c = norm_v+[norm_v[0]]
        fig_r = go.Figure(go.Scatterpolar(r=norm_c, theta=cats_c, fill="toself",
            fillcolor="rgba(230,57,70,0.25)", line=dict(color="#e63946",width=2), name=profile_name))
        fig_r.update_layout(polar=dict(bgcolor="#16213e",
            radialaxis=dict(visible=True,range=[0,1],showticklabels=False,gridcolor="#333"),
            angularaxis=dict(gridcolor="#333")),
            paper_bgcolor="#0e1117", font_color="white",
            height=360, margin=dict(l=30,r=30,t=30,b=30), showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)

        if not clean_df.empty:
            hist = get_fighter_history(profile_name, clean_df)
            if not hist.empty:
                col_donut, col_hist = st.columns([1, 2])
                with col_donut:
                    fig_d = method_donut(hist, profile_name)
                    if fig_d:
                        st.plotly_chart(fig_d, use_container_width=True)
                with col_hist:
                    st.markdown(f"**Fight History ({len(hist)} fights)**")
                    disp = hist[["event_date","opponent","won","method","weight_class"]].copy()
                    disp["event_date"] = disp["event_date"].dt.strftime("%b %Y")
                    disp["Result"] = disp["won"].map({True:"✅ Win",False:"❌ Loss"})
                    disp = disp.rename(columns={"event_date":"Date","opponent":"Opponent",
                                                "method":"Method","weight_class":"Class"})
                    st.dataframe(disp[["Date","Opponent","Result","Method","Class"]],
                                 hide_index=True, use_container_width=True)
