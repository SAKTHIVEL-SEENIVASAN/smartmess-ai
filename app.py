import streamlit as st
from database import create_tables, get_connection
from utils import hash_password, check_password, generate_pg_code
from datetime import date, datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import re
from utils import normalize_email

# TensorFlow safety check
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

create_tables()
st.set_page_config(page_title="Smart Mess", page_icon="🍛", layout="wide")

st.markdown("""
<style>
/* ── IMPORT FONTS ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* ── CUSTOM CURSOR ────────────────────────────────────── */
*, *::before, *::after {
    cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='22' height='22' viewBox='0 0 22 22'%3E%3Ccircle cx='6' cy='6' r='5' fill='%234a7c59' fill-opacity='0.85' stroke='white' stroke-width='1.5'/%3E%3C/svg%3E") 6 6, auto !important;
}
button, a, [role='button'], .stButton > button {
    cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='22' height='22' viewBox='0 0 22 22'%3E%3Ccircle cx='6' cy='6' r='5.5' fill='%234a7c59' stroke='white' stroke-width='1.5'/%3E%3Cline x1='6' y1='2' x2='6' y2='10' stroke='white' stroke-width='1.5'/%3E%3Cline x1='2' y1='6' x2='10' y2='6' stroke='white' stroke-width='1.5'/%3E%3C/svg%3E") 6 6, pointer !important;
}

/* ── ROOT VARIABLES ───────────────────────────────────── */
:root {
    --sage:        #4a7c59;
    --sage-light:  #6a9e7a;
    --sage-pale:   #d4e8da;
    --sage-ghost:  #eef6f1;
    --white:       #ffffff;
    --off-white:   #f7faf8;
    --ink:         #1a2e22;
    --ink-mid:     #3d5247;
    --ink-soft:    #6b8577;
    --border:      #c8e0d0;
    --shadow:      0 4px 24px rgba(74,124,89,0.10);
    --shadow-lg:   0 8px 40px rgba(74,124,89,0.16);
    --radius:      16px;
    --radius-sm:   10px;
    --radius-pill: 999px;
}

/* ── GLOBAL RESET ─────────────────────────────────────── */
.stApp {
    background: var(--off-white) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Grain texture overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.025'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

.block-container {
    background: transparent !important;
    padding: 2.5rem 3rem !important;
    max-width: 1200px !important;
}

/* ── TYPOGRAPHY ───────────────────────────────────────── */
h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.8rem !important;
    color: var(--ink) !important;
    letter-spacing: -0.02em !important;
    line-height: 1.15 !important;
    margin-bottom: 0.25rem !important;
}
h2, h3 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--ink) !important;
    font-weight: 700 !important;
}
p, li, span, div {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--ink-mid) !important;
}
label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--ink-soft) !important;
}
.stCaption, small {
    color: var(--ink-soft) !important;
    font-size: 0.85rem !important;
}

/* ── INPUTS ───────────────────────────────────────────── */
.stTextInput input,
.stTextArea textarea,
.stSelectbox select,
.stNumberInput input {
    background: var(--white) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--sage) !important;
    box-shadow: 0 0 0 3px rgba(74,124,89,0.12) !important;
    outline: none !important;
}
input::placeholder, textarea::placeholder {
    color: #b0c4b8 !important;
}

/* ── BUTTONS ──────────────────────────────────────────── */
.stButton > button {
    background: var(--sage) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: var(--radius-pill) !important;
    padding: 0.65rem 1.5rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
    transition: background 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 2px 12px rgba(74,124,89,0.20) !important;
}
.stButton > button:hover {
    background: var(--sage-light) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(74,124,89,0.28) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
.stButton > button:disabled {
    background: var(--sage-pale) !important;
    color: var(--ink-soft) !important;
    box-shadow: none !important;
}

/* ── TABS ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    background: var(--sage-ghost) !important;
    border-radius: var(--radius-pill) !important;
    padding: 5px !important;
    border: 1.5px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--ink-soft) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border-radius: var(--radius-pill) !important;
    padding: 0.4rem 1.1rem !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: var(--sage) !important;
    color: var(--white) !important;
}

/* ── CARDS / CONTAINERS ───────────────────────────────── */
div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
    gap: 1.5rem !important;
}

/* Section card wrapper */
.mess-card {
    background: var(--white);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 2rem 1.5rem;
    box-shadow: var(--shadow);
    transition: box-shadow 0.2s;
}
.mess-card:hover {
    box-shadow: var(--shadow-lg);
}

/* ── METRICS ──────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: var(--white) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem 1.4rem !important;
    box-shadow: var(--shadow) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: var(--shadow-lg) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: var(--ink-soft) !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--ink) !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.9rem !important;
}

/* ── ALERTS ───────────────────────────────────────────── */
.stSuccess {
    background: var(--sage-ghost) !important;
    border: 1.5px solid var(--sage-pale) !important;
    border-left: 4px solid var(--sage) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink) !important;
}
.stInfo {
    background: #f0f7ff !important;
    border: 1.5px solid #cce0f5 !important;
    border-left: 4px solid #5b9bd5 !important;
    border-radius: var(--radius-sm) !important;
}
.stWarning {
    background: #fffbf0 !important;
    border: 1.5px solid #fde68a !important;
    border-left: 4px solid #f59e0b !important;
    border-radius: var(--radius-sm) !important;
}
.stError {
    background: #fff5f5 !important;
    border: 1.5px solid #fecaca !important;
    border-left: 4px solid #ef4444 !important;
    border-radius: var(--radius-sm) !important;
}

/* ── EXPANDERS ────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--sage-ghost) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
}
.streamlit-expanderContent {
    background: var(--white) !important;
    border: 1.5px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
}

/* ── PROGRESS ─────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--sage), var(--sage-light)) !important;
    border-radius: var(--radius-pill) !important;
}
.stProgress > div > div {
    background: var(--sage-pale) !important;
    border-radius: var(--radius-pill) !important;
}

/* ── DATAFRAME ────────────────────────────────────────── */
.stDataFrame {
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}
.stDataFrame thead th {
    background: var(--sage-ghost) !important;
    color: var(--ink) !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── DIVIDER ──────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1.5px solid var(--border) !important;
    margin: 1.8rem 0 !important;
}

/* ── SIDEBAR (if ever used) ───────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1.5px solid var(--border) !important;
}

/* ── SELECTBOX ────────────────────────────────────────── */
.stSelectbox > div > div {
    background: var(--white) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink) !important;
}

/* ── SCROLL BAR ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--sage-ghost); }
::-webkit-scrollbar-thumb { background: var(--sage-pale); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--sage); }

/* ── LANDING HERO STYLES ──────────────────────────────── */
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--sage-ghost);
    border: 1.5px solid var(--border);
    border-radius: var(--radius-pill);
    padding: 5px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--sage);
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.4rem;
    color: var(--ink);
    line-height: 1.1;
    letter-spacing: -0.03em;
    margin-bottom: 0.6rem;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: var(--ink-soft);
    margin-bottom: 2rem;
    line-height: 1.6;
}
.panel-card {
    background: var(--white);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    box-shadow: var(--shadow);
    height: 100%;
}
.panel-heading {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--sage);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.feature-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--sage-ghost);
    border: 1px solid var(--border);
    border-radius: var(--radius-pill);
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--sage);
    margin: 3px 3px 3px 0;
}

/* ── VOTE MEAL CARD ───────────────────────────────────── */
.meal-row {
    background: var(--white);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    transition: box-shadow 0.2s, border-color 0.2s;
}
.meal-row:hover {
    box-shadow: var(--shadow);
    border-color: var(--sage-pale);
}
.meal-voted-yes {
    border-left: 4px solid var(--sage) !important;
    background: var(--sage-ghost) !important;
}
.meal-voted-no {
    border-left: 4px solid #f87171 !important;
    background: #fff5f5 !important;
}

/* ── TIME INPUT ───────────────────────────────────────── */
.stTimeInput input {
    background: var(--white) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink) !important;
}

/* ── PAGE FADE IN ─────────────────────────────────────── */
.block-container {
    animation: fadeUp 0.45s ease both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── LINE CHARTS ──────────────────────────────────────── */
.stVegaLiteChart { border-radius: var(--radius) !important; overflow: hidden !important; }
</style>
""", unsafe_allow_html=True)

for key in ["user", "role"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ─── HELPERS ──────────────────────────────────────────────

def parse_deadline(deadline_str):
    for fmt in ["%H:%M:%S", "%H:%M"]:
        try:
            return datetime.strptime(deadline_str, fmt).time()
        except Exception:
            continue
    return datetime.strptime("10:00", "%H:%M").time()


def get_reliability_scores(pg_code, meal_type, user_ids):
    if not user_ids:
        return {}
    placeholders = ",".join("?" * len(user_ids))
    conn = get_connection()
    rows = conn.execute(f"""
        SELECT v.user_id,
               COUNT(*) as total_yes,
               SUM(COALESCE(a.attended, 0)) as total_attended
        FROM votes v
        LEFT JOIN attendance_log a
            ON v.user_id = a.user_id
            AND v.date = a.date
            AND v.meal_type = a.meal_type
        WHERE v.user_id IN ({placeholders})
          AND v.pg_code=?
          AND v.meal_type=?
          AND v.vote=1
        GROUP BY v.user_id
    """, (*user_ids, pg_code, meal_type)).fetchall()
    conn.close()
    scores = {}
    for r in rows:
        if r["total_yes"] < 5:
            scores[r["user_id"]] = 0.85
        else:
            scores[r["user_id"]] = round(r["total_attended"] / r["total_yes"], 2)
    return scores


def detect_fake_users(pg_code):
    conn = get_connection()
    users = conn.execute("""
        SELECT u.id, u.name, u.food_pref,
               COUNT(v.id) as total_votes,
               SUM(COALESCE(a.attended, 0)) as total_attended
        FROM users u
        LEFT JOIN votes v ON u.id = v.user_id
        LEFT JOIN attendance_log a ON u.id = a.user_id AND v.date = a.date AND v.meal_type = a.meal_type
        WHERE u.pg_code = ?
        GROUP BY u.id
        HAVING total_votes > 0
    """, (pg_code,)).fetchall()
    conn.close()
    if not users:
        return pd.DataFrame()
    data = []
    for u in users:
        reliability = (u['total_attended'] / u['total_votes'] * 100) if u['total_votes'] > 0 else 0
        status = "⚠️ Fake" if reliability < 50 else "✅ Reliable" if reliability > 80 else "⚠️ Inconsistent"
        data.append({
            'name': u['name'],
            'food_pref': u['food_pref'],
            'total_votes': u['total_votes'],
            'attended': u['total_attended'],
            'reliability': round(reliability, 1),
            'status': status
        })
    return pd.DataFrame(data)


@st.cache_data(ttl=3600)
def train_dl_model(X, y):
    if not TF_AVAILABLE:
        return None
    model = Sequential([
        Dense(16, activation='relu', input_shape=(2,)),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, verbose=0)
    return model


def deep_learning_predict(pg_code, today, meal_type):
    if not TF_AVAILABLE:
        return None
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT a.date, a.actual_count, COUNT(v.id) as total_votes
        FROM actual_counts a
        LEFT JOIN votes v ON a.pg_code=v.pg_code AND a.date=v.date
            AND a.meal_type=v.meal_type AND v.vote=1
        WHERE a.pg_code=? AND a.meal_type=?
        GROUP BY a.date ORDER BY a.date
    """, conn, params=(pg_code, meal_type))
    today_votes = conn.execute("""
        SELECT COUNT(*) as c FROM votes
        WHERE pg_code=? AND date=? AND meal_type=? AND vote=1
    """, (pg_code, today, meal_type)).fetchone()["c"]
    conn.close()
    if len(df) < 7:
        return None
    df["day"] = pd.to_datetime(df["date"]).dt.dayofweek
    X = df[["total_votes", "day"]].values
    y = df["actual_count"].values
    model = train_dl_model(X, y)
    if model is None:
        return None
    today_day = pd.to_datetime(today).dayofweek
    pred = model.predict(np.array([[today_votes, today_day]]), verbose=0)[0][0]
    return max(0, int(pred))


def smart_predict(pg_code, today, meal_type):
    dl_pred = deep_learning_predict(pg_code, today, meal_type)
    if meal_type == "lunch":
        if dl_pred is not None:
            st.success("🤖 Deep Learning Model Active")
        else:
            st.info("📊 Using Linear ML — need 7+ days of data")
    if dl_pred is not None:
        return {
            "predicted": dl_pred,
            "veg": int(dl_pred * 0.6),
            "nonveg": int(dl_pred * 0.4),
            "confidence": 90,
            "walkin": max(0, dl_pred)
        }
    conn = get_connection()
    votes = conn.execute("""
        SELECT v.user_id, v.vote, u.food_pref FROM votes v
        JOIN users u ON v.user_id = u.id
        WHERE v.pg_code=? AND v.date=? AND v.meal_type=?
    """, (pg_code, today, meal_type)).fetchall()
    past = conn.execute("""
        SELECT actual_count FROM actual_counts
        WHERE pg_code=? AND meal_type=?
        ORDER BY date DESC LIMIT 14
    """, (pg_code, meal_type)).fetchall()
    total_students = conn.execute(
        "SELECT COUNT(*) as c FROM users WHERE pg_code=?", (pg_code,)
    ).fetchone()["c"]
    conn.close()
    yes_votes = [v for v in votes if v["vote"] == 1]
    total_yes = len(yes_votes)
    if len(past) < 7:
        baseline = total_students * 0.65
        predicted = (total_yes * 0.7) + (baseline * 0.3)
        veg_yes = sum(1 for v in yes_votes if v["food_pref"] == "veg")
        ratio = veg_yes / max(total_yes, 1)
        return {"predicted": round(predicted), "veg": round(predicted * ratio),
                "nonveg": round(predicted * (1 - ratio)), "confidence": 55.0, "walkin": 0}
    X_vals, y_vals = [], []
    for i, row in enumerate(past):
        X_vals.append([i])
        y_vals.append(row["actual_count"])
    model = LinearRegression()
    model.fit(X_vals, y_vals)
    ml_prediction = model.predict([[len(past)]])[0]
    user_ids = [v["user_id"] for v in yes_votes]
    reliability_scores = get_reliability_scores(pg_code, meal_type, user_ids)
    weighted_yes, veg_weighted, nonveg_weighted = 0, 0, 0
    for v in yes_votes:
        rel = reliability_scores.get(v["user_id"], 0.85)
        weighted_yes += rel
        if v["food_pref"] == "veg":
            veg_weighted += rel
        else:
            nonveg_weighted += rel
    avg_actual = sum(r["actual_count"] for r in past) / len(past)
    conn2 = get_connection()
    past_votes_row = conn2.execute("""
        SELECT COUNT(*) as c FROM votes
        WHERE pg_code=? AND meal_type=? AND vote=1
        AND date IN (SELECT date FROM actual_counts WHERE pg_code=? AND meal_type=? ORDER BY date DESC LIMIT 14)
    """, (pg_code, meal_type, pg_code, meal_type)).fetchone()
    conn2.close()
    avg_votes = past_votes_row["c"] / max(len(past), 1)
    walkin_avg = max(0, avg_actual - avg_votes)
    predicted = (weighted_yes * 0.6) + (ml_prediction * 0.4) + walkin_avg
    participation = total_yes / max(total_students, 1)
    confidence = min(95, round(50 + (participation * 45), 1))
    veg_ratio = veg_weighted / max(weighted_yes, 1)
    return {"predicted": round(predicted), "veg": round(predicted * veg_ratio),
            "nonveg": round(predicted * (1 - veg_ratio)), "confidence": confidence, "walkin": round(walkin_avg, 1)}


def save_vote(user_id, pg_code, today, meal, vote_val):
    conn = get_connection()
    conn.execute("""
        INSERT INTO votes (user_id, pg_code, date, meal_type, vote)
        VALUES (?,?,?,?,?)
        ON CONFLICT(user_id, date, meal_type)
        DO UPDATE SET vote=excluded.vote, timestamp=CURRENT_TIMESTAMP
    """, (user_id, pg_code, today, meal, vote_val))
    conn.commit()
    conn.close()


def update_attendance(pg_code, today, meal_type, actual_count):
    conn = get_connection()
    yes_voters = conn.execute("""
        SELECT user_id FROM votes
        WHERE pg_code=? AND date=? AND meal_type=? AND vote=1
        ORDER BY timestamp ASC LIMIT ?
    """, (pg_code, today, meal_type, actual_count)).fetchall()
    for voter in yes_voters:
        conn.execute("""
            INSERT INTO attendance_log (user_id, pg_code, date, meal_type, attended)
            VALUES (?,?,?,?,1)
            ON CONFLICT(user_id, date, meal_type) DO UPDATE SET attended=1
        """, (voter["user_id"], pg_code, today, meal_type))
    conn.commit()
    conn.close()


# ─── LANDING PAGE ─────────────────────────────────────────

def page_landing():
    # Hero section
    col_hero, col_spacer = st.columns([3, 1])
    with col_hero:
        st.markdown('<div class="hero-badge">🍛 AI-Powered Mess Management</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title">Smart Mess<br><em style="color:#4a7c59;">System</em></h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle">Predict food demand. Reduce waste. Keep everyone fed — intelligently.</p>', unsafe_allow_html=True)
        
        # Feature pills
        for pill in ["🤖 Deep Learning", "📊 Real-time Analytics", "✅ Vote Tracking", "🚨 Waste Reduction"]:
            st.markdown(f'<span class="feature-pill">{pill}</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="panel-heading">👨‍💼 PG Owner Portal</div>', unsafe_allow_html=True)
        tab_a1, tab_a2 = st.tabs(["🔐 Login", "📝 Register"])

        with tab_a1:
            email = normalize_email(st.text_input("Email Address", key="al_email", placeholder="owner@example.com"))
            password = st.text_input("Password", type="password", key="al_pass", placeholder="Enter your password")
            if st.button("Login as Owner →", key="al_btn", use_container_width=True):
                conn = get_connection()
                pg = conn.execute("SELECT * FROM pgs WHERE owner_email=?", (email,)).fetchone()
                conn.close()
                if pg and check_password(password, pg["password_hash"]):
                    st.session_state.user = dict(pg)
                    st.session_state.role = "admin"
                    st.rerun()
                else:
                    st.error("❌ Wrong email or password")

        with tab_a2:
            name = st.text_input("PG Name", key="as_name", placeholder="My PG House")
            email2 = normalize_email(st.text_input("Email Address", key="as_email", placeholder="owner@example.com"))
            password2 = st.text_input("Password", type="password", key="as_pass", placeholder="Min 6 characters")
            if st.button("Create PG Account →", key="as_btn", use_container_width=True):
                if not name or not email2 or not password2:
                    st.warning("⚠️ Please fill all fields")
                elif len(password2) < 6:
                    st.error("❌ Password must be at least 6 characters")
                elif not re.match(r"[^@]+@[^@]+\.[^@]+", email2):
                    st.error("❌ Invalid email format")
                else:
                    conn = get_connection()
                    pg_code = generate_pg_code(conn)
                    try:
                        conn.execute(
                            "INSERT INTO pgs (name, owner_email, password_hash, pg_code) VALUES (?,?,?,?)",
                            (name, email2, hash_password(password2), pg_code),
                        )
                        conn.commit()
                        st.success(f"✅ Account created! Your PG Code: **{pg_code}**")
                        st.info("📢 Share this code with your students to let them join.")
                    except Exception:
                        st.error("❌ Email already registered")
                    finally:
                        conn.close()

    with col2:
        st.markdown('<div class="panel-heading">🎓 Student Portal</div>', unsafe_allow_html=True)
        tab_s1, tab_s2 = st.tabs(["🔐 Login", "📝 Join PG"])

        with tab_s1:
            s_email_login = normalize_email(st.text_input("Email Address", key="sl_email_login", placeholder="student@example.com"))
            s_pass = st.text_input("Password", type="password", key="sl_pass", placeholder="Enter your password")
            if st.button("Login as Student →", key="sl_btn", use_container_width=True):
                if not s_email_login or not s_pass:
                    st.warning("⚠️ Please fill all fields")
                else:
                    conn = get_connection()
                    user = conn.execute("SELECT * FROM users WHERE email=?", (s_email_login,)).fetchone()
                    conn.close()
                    if user and check_password(s_pass, user["password_hash"]):
                        st.session_state.user = dict(user)
                        st.session_state.role = "student"
                        st.rerun()
                    else:
                        st.error("❌ Wrong email or password")

        with tab_s2:
            s_name = st.text_input("Full Name", key="ss_name", placeholder="John Doe")
            s_email_signup = normalize_email(st.text_input("Email Address", key="ss_email", placeholder="student@example.com"))
            s_pass2 = st.text_input("Password", type="password", key="ss_pass", placeholder="Min 6 characters")
            s_code = st.text_input("PG Code", key="ss_code", placeholder="Get this from your PG owner")
            s_pref = st.selectbox("Food Preference", ["veg", "non-veg"], key="ss_pref")
            if st.button("Join PG →", key="ss_btn", use_container_width=True):
                if not s_name or not s_email_signup or not s_pass2 or not s_code:
                    st.warning("⚠️ Please fill all fields")
                elif len(s_pass2) < 6:
                    st.error("❌ Password must be at least 6 characters")
                elif not re.match(r"[^@]+@[^@]+\.[^@]+", s_email_signup):
                    st.error("❌ Invalid email format")
                else:
                    conn = get_connection()
                    pg = conn.execute("SELECT * FROM pgs WHERE pg_code=?", (s_code,)).fetchone()
                    if not pg:
                        st.error("❌ Invalid PG code")
                    else:
                        try:
                            conn.execute(
                                "INSERT INTO users (name, email, password_hash, pg_code, food_pref) VALUES (?,?,?,?,?)",
                                (s_name, s_email_signup, hash_password(s_pass2), s_code, s_pref),
                            )
                            conn.commit()
                            st.success("✅ Joined successfully! Now login as a student.")
                            st.balloons()
                        except Exception as e:
                            if "UNIQUE constraint failed" in str(e):
                                st.error("❌ Email already registered")
                            else:
                                st.error(f"❌ Error: {str(e)}")
                    conn.close()


# ─── ADMIN DASHBOARD ──────────────────────────────────────

def page_admin():
    u = st.session_state.user
    today = str(date.today())

    # Header
    col_title, col_logout = st.columns([5, 1])
    with col_title:
        st.markdown(f'<h1 style="margin-bottom:0">🍛 {u["name"]}</h1>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:var(--ink-soft);font-size:0.9rem;margin-top:4px">'
            f'PG Code: <code style="background:var(--sage-ghost);padding:2px 8px;border-radius:6px;color:var(--sage);font-weight:700">'
            f'{u["pg_code"]}</code> — Share with students to join</p>',
            unsafe_allow_html=True
        )
    with col_logout:
        if st.button("🚪 Logout", key="admin_logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.role = None
            st.rerun()

    st.divider()

    conn = get_connection()
    total_students = conn.execute("SELECT COUNT(*) as c FROM users WHERE pg_code=?", (u['pg_code'],)).fetchone()["c"]
    today_menu = conn.execute("SELECT * FROM menus WHERE pg_code=? AND date=?", (u['pg_code'], today)).fetchone()
    conn.close()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Total Students", total_students)
    with col2:
        st.metric("📅 Today", today)
    with col3:
        st.metric("📋 Menu", "✅ Uploaded" if today_menu else "❌ Missing",
                  delta="Action needed" if not today_menu else None)

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Menu", "🗳️ Votes & AI", "📊 Analytics", "✅ Actual Count", "🔍 Insights"])

    with tab1:
        st.subheader("📝 Upload Today's Menu")
        col_a, col_b = st.columns(2)
        with col_a:
            breakfast = st.text_area("🥐 Breakfast", value=today_menu["breakfast"] if today_menu else "",
                                      key="m_breakfast", placeholder="e.g. Idli, Sambhar, Chutney")
            lunch = st.text_area("🍱 Lunch", value=today_menu["lunch"] if today_menu else "",
                                  key="m_lunch", placeholder="e.g. Rice, Dal, Veg Curry, Roti")
        with col_b:
            dinner = st.text_area("🍛 Dinner", value=today_menu["dinner"] if today_menu else "",
                                   key="m_dinner", placeholder="e.g. Chapati, Paneer, Rice")
            deadline = st.time_input("⏰ Voting Deadline", value=datetime.strptime("10:00", "%H:%M").time(), key="m_deadline")

        if st.button("💾 Save Menu", key="save_menu", use_container_width=True):
            conn = get_connection()
            conn.execute("""
                INSERT INTO menus (pg_code, date, breakfast, lunch, dinner, voting_deadline)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(pg_code, date)
                DO UPDATE SET breakfast=excluded.breakfast, lunch=excluded.lunch,
                              dinner=excluded.dinner, voting_deadline=excluded.voting_deadline
            """, (u['pg_code'], today, breakfast, lunch, dinner, str(deadline)))
            conn.commit()
            conn.close()
            st.success("✅ Menu saved successfully!")
            st.balloons()
            st.rerun()

    with tab2:
        st.subheader(f"🗳️ Today's Votes — {today}")
        conn = get_connection()
        for meal in ["breakfast", "lunch", "dinner"]:
            votes = conn.execute("""
                SELECT v.vote, u.food_pref, v.user_id FROM votes v
                JOIN users u ON v.user_id = u.id
                WHERE v.pg_code=? AND v.date=? AND v.meal_type=?
            """, (u['pg_code'], today, meal)).fetchall()
            yes_veg = sum(1 for v in votes if v["vote"] == 1 and v["food_pref"] == "veg")
            yes_nonveg = sum(1 for v in votes if v["vote"] == 1 and v["food_pref"] == "non-veg")
            total_yes = yes_veg + yes_nonveg
            pred = smart_predict(u['pg_code'], today, meal)
            with st.expander(f"**{meal.capitalize()}** — {total_yes} votes received", expanded=(meal == "lunch")):
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1: st.metric("✅ Yes", total_yes)
                with c2: st.metric("🥬 Veg", yes_veg)
                with c3: st.metric("🍗 Non-veg", yes_nonveg)
                with c4: st.metric("🤖 Predicted", pred["predicted"])
                with c5: st.metric("📊 Confidence", f"{pred['confidence']}%")
                if pred["walkin"] > 0:
                    st.info(f"🚶 ~{pred['walkin']} expected walk-ins included")
                st.progress(pred['confidence'] / 100, text=f"Prediction Confidence: {pred['confidence']}%")
                conn.execute("""
                    INSERT INTO predictions (pg_code, date, meal_type, predicted_count, veg_count, nonveg_count, confidence)
                    VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(pg_code, date, meal_type)
                    DO UPDATE SET predicted_count=excluded.predicted_count,
                                  veg_count=excluded.veg_count, nonveg_count=excluded.nonveg_count,
                                  confidence=excluded.confidence
                """, (u['pg_code'], today, meal, pred["predicted"], pred["veg"], pred["nonveg"], pred["confidence"]))
        conn.commit()
        conn.close()

    with tab3:
        st.subheader("📊 Analytics Dashboard")
        conn = get_connection()
        records = conn.execute("""
            SELECT a.date, a.meal_type, a.actual_count, p.predicted_count, p.confidence
            FROM actual_counts a
            LEFT JOIN predictions p ON a.pg_code=p.pg_code AND a.date=p.date AND a.meal_type=p.meal_type
            WHERE a.pg_code=? ORDER BY a.date DESC LIMIT 30
        """, (u['pg_code'],)).fetchall()
        conn.close()
        if records:
            df = pd.DataFrame([dict(r) for r in records])
            df["error"] = abs(df["actual_count"] - df["predicted_count"].fillna(0))
            df["accuracy_%"] = (100 - (df["error"] / df["actual_count"].replace(0, 1) * 100)).round(1)
            df["waste"] = df["predicted_count"].fillna(0) - df["actual_count"]
            df["waste_%"] = ((df["waste"] / df["predicted_count"].replace(0, 1)) * 100).round(1)
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("🎯 Avg Accuracy", f"{df['accuracy_%'].mean():.1f}%")
            with c2: st.metric("⚠️ Over-predicted Days", len(df[df["predicted_count"] > df["actual_count"]]))
            with c3: st.metric("📆 Days Tracked", len(df))
            with c4: st.metric("🗑️ Avg Waste %", f"{df['waste_%'].mean():.1f}%")
            st.subheader("📈 Actual vs Predicted")
            st.line_chart(df.set_index("date")[["actual_count", "predicted_count"]])
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📉 Waste Trend")
                st.line_chart(df.set_index("date")[["waste_%"]])
            with col2:
                st.subheader("📊 Prediction Error")
                st.line_chart(df.set_index("date")[["error"]])
            st.subheader("📋 Full Record")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("💡 No analytics yet. Enter actual counts after meals to start tracking.")

    with tab4:
        st.subheader("✅ Enter Actual Count After Meal")
        st.warning("⚠️ This is how the AI learns — enter counts daily for better predictions!")
        meal_sel = st.selectbox("Meal", ["breakfast", "lunch", "dinner"], key="ac_meal")
        actual_n = st.number_input("Actual number who ate", min_value=0, step=1, key="ac_num")
        if st.button("💾 Save Count", key="ac_save", use_container_width=True):
            conn = get_connection()
            conn.execute("""
                INSERT INTO actual_counts (pg_code, date, meal_type, actual_count)
                VALUES (?,?,?,?)
                ON CONFLICT(pg_code, date, meal_type) DO UPDATE SET actual_count=excluded.actual_count
            """, (u['pg_code'], today, meal_sel, actual_n))
            conn.commit()
            conn.close()
            update_attendance(u['pg_code'], today, meal_sel, actual_n)
            st.success("✅ Saved! Attendance updated. AI will improve from tomorrow.")
            st.balloons()

    with tab5:
        st.subheader("🔍 User Reliability Insights")
        df_users = detect_fake_users(u['pg_code'])
        if not df_users.empty:
            def color_status(val):
                if val == '✅ Reliable': return 'background-color: #d4f7e0'
                elif val == '⚠️ Inconsistent': return 'background-color: #fef9c3'
                elif val == '⚠️ Fake': return 'background-color: #fee2e2'
                return ''
            st.dataframe(df_users.style.applymap(color_status, subset=['status']), use_container_width=True)
            st.subheader("📊 Reliability Scores")
            st.bar_chart(df_users.set_index("name")["reliability"])
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("✅ Reliable", len(df_users[df_users['reliability'] > 80]))
            with c2: st.metric("⚠️ Inconsistent", len(df_users[(df_users['reliability'] >= 50) & (df_users['reliability'] <= 80)]))
            with c3:
                fake_count = len(df_users[df_users['reliability'] < 50])
                st.metric("🚨 Potential Fake", fake_count, delta="Needs attention" if fake_count > 0 else None)
        else:
            st.info("💡 No data yet. Collect votes and attendance to see insights here.")


# ─── STUDENT DASHBOARD ────────────────────────────────────

def page_student():
    u = st.session_state.user
    today = str(date.today())

    col_title, col_logout = st.columns([5, 1])
    with col_title:
        st.markdown(f'<h1 style="margin-bottom:0">👋 Hey, {u["name"]}!</h1>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:var(--ink-soft);font-size:0.9rem;margin-top:4px">'
            f'🏠 PG: <strong>{u["pg_code"]}</strong> &nbsp;·&nbsp; 🍽️ Preference: <strong>{u["food_pref"].capitalize()}</strong></p>',
            unsafe_allow_html=True
        )
    with col_logout:
        if st.button("🚪 Logout", key="student_logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.role = None
            st.rerun()

    st.divider()

    conn = get_connection()
    menu = conn.execute("SELECT * FROM menus WHERE pg_code=? AND date=?", (u['pg_code'], today)).fetchone()

    if not menu:
        st.info("📋 No menu uploaded yet. Check back once your PG owner uploads today's menu.")
        conn.close()
        return

    deadline_time = parse_deadline(menu["voting_deadline"] or "10:00")
    voting_open = datetime.now().time() < deadline_time

    st.subheader(f"📅 Today's Menu — {today}")
    if voting_open:
        st.success(f"✅ Voting OPEN — closes at {menu['voting_deadline']}")
        time_left = datetime.combine(datetime.today(), deadline_time) - datetime.now()
        if 0 < time_left.total_seconds() < 3600:
            st.warning(f"⏰ Hurry! Closes in **{int(time_left.total_seconds() // 60)} minutes**")
    else:
        st.error(f"❌ Voting closed at {menu['voting_deadline']}")

    st.markdown("---")

    for meal in ["breakfast", "lunch", "dinner"]:
        item = menu[meal]
        if not item:
            continue
        existing = conn.execute("""
            SELECT vote FROM votes WHERE user_id=? AND date=? AND meal_type=?
        """, (u['id'], today, meal)).fetchone()
        current_vote = existing["vote"] if existing else None

        card_class = ""
        if current_vote == 1:
            card_class = "meal-voted-yes"
        elif current_vote == 0:
            card_class = "meal-voted-no"

        st.markdown(f'<div class="meal-row {card_class}">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            vote_label = " &nbsp;✅ <em>You're coming</em>" if current_vote == 1 else (" &nbsp;❌ <em>Skipping</em>" if current_vote == 0 else "")
            st.markdown(
                f'<p style="margin:0;font-weight:700;font-size:1rem;color:var(--ink)">'
                f'{meal.capitalize()}{vote_label}</p>'
                f'<p style="margin:4px 0 0;color:var(--ink-soft);font-size:0.9rem">{item}</p>',
                unsafe_allow_html=True
            )
        if voting_open:
            with col2:
                if st.button("✅ Coming", key=f"yes_{meal}", use_container_width=True):
                    save_vote(u['id'], u['pg_code'], today, meal, 1)
                    st.rerun()
            with col3:
                if st.button("❌ Skip", key=f"no_{meal}", use_container_width=True):
                    save_vote(u['id'], u['pg_code'], today, meal, 0)
                    st.rerun()
        else:
            with col2:
                st.button("✅ Coming", key=f"yes_{meal}_d", disabled=True, use_container_width=True)
            with col3:
                st.button("❌ Skip", key=f"no_{meal}_d", disabled=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    conn.close()
    st.divider()

    with st.expander("📜 My Vote History", expanded=False):
        conn2 = get_connection()
        history = conn2.execute("""
            SELECT date, meal_type, vote FROM votes
            WHERE user_id=? ORDER BY date DESC LIMIT 20
        """, (u['id'],)).fetchall()
        conn2.close()
        if history:
            df = pd.DataFrame([dict(h) for h in history])
            df["vote"] = df["vote"].map({1: "✅ Coming", 0: "❌ Skip"})
            st.dataframe(df, use_container_width=True)
        else:
            st.info("📭 No vote history yet. Start voting to track your history!")


# ─── ROUTER ───────────────────────────────────────────────

if st.session_state.user is None:
    page_landing()
elif st.session_state.role == "admin":
    page_admin()
else:
    page_student()