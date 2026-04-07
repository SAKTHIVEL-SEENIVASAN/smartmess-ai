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

/* 🌈 Background */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* 📦 Main container (stable selector) */
.block-container {
    background: rgba(255, 255, 255, 0.12);
    padding: 2rem;
    border-radius: 20px;
}

/* 🧠 Headings (REMOVE gradient text → use solid white) */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* 🏷 Labels */
label {
    color: #ffffff !important;
    font-weight: 500;
}

/* ✍ Inputs (IMPORTANT FIX) */
.stTextInput input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 12px;
    border: none;
    padding: 10px;
}

/* 👁 Password input */
.stTextInput input[type="password"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* 🔘 Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4ecb7a, #1f9b4a);
    color: white;
    border: none;
    border-radius: 50px;
    padding: 12px;
    font-weight: bold;
    width: 100%;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);
}

/* 📑 Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.15);
    border-radius: 50px;
    padding: 5px;
}

.stTabs [data-baseweb="tab"] {
    color: white !important;
    font-weight: bold;
}

/* 🚨 Alerts */
.stAlert {
    border-radius: 12px;
}

/* 📊 Metrics */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.15);
    border-radius: 15px;
    padding: 15px;
    color: white;
}

/* 📋 Fix placeholder visibility */
input::placeholder {
    color: #666 !important;
}

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
    """Detect users who vote but rarely attend"""
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
    """Cached deep learning model training"""
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
        SELECT a.date,
               a.actual_count,
               COUNT(v.id) as total_votes
        FROM actual_counts a
        LEFT JOIN votes v
            ON a.pg_code=v.pg_code
            AND a.date=v.date
            AND a.meal_type=v.meal_type
            AND v.vote=1
        WHERE a.pg_code=? AND a.meal_type=?
        GROUP BY a.date
        ORDER BY a.date
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
    
    # Show model status ONLY once (clean UI)
    if meal_type == "lunch":
        if dl_pred is not None:
            st.success("🤖 Deep Learning Active")
        else:
            st.info("📊 Using Basic ML (Need 7+ days data)")
    
    # ✅ FIXED CONDITION
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
        SELECT v.user_id, v.vote, u.food_pref
        FROM votes v
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
        confidence = 55.0
        veg_yes = sum(1 for v in yes_votes if v["food_pref"] == "veg")
        ratio = veg_yes / max(total_yes, 1)
        return {
            "predicted": round(predicted),
            "veg": round(predicted * ratio),
            "nonveg": round(predicted * (1 - ratio)),
            "confidence": confidence,
            "walkin": 0,
        }

    X = []
    y = []
    for i, row in enumerate(past):
        X.append([i])
        y.append(row["actual_count"])

    model = LinearRegression()
    model.fit(X, y)
    next_day = np.array([[len(past)]])
    ml_prediction = model.predict(next_day)[0]

    user_ids = [v["user_id"] for v in yes_votes]
    reliability_scores = get_reliability_scores(pg_code, meal_type, user_ids)

    weighted_yes = 0
    veg_weighted = 0
    nonveg_weighted = 0

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
        AND date IN (
            SELECT date FROM actual_counts
            WHERE pg_code=? AND meal_type=?
            ORDER BY date DESC LIMIT 14
        )
    """, (pg_code, meal_type, pg_code, meal_type)).fetchone()
    conn2.close()

    avg_votes = past_votes_row["c"] / max(len(past), 1)
    walkin_avg = max(0, avg_actual - avg_votes)

    predicted = (weighted_yes * 0.6) + (ml_prediction * 0.4) + walkin_avg
    participation = total_yes / max(total_students, 1)
    confidence = min(95, round(50 + (participation * 45), 1))
    veg_ratio = veg_weighted / max(weighted_yes, 1)

    return {
        "predicted": round(predicted),
        "veg": round(predicted * veg_ratio),
        "nonveg": round(predicted * (1 - veg_ratio)),
        "confidence": confidence,
        "walkin": round(walkin_avg, 1),
    }


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
            ON CONFLICT(user_id, date, meal_type)
            DO UPDATE SET attended=1
        """, (voter["user_id"], pg_code, today, meal_type))

    conn.commit()
    conn.close()


# ─── LANDING PAGE ─────────────────────────────────────────

def page_landing():
    st.title("🍛 Smart Mess System")
    st.markdown("### AI-powered food management for PGs")
    st.markdown("---")
    
    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container():
            st.markdown("### 👨‍💼 PG Owner")
            tab_a1, tab_a2 = st.tabs(["🔐 Login", "📝 Signup"])

            with tab_a1:
                email = normalize_email(st.text_input("Email", key="al_email", placeholder="owner@example.com"))
                password = st.text_input("Password", type="password", key="al_pass", placeholder="••••••••")
                if st.button("Login as Owner", key="al_btn", use_container_width=True):
                    conn = get_connection()
                    pg = conn.execute(
                        "SELECT * FROM pgs WHERE owner_email=?", (email,)
                    ).fetchone()
                    conn.close()
                    if pg and check_password(password, pg["password_hash"]):
                        st.session_state.user = dict(pg)
                        st.session_state.role = "admin"
                        st.rerun()
                    else:
                        st.error("❌ Wrong email or password")

            with tab_a2:
                name = st.text_input("PG Name", key="as_name", placeholder="My PG House")
                email2 = normalize_email(st.text_input("Email", key="as_email", placeholder="owner@example.com"))
                password2 = st.text_input("Password", type="password", key="as_pass", placeholder="•••••••• (min 6 chars)")

                if st.button("Create PG Account", key="as_btn", use_container_width=True):
                    if not name or not email2 or not password2:
                        st.warning("⚠️ Fill all fields")
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
                            st.success(f"✅ Done! Your PG Code: **{pg_code}**")
                            st.info("📢 Share this code with your students to join")
                        except Exception:
                            st.error("❌ Email already registered")
                        finally:
                            conn.close()

    with col2:
        with st.container():
            st.markdown("### 🎓 Student")
            tab_s1, tab_s2 = st.tabs(["🔐 Login", "📝 Join PG"])

            with tab_s1:
                s_email_login = normalize_email(st.text_input("Email", key="sl_email_login", placeholder="student@example.com"))
                s_pass = st.text_input("Password", type="password", key="sl_pass", placeholder="••••••••")
                if st.button("Login as Student", key="sl_btn", use_container_width=True):
                    if not s_email_login or not s_pass:
                        st.warning("⚠️ Please fill all fields")
                    else:
                        conn = get_connection()
                        user = conn.execute(
                            "SELECT * FROM users WHERE email=?", (s_email_login,)
                        ).fetchone()
                        conn.close()
                        if user and check_password(s_pass, user["password_hash"]):
                            st.session_state.user = dict(user)
                            st.session_state.role = "student"
                            st.rerun()
                        else:
                            st.error("❌ Wrong email or password")

            with tab_s2:
                s_name = st.text_input("Your Name", key="ss_name", placeholder="John Doe")
                s_email_signup = normalize_email(st.text_input("Email", key="ss_email", placeholder="student@example.com"))
                s_pass2 = st.text_input("Password", type="password", key="ss_pass", placeholder="•••••••• (min 6 chars)")
                s_code = st.text_input("PG Code", key="ss_code", placeholder="Enter code from owner")
                s_pref = st.selectbox("Food Preference", ["veg", "non-veg"], key="ss_pref")
                
                if st.button("Join PG", key="ss_btn", use_container_width=True):
                    # Validation
                    if not s_name or not s_email_signup or not s_pass2 or not s_code:
                        st.warning("⚠️ Fill all fields")
                    elif len(s_pass2) < 6:
                        st.error("❌ Password must be at least 6 characters")
                    elif not re.match(r"[^@]+@[^@]+\.[^@]+", s_email_signup):
                        st.error("❌ Invalid email format")
                    else:
                        conn = get_connection()
                        pg = conn.execute(
                            "SELECT * FROM pgs WHERE pg_code=?", (s_code,)
                        ).fetchone()
                        
                        if not pg:
                            st.error("❌ Invalid PG code")
                        else:
                            try:
                                conn.execute(
                                    "INSERT INTO users (name, email, password_hash, pg_code, food_pref) VALUES (?,?,?,?,?)",
                                    (s_name, s_email_signup, hash_password(s_pass2), s_code, s_pref),
                                )
                                conn.commit()
                                st.success("✅ Successfully joined! Now login as student")
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

    st.title(f"🍛 {u['name']}")
    st.caption(f"📱 PG Code: **`{u['pg_code']}`** — Share this with students to join")

    conn = get_connection()
    total_students = conn.execute(
        "SELECT COUNT(*) as c FROM users WHERE pg_code=?", (u['pg_code'],)
    ).fetchone()["c"]
    today_menu = conn.execute(
        "SELECT * FROM menus WHERE pg_code=? AND date=?", (u['pg_code'], today)
    ).fetchone()
    conn.close()

    # Animated metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Total Students", total_students, delta=None)
    with col2:
        st.metric("📅 Today's Date", today, delta=None)
    with col3:
        st.metric("📋 Menu Status", "✅ Uploaded" if today_menu else "❌ Not uploaded", 
                 delta="Action needed" if not today_menu else None)

    st.divider()
    
    # Beautiful tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Menu", "🗳️ Votes & Prediction", "📊 Analytics", "✅ Actual Count", "🔍 User Insights"])

    with tab1:
        st.subheader("📝 Upload Today's Menu")
        col_a, col_b = st.columns(2)
        with col_a:
            breakfast = st.text_area("🥐 Breakfast", value=today_menu["breakfast"] if today_menu else "", key="m_breakfast", placeholder="Eg: Idli, Sambhar, Chutney")
            lunch = st.text_area("🍱 Lunch", value=today_menu["lunch"] if today_menu else "", key="m_lunch", placeholder="Eg: Rice, Dal, Veg Curry, Roti")
        with col_b:
            dinner = st.text_area("🍛 Dinner", value=today_menu["dinner"] if today_menu else "", key="m_dinner", placeholder="Eg: Chapati, Paneer, Rice")
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

            with st.expander(f"**{meal.capitalize()}** — {total_yes} votes received", expanded=(meal=="lunch")):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("✅ Yes Votes", total_yes)
                with col2:
                    st.metric("🥬 Veg", yes_veg)
                with col3:
                    st.metric("🍗 Non-veg", yes_nonveg)
                with col4:
                    st.metric("🤖 AI Prediction", pred["predicted"])
                with col5:
                    st.metric("📊 Confidence", f"{pred['confidence']}%")
                
                if pred["walkin"] > 0:
                    st.info(f"🚶 Includes ~{pred['walkin']} expected walk-ins")

                # Progress bar for confidence
                st.progress(pred['confidence']/100, text=f"Prediction Confidence: {pred['confidence']}%")

                conn.execute("""
                    INSERT INTO predictions (pg_code, date, meal_type, predicted_count, veg_count, nonveg_count, confidence)
                    VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(pg_code, date, meal_type)
                    DO UPDATE SET predicted_count=excluded.predicted_count,
                                  veg_count=excluded.veg_count,
                                  nonveg_count=excluded.nonveg_count,
                                  confidence=excluded.confidence
                """, (u['pg_code'], today, meal, pred["predicted"], pred["veg"], pred["nonveg"], pred["confidence"]))
        conn.commit()
        conn.close()

    with tab3:
        st.subheader("📊 Food Analytics Dashboard")
        conn = get_connection()
        records = conn.execute("""
            SELECT a.date, a.meal_type, a.actual_count, p.predicted_count, p.confidence
            FROM actual_counts a
            LEFT JOIN predictions p
                ON a.pg_code=p.pg_code AND a.date=p.date AND a.meal_type=p.meal_type
            WHERE a.pg_code=?
            ORDER BY a.date DESC LIMIT 30
        """, (u['pg_code'],)).fetchall()
        conn.close()

        if records:
            df = pd.DataFrame([dict(r) for r in records])
            df["error"] = abs(df["actual_count"] - df["predicted_count"].fillna(0))
            df["accuracy_%"] = (
                100 - (df["error"] / df["actual_count"].replace(0, 1) * 100)
            ).round(1)
            df["waste"] = df["predicted_count"].fillna(0) - df["actual_count"]
            df["waste_%"] = (
                (df["waste"] / df["predicted_count"].replace(0, 1)) * 100
            ).round(1)

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Avg Accuracy", f"{df['accuracy_%'].mean():.1f}%", delta="↑" if df['accuracy_%'].mean() > 70 else "↓")
            with col2:
                st.metric("⚠️ Days Overpredicted", len(df[df["predicted_count"] > df["actual_count"]]))
            with col3:
                st.metric("📆 Days Tracked", len(df))
            with col4:
                st.metric("🗑️ Avg Waste %", f"{df['waste_%'].mean():.1f}%")

            st.subheader("📈 Actual vs Predicted Trend")
            st.line_chart(df.set_index("date")[["actual_count", "predicted_count"]])

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📉 Waste Trend (%)")
                st.line_chart(df.set_index("date")[["waste_%"]])
            with col2:
                st.subheader("📊 Prediction Error")
                st.line_chart(df.set_index("date")[["error"]])

            st.subheader("📋 Detailed Records")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("💡 No data yet. Enter actual counts after each meal to see analytics.")

    with tab4:
        st.subheader("✅ Enter Actual Count After Meal")
        st.warning("⚠️ Critical — AI learns from this every day! This improves future predictions.")
        meal_sel = st.selectbox("Select Meal", ["breakfast", "lunch", "dinner"], key="ac_meal")
        actual_n = st.number_input("How many people actually ate?", min_value=0, step=1, key="ac_num")

        if st.button("💾 Save Actual Count", key="ac_save", use_container_width=True):
            conn = get_connection()
            conn.execute("""
                INSERT INTO actual_counts (pg_code, date, meal_type, actual_count)
                VALUES (?,?,?,?)
                ON CONFLICT(pg_code, date, meal_type)
                DO UPDATE SET actual_count=excluded.actual_count
            """, (u['pg_code'], today, meal_sel, actual_n))
            conn.commit()
            conn.close()
            update_attendance(u['pg_code'], today, meal_sel, actual_n)
            st.success("✅ Saved! Attendance log updated. AI will improve from tomorrow.")
            st.balloons()

    with tab5:
        st.subheader("🔍 User Reliability Insights")
        
        # Detect fake/inconsistent users
        df_users = detect_fake_users(u['pg_code'])
        
        if not df_users.empty:
            st.markdown("### 👥 User Behavior Analysis")
            
            # Color code the dataframe
            def color_status(val):
                if val == '✅ Reliable':
                    return 'background-color: #90EE90'
                elif val == '⚠️ Inconsistent':
                    return 'background-color: #FFD700'
                elif val == '⚠️ Fake':
                    return 'background-color: #FFB6C1'
                return ''
            
            styled_df = df_users.style.applymap(color_status, subset=['status'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Bar chart for reliability
            st.subheader("📊 User Reliability Scores")
            st.bar_chart(df_users.set_index("name")["reliability"])
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                reliable = len(df_users[df_users['reliability'] > 80])
                st.metric("✅ Reliable Users", reliable)
            with col2:
                inconsistent = len(df_users[(df_users['reliability'] >= 50) & (df_users['reliability'] <= 80)])
                st.metric("⚠️ Inconsistent Users", inconsistent)
            with col3:
                fake = len(df_users[df_users['reliability'] < 50])
                st.metric("🚨 Potential Fake Users", fake, delta="Needs attention" if fake > 0 else None)
        else:
            st.info("💡 No user data available yet. Start collecting votes and attendance to see insights.")

    st.divider()
    if st.button("🚪 Logout", key="admin_logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.role = None
        st.rerun()


# ─── STUDENT DASHBOARD ────────────────────────────────────

def page_student():
    u = st.session_state.user
    today = str(date.today())

    st.title(f"👋 Hey {u['name']}!")
    st.caption(f"🏠 PG: **{u['pg_code']}** | 🍽️ Preference: **{u['food_pref'].capitalize()}**")

    conn = get_connection()
    menu = conn.execute(
        "SELECT * FROM menus WHERE pg_code=? AND date=?", (u['pg_code'], today)
    ).fetchone()

    if not menu:
        st.info("📋 No menu uploaded yet. Check back after your PG owner uploads it.")
        conn.close()
    else:
        deadline_time = parse_deadline(menu["voting_deadline"] or "10:00")
        voting_open = datetime.now().time() < deadline_time

        st.subheader(f"📅 Today's Menu — {today}")
        
        # Voting status card
        if voting_open:
            st.success(f"✅ Voting is OPEN! You can vote until {menu['voting_deadline']}")
            time_left = datetime.combine(datetime.today(), deadline_time) - datetime.now()
            if 0 < time_left.total_seconds() < 3600:
                st.warning(f"⏰ **Hurry!** Voting closes in **{int(time_left.total_seconds() // 60)} minutes**")
        else:
            st.error(f"❌ Voting CLOSED at {menu['voting_deadline']}")

        st.markdown("---")
        
        # Display menu with voting buttons
        for meal in ["breakfast", "lunch", "dinner"]:
            item = menu[meal]
            if not item:
                continue

            existing = conn.execute("""
                SELECT vote FROM votes
                WHERE user_id=? AND date=? AND meal_type=?
            """, (u['id'], today, meal)).fetchone()

            current_vote = existing["vote"] if existing else None
            
            with st.container():
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    if current_vote == 1:
                        st.markdown(f"**{meal.capitalize()}** 🍽️\n{item}\n\n✅ **You're coming!**")
                    elif current_vote == 0:
                        st.markdown(f"**{meal.capitalize()}** 🍽️\n{item}\n\n❌ **You're skipping**")
                    else:
                        st.markdown(f"**{meal.capitalize()}** 🍽️\n{item}")
                
                if voting_open:
                    with col2:
                        if st.button("✅ Coming", key=f"yes_{meal}", use_container_width=True):
                            save_vote(u['id'], u['pg_code'], today, meal, 1)
                            st.success("Voted! ✅")
                            st.rerun()
                    with col3:
                        if st.button("❌ Skip", key=f"no_{meal}", use_container_width=True):
                            save_vote(u['id'], u['pg_code'], today, meal, 0)
                            st.success("Voted! ❌")
                            st.rerun()
                else:
                    with col2:
                        st.button("✅ Coming", key=f"yes_{meal}_disabled", disabled=True, use_container_width=True)
                    with col3:
                        st.button("❌ Skip", key=f"no_{meal}_disabled", disabled=True, use_container_width=True)
                
                st.markdown("---")

        conn.close()

        st.divider()
        
        # Vote history
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
                st.info("📭 No vote history yet. Start voting to see your history!")

    st.divider()
    if st.button("🚪 Logout", key="student_logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.role = None
        st.rerun()


# ─── ROUTER ───────────────────────────────────────────────

if st.session_state.user is None:
    page_landing()
elif st.session_state.role == "admin":
    page_admin()
else:
    page_student()