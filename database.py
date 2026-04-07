import sqlite3

def get_connection():
    conn = sqlite3.connect("smartmess.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables():
    conn = get_connection()
    c = conn.cursor()

    # ─── PG TABLE ─────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS pgs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            owner_email TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
            pg_code TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ─── USERS ────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
            pg_code TEXT NOT NULL,
            food_pref TEXT DEFAULT 'veg',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(pg_code) REFERENCES pgs(pg_code)
        )
    """)

    # ─── MENUS ────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS menus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pg_code TEXT NOT NULL,
            date TEXT NOT NULL,
            breakfast TEXT,
            lunch TEXT,
            dinner TEXT,
            voting_deadline TEXT,
            UNIQUE(pg_code, date),
            FOREIGN KEY(pg_code) REFERENCES pgs(pg_code)
        )
    """)

    # ─── VOTES ────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            pg_code TEXT NOT NULL,
            date TEXT NOT NULL,
            meal_type TEXT NOT NULL,
            vote INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, date, meal_type),
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(pg_code) REFERENCES pgs(pg_code)
        )
    """)

    # ─── ACTUAL COUNTS ───────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS actual_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pg_code TEXT NOT NULL,
            date TEXT NOT NULL,
            meal_type TEXT NOT NULL,
            actual_count INTEGER NOT NULL,
            UNIQUE(pg_code, date, meal_type),
            FOREIGN KEY(pg_code) REFERENCES pgs(pg_code)
        )
    """)

    # ─── PREDICTIONS ─────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pg_code TEXT NOT NULL,
            date TEXT NOT NULL,
            meal_type TEXT NOT NULL,
            predicted_count REAL,
            veg_count REAL,
            nonveg_count REAL,
            confidence REAL,
            UNIQUE(pg_code, date, meal_type),
            FOREIGN KEY(pg_code) REFERENCES pgs(pg_code)
        )
    """)

    # ─── ATTENDANCE ──────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            pg_code TEXT NOT NULL,
            date TEXT NOT NULL,
            meal_type TEXT NOT NULL,
            attended INTEGER DEFAULT 0,
            UNIQUE(user_id, date, meal_type),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # ─── INDEXES (🔥 SPEED BOOST)
    c.execute("CREATE INDEX IF NOT EXISTS idx_votes_pg_date ON votes(pg_code, date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_votes_user ON votes(user_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_users_pg ON users(pg_code)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_menus_pg_date ON menus(pg_code, date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_actual_pg_date ON actual_counts(pg_code, date)")

    conn.commit()
    conn.close()