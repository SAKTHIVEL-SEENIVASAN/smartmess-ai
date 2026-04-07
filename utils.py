import bcrypt
import random
import string


# 🔐 HASH PASSWORD
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


# 🔐 VERIFY PASSWORD
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)


# 🎯 GENERATE UNIQUE PG CODE
def generate_pg_code(conn):
    for _ in range(100):
        letters = ''.join(random.choices(string.ascii_uppercase, k=2))
        numbers = ''.join(random.choices(string.digits, k=4))
        code = f"PG-{letters}{numbers}"

        exists = conn.execute(
            "SELECT 1 FROM pgs WHERE pg_code=?", (code,)
        ).fetchone()

        if not exists:
            return code

    raise Exception("PG code generation failed")


# 📧 CLEAN EMAIL (IMPORTANT FIX)
def normalize_email(email):
    return email.strip().lower()