import sqlite3
from typing import Optional, Dict
import bcrypt

DB_PATH = "users.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
        """
    )
    conn.commit()
    # ensure active_session column exists for session tokens
    try:
        cur.execute("ALTER TABLE users ADD COLUMN active_session TEXT DEFAULT NULL")
    except Exception:
        # column probably exists
        pass
    conn.commit()
    conn.close()


def get_user_by_username(username: str) -> Optional[Dict]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def get_user_by_email(email: str) -> Optional[Dict]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def create_user(name: str, username: str, email: str, password: str) -> bool:
    if get_user_by_username(username) or get_user_by_email(email):
        return False

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    # store as utf-8 string
    password_hash_str = password_hash.decode("utf-8")

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (name, username, email, password_hash) VALUES (?, ?, ?, ?)",
        (name, username, email, password_hash_str),
    )
    conn.commit()
    conn.close()
    return True


def set_active_session(username: str, token: Optional[str]):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET active_session = ? WHERE username = ?", (token, username))
    conn.commit()
    conn.close()


def get_active_session(username: str) -> Optional[str]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT active_session FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return row["active_session"]


def verify_user(username: str, password: str) -> Optional[Dict]:
    user = get_user_by_username(username)
    if not user:
        return None
    stored = user.get("password_hash")
    if stored is None:
        return None
    try:
        ok = bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
    except Exception:
        return None
    return user if ok else None
