# backend/rag_agent/db.py

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "rag_data.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    # Table for ingested documents
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ingested_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            status TEXT,
            created_at TIMESTAMP
        )
    """)
    # Table for chat history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_query TEXT,
            bot_response TEXT,
            timestamp TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def log_ingestion(filename: str, status: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO ingested_files (filename, status, created_at)
        VALUES (?, ?, ?)
    """, (filename, status, datetime.now()))
    conn.commit()
    conn.close()


def save_chat(user_query: str, bot_response: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_history (user_query, bot_response, timestamp)
        VALUES (?, ?, ?)
    """, (user_query, bot_response, datetime.now()))
    conn.commit()
    conn.close()


def get_chat_history(limit: int = 50):
    """
    Returns last `limit` messages from chat history as a list of tuples:
    [(user_query, bot_response, timestamp), ...]
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_query, bot_response, timestamp
        FROM chat_history
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows


# Initialize DB automatically
init_db()
