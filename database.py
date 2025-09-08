import os
import mysql.connector
import json
import sqlite3
from pathlib import Path

# Database configuration for MariaDB
db_config = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_NAME")
}

# SQLite fallback database path
SQLITE_DB_PATH = "agent_memory.db"

def get_db_connection():
    """Establishes a connection to the MariaDB database."""
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as e:
        print(f"Error connecting to MariaDB: {e}")
        return None

def get_sqlite_connection():
    """Establishes a connection to the SQLite fallback database."""
    return sqlite3.connect(SQLITE_DB_PATH)

def setup_database():
    """Creates the tables if they don't exist in MariaDB or SQLite fallback."""
    # Try MariaDB first
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS seen_urls (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    url VARCHAR(2048) NOT NULL,
                    topic VARCHAR(255) NOT NULL,
                    UNIQUE KEY `unique_url_topic` (`url`(255),`topic`)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS synthesis_cache (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    topic VARCHAR(255) NOT NULL,
                    persona VARCHAR(255) NOT NULL,
                    report TEXT NOT NULL,
                    sources TEXT,
                    UNIQUE KEY `unique_topic_persona` (`topic`,`persona`)
                )
            ''')
            conn.commit()
            print("MariaDB tables created successfully")
        finally:
            cursor.close()
            conn.close()
        return
    
    # Fallback to SQLite
    print("Using SQLite fallback database")
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS seen_urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                topic TEXT NOT NULL,
                UNIQUE(url, topic)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synthesis_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                persona TEXT NOT NULL,
                report TEXT NOT NULL,
                sources TEXT,
                UNIQUE(topic, persona)
            )
        ''')
        conn.commit()
        print("SQLite tables created successfully")
    finally:
        cursor.close()
        conn.close()

# --- URL Memory Functions ---

def check_if_url_exists(url: str, topic: str):
    """Checks if a URL for a specific topic is already in the database."""
    # Try MariaDB first
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT id FROM seen_urls WHERE url = %s AND topic = %s", (url, topic.lower()))
            result = cursor.fetchone()
            return result is not None
        finally:
            cursor.close()
            conn.close()
    
    # Fallback to SQLite
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM seen_urls WHERE url = ? AND topic = ?", (url, topic.lower()))
        result = cursor.fetchone()
        return result is not None
    finally:
        cursor.close()
        conn.close()

def add_url(url: str, topic: str):
    """Adds a new URL for a specific topic to the database."""
    if not check_if_url_exists(url, topic):
        # Try MariaDB first
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO seen_urls (url, topic) VALUES (%s, %s)", (url, topic.lower()))
                conn.commit()
                return
            finally:
                cursor.close()
                conn.close()
        
        # Fallback to SQLite
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT OR IGNORE INTO seen_urls (url, topic) VALUES (?, ?)", (url, topic.lower()))
            conn.commit()
        finally:
            cursor.close()
            conn.close()

def get_seen_urls_for_topic(topic: str):
    """Retrieves all previously seen URLs for a specific topic."""
    # Try MariaDB first
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT url FROM seen_urls WHERE topic = %s", (topic.lower(),))
            results = cursor.fetchall()
            return [row[0] for row in results]
        finally:
            cursor.close()
            conn.close()
    
    # Fallback to SQLite
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT url FROM seen_urls WHERE topic = ?", (topic.lower(),))
        results = cursor.fetchall()
        return [row[0] for row in results]
    finally:
        cursor.close()
        conn.close()

# --- Synthesis Cache Functions ---

def get_cached_report(topic: str, persona: str):
    """Retrievels a cached report and safely handles JSON parsing."""
    # Try MariaDB first
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT report, sources FROM synthesis_cache WHERE topic = %s AND persona = %s", (topic.lower(), persona.lower()))
            result = cursor.fetchone()
            
            if result and result.get('sources'):
                try:
                    result['sources'] = json.loads(result['sources'])
                except (json.JSONDecodeError, TypeError):
                    print("Warning: Failed to parse sources from cache. Treating as empty.")
                    result['sources'] = []
            return result
        finally:
            cursor.close()
            conn.close()
    
    # Fallback to SQLite
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT report, sources FROM synthesis_cache WHERE topic = ? AND persona = ?", (topic.lower(), persona.lower()))
        result = cursor.fetchone()
        
        if result:
            result_dict = {'report': result[0], 'sources': result[1]}
            if result_dict.get('sources'):
                try:
                    result_dict['sources'] = json.loads(result_dict['sources'])
                except (json.JSONDecodeError, TypeError):
                    print("Warning: Failed to parse sources from cache. Treating as empty.")
                    result_dict['sources'] = []
            return result_dict
        return None
    finally:
        cursor.close()
        conn.close()

def cache_report(topic: str, persona: str, report: str, sources: list):
    """Saves a new report to the cache with sources as a proper JSON string."""
    sources_json = json.dumps(sources)
    
    # Try MariaDB first
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            query = """
            INSERT INTO synthesis_cache (topic, persona, report, sources) 
            VALUES (%s, %s, %s, %s) 
            ON DUPLICATE KEY UPDATE report = VALUES(report), sources = VALUES(sources)
            """
            cursor.execute(query, (topic.lower(), persona.lower(), report, sources_json))
            conn.commit()
            return
        finally:
            cursor.close()
            conn.close()
    
    # Fallback to SQLite
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO synthesis_cache (topic, persona, report, sources) 
            VALUES (?, ?, ?, ?)
        """, (topic.lower(), persona.lower(), report, sources_json))
        conn.commit()
    finally:
        cursor.close()
        conn.close()