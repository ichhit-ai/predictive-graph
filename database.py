import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "graphify_swarm.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Enable WAL mode for high concurrency
    cursor.execute("PRAGMA journal_mode=WAL;")
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            embedding TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            description TEXT,
            confidence REAL DEFAULT 1.0
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            type TEXT NOT NULL,
            quote TEXT,
            weight INTEGER DEFAULT 1,
            confidence REAL DEFAULT 1.0,
            FOREIGN KEY(source) REFERENCES nodes(id),
            FOREIGN KEY(target) REFERENCES nodes(id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            node_id TEXT PRIMARY KEY,
            embedding TEXT NOT NULL,
            FOREIGN KEY(node_id) REFERENCES nodes(id)
        )
    """)
    
    conn.commit()
    conn.close()

def clear_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS embeddings")
    cursor.execute("DROP TABLE IF EXISTS edges")
    cursor.execute("DROP TABLE IF EXISTS nodes")
    cursor.execute("DROP TABLE IF EXISTS chunks")
    conn.commit()
    conn.close()
    init_db()

def save_chunk(text, embedding_vector=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    emb_str = json.dumps(embedding_vector) if embedding_vector else None
    cursor.execute("INSERT INTO chunks (text, embedding) VALUES (?, ?)", (text, emb_str))
    chunk_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return chunk_id

def get_all_chunks():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chunks")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_node(node_id, name, node_type, description, confidence=1.0):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO nodes (id, name, type, description, confidence)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            description = excluded.description,
            confidence = excluded.confidence
    """, (node_id.lower(), name, node_type, description, confidence))
    conn.commit()
    conn.close()

def save_edge(edge_id, source, target, edge_type, quote, confidence=1.0):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO edges (id, source, target, type, quote, weight, confidence)
        VALUES (?, ?, ?, ?, ?, 1, ?)
        ON CONFLICT(id) DO UPDATE SET
            weight = weight + 1,
            confidence = (confidence + excluded.confidence) / 2.0
    """, (edge_id, source.lower(), target.lower(), edge_type, quote, confidence))
    conn.commit()
    conn.close()

def save_embedding(node_id, embedding_vector):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO embeddings (node_id, embedding)
        VALUES (?, ?)
        ON CONFLICT(node_id) DO UPDATE SET
            embedding = excluded.embedding
    """, (node_id.lower(), json.dumps(embedding_vector)))
    conn.commit()
    conn.close()

# Batch save functions — one connection, one commit per batch
def save_nodes_batch(nodes_list):
    """Save a list of dicts [{id, name, type, description, confidence}, ...] in one transaction."""
    conn = get_db_connection()
    cursor = conn.cursor()
    for n in nodes_list:
        conf = n.get("confidence", 1.0)
        cursor.execute("""
            INSERT INTO nodes (id, name, type, description, confidence)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                description = excluded.description,
                confidence = excluded.confidence
        """, (n["id"].lower(), n["name"], n["type"], n["description"], conf))
    conn.commit()
    conn.close()

def save_edges_batch(edges_list):
    """Save a list of dicts [{id, source, target, type, quote, confidence}, ...] in one transaction."""
    conn = get_db_connection()
    cursor = conn.cursor()
    for e in edges_list:
        conf = e.get("confidence", 1.0)
        cursor.execute("""
            INSERT INTO edges (id, source, target, type, quote, weight, confidence)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(id) DO UPDATE SET
                weight = weight + 1,
                confidence = (confidence + excluded.confidence) / 2.0
        """, (e["id"], e["source"].lower(), e["target"].lower(), e["type"], e["quote"], conf))
    conn.commit()
    conn.close()

def save_embeddings_batch(embeddings_list):
    """Save a list of dicts [{node_id, embedding}, ...] in one transaction."""
    conn = get_db_connection()
    cursor = conn.cursor()
    for emb in embeddings_list:
        cursor.execute("""
            INSERT INTO embeddings (node_id, embedding)
            VALUES (?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                embedding = excluded.embedding
        """, (emb["node_id"].lower(), json.dumps(emb["embedding"])))
    conn.commit()
    conn.close()

def get_all_nodes():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM nodes")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_edges():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM edges")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_embeddings():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM embeddings")
    rows = cursor.fetchall()
    conn.close()
    return {r["node_id"]: json.loads(r["embedding"]) for r in rows}

