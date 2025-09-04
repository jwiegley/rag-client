"""PostgreSQL storage implementations for RAG client."""

import pickle
from typing import Any
from urllib.parse import urlparse

import psycopg2


class PostgresDetails:
    """Helper class for PostgreSQL connection details and operations."""
    
    connection_string: str
    database: str
    host: str
    password: str
    port: int
    user: str
    
    def __init__(self, connection_string: str):
        """Initialize from connection string.
        
        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string
        parsed = urlparse(self.connection_string)
        self.database = parsed.path.lstrip("/")
        self.host = parsed.hostname or "localhost"
        self.password = parsed.password or ""
        self.port = parsed.port or 5432
        self.user = parsed.username or "postgres"
    
    def unpickle_from_table[T](self, tablename: str, row_id: int) -> Any:
        """Unpickle an object from a PostgreSQL table.
        
        Args:
            tablename: Name of the table
            row_id: ID of the row to retrieve
            
        Returns:
            Unpickled object or None if not found
        """
        with psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT data FROM {tablename} WHERE id = %s",
                    (row_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                
                binary_data = row[0]
                if isinstance(binary_data, memoryview):
                    binary_data = binary_data.tobytes()
                return pickle.loads(binary_data)
    
    def pickle_to_table[U](self, tablename: str, row_id: int, data: object):
        """Pickle an object to a PostgreSQL table.
        
        Args:
            tablename: Name of the table
            row_id: ID for the row
            data: Object to pickle and store
        """
        # Connect to PostgreSQL
        with psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {tablename} (
                        id SERIAL PRIMARY KEY,
                        data BYTEA
                    )
                """
                )
                
                pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                
                cur.execute(
                    f"""
                    INSERT INTO {tablename} (id, data)
                    VALUES (%s, %s)
                    ON CONFLICT (id)
                    DO UPDATE SET data = EXCLUDED.data
                """,
                    (row_id, psycopg2.Binary(pickled)),
                )