import sqlite3
from typing import Any, Iterable


class SQLiteDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def create_table(self, table: str, columns: str):
        """
        Create a table.

        Example:
            db.create_table(
                "users",
                
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                age INTEGER
                
            )
        """
        query = f"CREATE TABLE IF NOT EXISTS {table} ({columns})"
        self.conn.execute(query)
        self.conn.commit()


    def insert(self, table: str, data: dict[str, Any]) -> int:
        """
        Insert one row and return its ID.
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)

        query = f"""
            INSERT INTO {table} ({columns})
            VALUES ({placeholders})
        """

        cursor = self.conn.execute(query, tuple(data.values()))
        self.conn.commit()

        return cursor.lastrowid


    def insert_many(
        self,
        table: str,
        rows: Iterable[dict[str, Any]]
    ):
        """
        Insert multiple rows.
        """
        rows = list(rows)

        if not rows:
            return

        columns = list(rows[0].keys())
        column_names = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)

        query = f"""
            INSERT INTO {table} ({column_names})
            VALUES ({placeholders})
        """

        values = [
            tuple(row[column] for column in columns)
            for row in rows
        ]

        self.conn.executemany(query, values)
        self.conn.commit()


    def get_by_id(self, table: str, record_id: int):
        """
        Retrieve one row by its primary key.
        """
        query = f"SELECT * FROM {table} WHERE id = ?"

        cursor = self.conn.execute(query, (record_id,))
        row = cursor.fetchone()

        return dict(row) if row else None


    def get_all(self, table: str):
        """
        Retrieve all rows.
        """
        query = f"SELECT * FROM {table}"

        cursor = self.conn.execute(query)
        return [dict(row) for row in cursor.fetchall()]


    def find(
        self,
        table: str,
        where: dict[str, Any]
    ):
        """
        Retrieve rows matching conditions.

        Example:
            db.find("users", {"age": 25})
        """
        conditions = " AND ".join(
            f"{column} = ?" for column in where
        )

        query = f"""
            SELECT * FROM {table}
            WHERE {conditions}
        """

        cursor = self.conn.execute(
            query,
            tuple(where.values())
        )

        return [dict(row) for row in cursor.fetchall()]


    def update(
        self,
        table: str,
        data: dict[str, Any],
        where: dict[str, Any]
    ):
        """
        Update rows matching conditions.

        Example:
            db.update(
                "users",
                {"name": "John"},
                {"id": 1}
            )
        """
        set_clause = ", ".join(
            f"{column} = ?" for column in data
        )

        where_clause = " AND ".join(
            f"{column} = ?" for column in where
        )

        query = f"""
            UPDATE {table}
            SET {set_clause}
            WHERE {where_clause}
        """

        values = tuple(data.values()) + tuple(where.values())

        cursor = self.conn.execute(query, values)
        self.conn.commit()

        return cursor.rowcount


    def delete(
        self,
        table: str,
        where: dict[str, Any]
    ):
        """
        Delete rows matching conditions.

        Example:
            db.delete("users", {"id": 1})
        """
        where_clause = " AND ".join(
            f"{column} = ?" for column in where
        )

        query = f"""
            DELETE FROM {table}
            WHERE {where_clause}
        """

        cursor = self.conn.execute(
            query,
            tuple(where.values())
        )

        self.conn.commit()

        return cursor.rowcount


    def execute(
        self,
        query: str,
        params: tuple = ()
    ):
        """
        Execute arbitrary SQL.
        """
        cursor = self.conn.execute(query, params)
        self.conn.commit()

        return cursor

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
