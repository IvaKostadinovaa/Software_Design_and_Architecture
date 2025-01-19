# models/stock_model.py

import sqlite3
import pandas as pd
import numpy as np

DB_NAME = 'stock_data.db'

def get_stock_data(page=1, table="stock_data", limit=10):
    """
    Fetch distinct issuers (Код_на_издавач) from the database
    using pagination (page and limit).
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    offset = (page - 1) * limit
    query = f"""
        SELECT DISTINCT Код_на_издавач
        FROM {table}
        LIMIT {limit} OFFSET {offset}
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    # Map each row into a dict with Код_на_издавач
    stock_data = [{'Код_на_издавач': row[0]} for row in rows]

    conn.close()
    return stock_data


def get_all_stock_data(table="stock_data"):
    """
    Fetch all rows from the database table.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    query = f"SELECT * FROM {table}"
    cursor.execute(query)
    rows = cursor.fetchall()

    # Map rows to a list of dictionaries
    stock_data = [
        {
            'Код_на_издавач': row[0],
            'Датум': row[1],
            'Цена_на_последна_трансакција': row[2],
            'Макс': row[3],
            'Мин': row[4],
            'Просечна_цена': row[5],
            'Промет_во_БЕСТ_во_денари': row[6],
            'Купен_промет_во_денари': row[7],
            'Количина': row[8],
            'Промет_во_Бест_во_денари_друга': row[9]
        }
        for row in rows
    ]

    conn.close()
    return stock_data


def get_total_issuers_count(table="stock_data"):
    """
    Return the total count of unique Код_на_издавач in the table.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    query = f"SELECT COUNT(DISTINCT Код_на_издавач) FROM {table}"
    cursor.execute(query)

    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_filtered_data_for_analysis(issuer='', page=1, limit=10, table="stock_data"):
    """
    Fetch data filtered by issuer (optional), with pagination.
    Returns the rows, total_pages, and total_rows.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Base query
    query = f"SELECT * FROM {table} WHERE 1=1"
    params = []

    # If issuer is provided, filter by issuer
    if issuer:
        query += " AND Код_на_издавач = ?"
        params.append(issuer)

    # Count total rows for this filter
    count_query = f"SELECT COUNT(*) FROM {table} WHERE 1=1"
    if issuer:
        count_query += " AND Код_на_издавач = ?"

    cursor.execute(count_query, [issuer] if issuer else [])
    total_rows = cursor.fetchone()[0]

    total_pages = (total_rows + limit - 1) // limit

    # Add pagination
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, (page - 1) * limit])

    cursor.execute(query, params)
    rows = cursor.fetchall()

    # Format into a list of dictionaries
    stock_data = [
        {
            'Код_на_издавач': row[0],
            'Датум': row[1],
            'Цена_на_последна_трансакција': row[2],
            'Макс': row[3],
            'Мин': row[4],
            'Просечна_цена': row[5],
            'Промет_во_БЕСТ_во_денари': row[6],
            'Купен_промет_во_денари': row[7],
            'Количина': row[8],
            'Промет_во_Бест_во_денари_друга': row[9],
        }
        for row in rows
    ]

    conn.close()

    return stock_data, total_rows, total_pages


def get_issuer_details(issuer_code, table="stock_data"):
    """
    Fetch all rows for a particular issuer_code.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    query = f"SELECT * FROM {table} WHERE Код_на_издавач = ?"
    cursor.execute(query, (issuer_code,))
    rows = cursor.fetchall()

    stock_data = [
        {
            'Код_на_издавач': row[0],
            'Датум': row[1],
            'Цена_на_последна_трансакција': row[2],
            'Макс': row[3],
            'Мин': row[4],
            'Просечна_цена': row[5],
            'Промет_во_БЕСТ_во_денари': row[6],
            'Купен_промет_во_денари': row[7],
            'Количина': row[8],
            'Промет_во_Бест_во_денари_друга': row[9],
        }
        for row in rows
    ]

    conn.close()
    return stock_data


def get_issuer_data_for_graph(issuer_code, table="stock_data"):
    """
    Fetch date, last transaction price, max, and min
    for plotting/technical strategies.
    """
    conn = sqlite3.connect(DB_NAME)
    query = f"""
        SELECT Датум, Цена_на_последна_трансакција, Мак_, Мин_
        FROM {table}
        WHERE Код_на_издавач = ?
        ORDER BY Датум ASC
    """
    df = pd.read_sql_query(query, conn, params=(issuer_code,))
    conn.close()

    return df


def fetch_data(issuer_code, table="stock_data"):
    """
    Utility to fetch date + last transaction price,
    then resample to weekly. Used by the LSTM prediction logic.
    """
    conn = sqlite3.connect(DB_NAME)
    query = f"""
        SELECT Датум, Цена_на_последна_трансакција
        FROM {table}
        WHERE Код_на_издавач = ?
        ORDER BY Датум
    """
    df = pd.read_sql_query(query, conn, params=(issuer_code,))
    conn.close()

    # Convert date column to datetime
    df['Датум'] = pd.to_datetime(df['Датум'])

    # Set the date as the index
    df.set_index('Датум', inplace=True)

    # Clean numeric data (match your original replacements)
    df['Цена_на_последна_трансакција'] = (
        df['Цена_на_последна_трансакција']
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )

    # Resample to weekly average
    df = df.resample('W').mean()

    # Drop rows with missing values
    df.dropna(inplace=True)

    print(f"Number of rows after aggregation: {len(df)}")
    return df
