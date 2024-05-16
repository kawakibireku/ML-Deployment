import sqlite3
import pandas as pd

def create_table():
    conn = sqlite3.connect('database.db')
    conn.execute('CREATE TABLE IF NOT EXISTS review (id INTEGER PRIMARY KEY AUTOINCREMENT, review TEXT, sentiment_positive FLOAT, sentiment_negative FLOAT, sentiment_neutral FLOAT, summary_sentiment TEXT)')
    conn.commit()
    conn.close()

def insert_review(review, sentiment_positive, sentiment_negative, sentiment_neutral, summary_sentiment):
    conn = sqlite3.connect('database.db')
    conn.execute('INSERT INTO review (review, sentiment_positive, sentiment_negative, sentiment_neutral, summary_sentiment) VALUES (?, ?, ?, ?, ?)', (review, sentiment_positive, sentiment_negative, sentiment_neutral, summary_sentiment))
    conn.commit()
    conn.close()

def get_reviews():
    conn = sqlite3.connect('database.db')
    cursor = conn.execute('SELECT * FROM review')
    reviews = cursor.fetchall()
    conn.close()

    df = pd.DataFrame(reviews, columns=['id', 'review', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 'summary_sentiment'])
    return df