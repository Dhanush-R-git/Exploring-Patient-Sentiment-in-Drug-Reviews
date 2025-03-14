import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
from preprocessing import preprocess_data

def store_in_database(df):
    try:
        # Connect to MySQL database
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Saravanan007",
            database="drug_reviews"
        )

        # Create SQLAlchemy engine
        engine = create_engine('mysql+mysqlconnector://root:Saravanan007@localhost/drug_reviews')

        # Insert data into MySQL table
        df.to_sql('processed_drug', con=engine, if_exists='append', index=False)
        
        print("Data inserted successfully!")

    except Exception as e:
        print("An error occurred:", e)

    finally:
        # Close connection
        if 'conn' in locals() and conn.is_connected():
            conn.close()


df = pd.read_csv("preprocessed_drugtrain.csv")
store_in_database(df)

