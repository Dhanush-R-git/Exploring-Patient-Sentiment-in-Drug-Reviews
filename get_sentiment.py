from sqlalchemy import create_engine,text

def get_avg_sentiment_score(drug_name):
    try:
        # Create SQLAlchemy engine
        engine = create_engine('mysql+mysqlconnector://root:Saravanan007@localhost/drug_reviews')

        # Establish a connection to the database
        conn = engine.connect()

        # Execute SQL query to get the average sentiment score for the specific drug
        query = f"SELECT AVG(sentiment_score) FROM preprocessed_drugtrain WHERE drugName='{drug_name}'"
        print("Query:", query)
        result = conn.execute(text(query)).fetchone()
        avg_sentiment_score = result[0] if result else None
        if avg_sentiment_score==None:
            print("No drug has been found.....")
        else:
            print(f"Average sentiment score for {drug_name}: {avg_sentiment_score}")
    except Exception as e:
        print("An error occurred:", e)

    finally:
        # Close connection
        if conn:
            conn.close()

# Example usage:
drug_name = 'Valsartan'
get_avg_sentiment_score(drug_name)
