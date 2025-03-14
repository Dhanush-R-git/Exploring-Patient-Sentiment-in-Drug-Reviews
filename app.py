from flask import Flask, request, jsonify,redirect,url_for,render_template
from sqlalchemy import create_engine, text
import pandas as pd
from preprocessing import preprocess_data
from sentiment_analysis import analyze_sentiment

app = Flask(__name__)

# Connect to MySQL database
engine = create_engine('mysql+mysqlconnector://root:Saravanan007@localhost/drug_reviews')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/enter_review', methods=['GET', 'POST'])
def enter_review():
    if request.method == 'POST':
        # Retrieve drug name and review from the form submission
        drug_name = request.form['drug_name']
        review = request.form['review']
        rating = int(request.form.get('rating'))
        df = pd.DataFrame({'Review': [review],'rating':[rating]})
        df = preprocess_data(df)

        # Analyze sentiment
        df = analyze_sentiment(df)

        # Get the sentiment score
        sentiment_score = df['sentiment_score'][0]
        processed_review="N"
        print(drug_name,review,rating,processed_review,sentiment_score)
        conn = engine.connect()
        query = text("INSERT INTO processed_drug (drugName, Review, rating, Processed_Review, sentiment_score) VALUES (:drug_name, :review, :rating, :processed_review, :sentiment_score)")
        conn.execute(query,{
    'drug_name': drug_name,
    'review': review,
    'rating': rating,
    'processed_review': processed_review,
    'sentiment_score': sentiment_score
})
        conn.commit()
        print("Data inserted")
        # Redirect to the home page after adding the review
        return redirect(url_for('index'))
    else:
        return render_template('feedback.html')


@app.route('/get_sentiment')
def get_sentiment():
    drug_name = request.args.get('drug_name')
    conn = engine.connect()
    # Execute SQL query to get the average sentiment score for the specific drug
    query = f"SELECT AVG(sentiment_score) FROM processed_drug WHERE drugName='{drug_name}'"
    result = conn.execute(text(query)).fetchone()

    sentiment_score = result[0] if result else None
    if sentiment_score is not None:
        return jsonify(sentiment_score=sentiment_score)
    else:
        return jsonify(error="Drug Not Found")


@app.route('/get_review_count', methods=['GET'])
def get_review_count():
    drug_name = request.args.get('drug_name')
    conn = engine.connect()
    # Execute SQL query to get the average sentiment score for the specific drug
    query = f"SELECT COUNT(*) FROM processed_drug WHERE drugName ='{drug_name}'"
    result = conn.execute(text(query)).fetchone()

    count = result[0] if result else None
    if count is not None:
        return jsonify(review_count=count)
    else:
        return jsonify(error="Drug Not Found")


if __name__ == '__main__':
    app.run(debug=True)
