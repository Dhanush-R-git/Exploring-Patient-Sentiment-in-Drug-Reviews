from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
def analyze_sentiment(sorted_df):
    # Load the model
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    res = {}
    for i, row in tqdm(sorted_df.iterrows(), total=len(sorted_df)):
        try:
            text = row['Review']
            rating=row['rating']
            encoded_text = tokenizer(text, return_tensors='pt')
            output = model(**encoded_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            scores_dict = {
                'roberta_neg': scores[0],
                'roberta_neu': scores[1],
                'roberta_pos': scores[2]
            }
            if(rating>=8):
                sentiment=(rating*0.1*scores[2])+scores[1]+scores[0]
            elif rating<=4:
                sentiment=(rating*(-0.1)*scores[0])+scores[1]+scores[2]
            else:
                sentiment=(rating*(0.1)*scores[1])+scores[0]+scores[2]
            sorted_df.at[i, 'sentiment_score'] = sentiment
        except RuntimeError:
            print(f'Broke for id')
    return sorted_df
