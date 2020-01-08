from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import sys
import nltk
import pandas as pd
import string

from nltk.corpus import stopwords
stop_words=stopwords.words('english')
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB

ps=PorterStemmer()
cv=CountVectorizer()

df = pd.read_csv('https://storage.googleapis.com/datsets/Restaurant_Reviews.txt',sep='\t')

def clean_text(msg):
    '''
    1:remove punctuation
    2:remove stopwords
    3:steming
    '''
    new_msg=[w for w in msg if w not in string.punctuation]
    new_msg2=''.join(new_msg)
    tmp_list=[]
    for ww in new_msg2.split():        
        if(ww.lower() not in stop_words):
            tmp_list.append(ww.lower())
    new_msg3=' '.join(tmp_list)
    new_msg4=[ps.stem(w) for w in new_msg3.split()]
    return ' '.join(new_msg4)
df['Review']=df.Review.apply(clean_text)
sparse_mtr=cv.fit_transform(df.Review)
X=sparse_mtr.toarray()
y=df.Liked	
gnb=MultinomialNB()
gnb.fit(X,y)

app = Flask(__name__)
api = Api(app)
# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        # vectorize the user's query and make a prediction
        test=clean_text([user_query])
        test_X=cv.transform([user_query]).toarray()
        prediction = gnb.predict(test_X)

        # Output either 'Negative' or 'Positive' along with the score
        if prediction == 0:
            pred_text = 'Not Liked'
        else:
            pred_text = 'Liked'
            

        # create JSON object
        output = {'prediction': pred_text}
        
        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')


if __name__ == '__main__':
    app.run(debug=True)
