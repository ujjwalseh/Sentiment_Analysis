from django.shortcuts import render
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle as pkl
import re
import nltk

# Create your views here.

def index(request):
    return render(request, 'index.html')

def load(fileName):
    file = open(fileName, 'rb')
    data = pkl.load(file)
    file.close()
    return data

def predict(request):

    # Loading our saved Objects
    # 1.) Object for CountVectorizer
    cv = load('bow_dict.pkl')

    # 2.) Object for prediction
    classifier = load('model.pkl')

    # Will get the Review written by the user
    review = request.GET['Review']
    
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    # Performing Data Preprocessing, i.e Function to clean the Review entirely.
    def filtering(review):
        corpus = []
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
        return corpus

    # Transformations for Cleaned Review
    x_fresh = cv.transform(filtering(review)).toarray()

    # Final Prediction
    pred = classifier.predict(x_fresh)

    if pred[0] == 1:
        msg = "Positive😃"
    else:
        msg = "Negative😔"    
    return render(request, 'prediction.html', {'prediction' : msg})
