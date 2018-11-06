#Best model
import pickle
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessor as prep ###### Twitter preprocessor
from sklearn.externals import joblib # to save & load the model 

## Load the model:
cbc = joblib.load('./Models/best-model.joblib')

"""
Both files are stored in .pkl format.
1) x_test : list containing all tweets of users
2) y_test : contains binary class values as 1: Hate | 0:Counter

"""

## Load your testing test features and lablels:
x_test = pickle.load(open(os.path.join(PATH,'x_test.pkl'),'rb'))
y_test = pickle.load(open(os.path.join(PATH,'y_test.pkl'),'rb'))

"""
We have already described how to create Tfidf and Lexicon Features in our previous model.
Now to get complete feature set use: Vader,Textblob and profanity.

# Dependencies:
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import textblob
Profanity : Download the profane words from: 
            https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words 
"""
#Run vader sentiment on x_test, you will get 4 different sentiment values namely neutral,negative
#,positive,compound and for textblob you will get polarity and subjectivity.
#Store them as sentiment.pkl and textblob.pkl.
# For calculating profanity we check the number of profane words in tweets of individual users 
#to prepare a list for all users and store it as profane.pkl.
"""
So Here features are composed of :
tfidf[word+char] + user history + Lexical[empath] + Sentiments[Vader+TextBlob+Profane] 
in the respective order
"""
feature = np.c_[np.asarray(word_features.todense()),np.asarray(char_features.todense()),lexical_features,sentiment.pkl,textblob.pkl,profane.pkl]

"""
We have provided our already prepared feature vector and labels for a test run : 
./Models/features.pkl | ./Models/labels.pkl
"""
#print classification report of your model's performance:
print(classification_report(y_test,cbc.predict(features)))
print('Accuracy:',cbc.score(features,y_test))





