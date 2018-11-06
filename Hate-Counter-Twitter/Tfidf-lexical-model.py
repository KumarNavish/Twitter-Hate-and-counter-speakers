# Tfidf-Lexical Model:
import pickle
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessor as prep ###### Twitter preprocessor
from sklearn.externals import joblib # to save & load the model 

## Load the model:
cbc = joblib.load('./Models/tfidf-lexical-model.joblib')

PATH = './Dataset'
# Fetch your test tweets and labels:
"""
Both files are stored in .pkl format.
1) x_test : list containing all tweets of users
2) y_test : contains binary class values as 1: Hate | 0:Counter

"""
x_test = pickle.load(open(os.path.join(PATH,'x_test.pkl'),'rb'))
y_test = pickle.load(open(os.path.join(PATH,'y_test.pkl'),'rb'))


#>>>>Preprocessing:
prep_tweets = []
for tweet in tqdm(x_test):
    prep_tweets.append(prep.tokenize(x_test))
    
##****************************************************************************
# # TF-IDF Vectorizers :
word_vectorizer = TfidfVectorizer(vocabulary=pickle.load(open("word_vocab.pkl", "rb")) # pretrained vocabulary from 6 million tweets on word level
char_vectorizer = TfidfVectorizer(vocabulary=pickle.load(open("char_vocab.pkl", "rb")) # pretrained vocabulary from 6 million tweets on char level

char_features = char_vectorizer.transform(prep_tweets)
word_features = word_vectorizer.transform(prep_tweets)
#*******************************************************************************

#>>>>Lexical Features : 
'''
!pip install empath
from empath import Empath
Run it on x_test and store it in ./Models/ in .pkl format
you will get different 194 lexicon class per user, convert it into a numpy array of shape [None,194]
       
'''
lexical_features = pickle.load(open('./Models/lexicons.pkl','rb'))

#Prepare features[word + char Tfidfs+ Lexicons]:
features = np.c_[np.asarray(word_features.todense()),np.asarray(char_features.todense()),lexical_features]

#print classification report of your model's performance:
print(classification_report(y_test,cbc.predict(features)))
print('Accuracy:',cbc.score(features,y_test))





