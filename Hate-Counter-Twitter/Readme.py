#Dataset Description : 

"""
Hate_Counter_Dataset.csv contains the Twitter IDs of Hate users mapped with corresponding  Counter users.
There are in total 1290 pairs of hate and counter users.

"""



#Code Reproduction : 
##Make sure you have got all Dependencies before running the model

import pickle
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessor as prep ###### Twitter preprocessor
from sklearn.externals import joblib # to save & load the model 

#We have got the best accuracy score on CatBoostClassifier. 
#You can always reproduce the results by importing  the model by following commands below:


#**********************************************************************************************************************************


#Model-I
""" 
Description : In this model we  only used Tfidf Vectors (Generated from both Word and Character Vocabulary) as our features

"""

#Feature Preparation:
# For Tfidf feature generation use our pretrained vocabulary with the code below:
"""
	char_vocab.pkl - contains character level vocabulary trained on 6 million tweets. 
	word_vocab.pkl - contains word level vocabulary trained on 6 million tweets. 

"""

## Load the model:
cbc = joblib.load('Refined_Project_Dataset/Catboost-model-tfidf.joblib')

# Fetch your test tweets and labels:
"""
1) x_test : list containing all tweets of users
2) y_test : contains binary class values as 1: Hate | 0:Counter

"""
x_test = pickle.load(open('Refined_Project_Dataset/xw_test_tfidf.pkl','rb'))
y_test = pickle.load(open('Refined_Project_Dataset/yw_test_tfidf.pkl','rb'))

# # Preprocessing:-------------------------------------
prep_tweets = []
for tweet in tqdm(x_test):
    prep_tweets.append(prep.tokenize(x_test))
#------------------------------------------------------

# # TF-IDF Vectorizers :
word_vectorizer = TfidfVectorizer(vocabulary=pickle.load(open("word_vocab.pkl", "rb")) # pretrained vocabulary from 6 million tweets on word level
char_vectorizer = TfidfVectorizer(vocabulary=pickle.load(open("char_vocab.pkl", "rb")) # pretrained vocabulary from 6 million tweets on char level

char_features = char_vectorizer.transform(prep_tweets)
word_features = word_vectorizer.transform(prep_tweets)

# # Prepare features[word + char]:
features = np.c_[np.asarray(word_features.todense()),np.asarray(char_features.todense())]

#print classification report of your model's performance:
print(classification_report(y_test,cbc.predict(x_test)))
print('Accuracy:',cbc.score(x_test,y_test)) 




#******************************************************************************************************************************************





# Model-II
"""
 Description : This is model is where we have got our best accuracy results on testing data

"""
# Use this model after you get all your required features:
"""
  Features Description : [Orderly]
  i)Tfidf : Word Vectors + Char Vectors 	[Orderly]
  ii)Lexicon Features [Empath]
  iii)Sentiment Features [Vader] + TextBlob 
  iv)User History:[Order is maintained]
				   1)followers_count/tweet	
				   2)favourites_count/tweet
				   3)friends_count/tweet
				   4)listed_count/tweet
				   5)statuses_count/tweet
				   6)verified

"""

## Load the model:
cbc = joblib.load('Refined_Project_Dataset/Catboost-model.joblib')


## Load your testing test features and lablels:
x_test = pickle.load(open('Refined_Project_Dataset/xw_test.pkl','rb'))
y_test = pickle.load(open('Refined_Project_Dataset/yw_test.pkl','rb'))

#print classification report of your model's performance:
print(classification_report(y_test,cbc.predict(x_test)))
print('Accuracy:',cbc.score(x_test,y_test))

