import pandas as pd 
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
df = pd.read_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\NLP\project\sentiments\csv\sentiment.csv')
print(df['reviewText'][120])
def puntuation(df_name):
    df[df_name] = [review.translate(str.maketrans('','',string.punctuation)) for review in df[df_name]]
    df[df_name] =df[df_name].str.strip()
puntuation('reviewText')
stopword_list = set(stopwords.words('english'))
def remove_stopword(text):
    words = nltk.word_tokenize(text)
    remove = [word for word in words if word.lower() not in stopword_list]
    return ' '.join(remove)
df['reviewText'] = df['reviewText'].apply(remove_stopword)
        
# print(df.columns)#reviewText,Positive
# print(df['Positive'].unique())
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
wordnet_map = {'N':wordnet.NOUN,'V':wordnet.VERB,'J':wordnet.ADJ,'R':wordnet.ADV}
lematizer = WordNetLemmatizer()
def lematization(text) : 
    lemmatized_words = []
    words = word_tokenize(text)
    pos_tagged = nltk.pos_tag(words)
    for word, tag in pos_tagged : 
        pos = wordnet_map.get(tag[0].upper(),wordnet.NOUN)
        lemmatized_words.append(lematizer.lemmatize(word,pos = pos))
    return ' '.join(lemmatized_words)
df['reviewText'] = df['reviewText'].apply(lematization)


# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from sklearn.metrics import classification_report
# sia = SentimentIntensityAnalyzer()


# #{'neg': 0.0, 'neu': 0.295, 'pos': 0.705, 'compound': 0.8012}
# def classify_sentiment(text):
#     score = sia.polarity_scores(text)
#     sentiment = 1 if score['pos'] > score['neg'] else 0
#     return sentiment
# df['sentiment_label'] = df['reviewText'].apply(classify_sentiment)
# print(classification_report(df['Positive'],df['sentiment_label']))
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

from imblearn.pipeline import Pipeline
X = df['reviewText']
y = df['Positive']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# pipeline = Pipeline([
#     ('vectorizer', CountVectorizer(max_features=5000)),
#     ('classifier', BernoulliNB())
# ])
# pipeline.fit(X_train, y_train) 
# y_pred = pipeline.predict(X_test)
# print(classification_report(y_test, y_pred, target_names=['neg', 'pos']))
