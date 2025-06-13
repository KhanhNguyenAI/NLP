import pandas as pd 
import string
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords,wordnet
from nltk import word_tokenize,sent_tokenize
from nltk import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#read file csv
df = pd.read_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\NLP\project\sentiments\csv\tripadvisor_hotel_reviews.csv')

#check file
'''
print(df.columns) # Review Rating

print(df.head())
print(df.describe)
print(df.info())
print(df.isna().sum()) #0 0
'''
#remove space, double space,replace punctuation 
def remove_punctuation(df_name):
    df[df_name] = df[df_name].str.replace(r'/n','',regex = True)
    df[df_name] = df[df_name].str.replace(r'/s+','',regex = True)
    df[df_name] = df[df_name].apply(lambda x : x.translate(str.maketrans('','',string.punctuation)))
    df[df_name] = df[df_name].str.lower().str.strip()
remove_punctuation('Review')

# remove stop words which have in Review(text)
stop_words_list = set(stopwords.words('english'))
def remove_stopwords(text):
    words = nltk.word_tokenize(text)
    removed = [word for word in words if word.lower() not in stop_words_list]
    return ' '.join(removed)
df['Review'] = df['Review'].apply(remove_stopwords)

#convert to standard form of word in text (review)
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
df['Review'] = df['Review'].apply(lematization)

# i want to copt a Rating to compare to origin Rating of customers to know exactly good/bad rating
df['Rating_copy'] = df['Rating'].copy()

# convert 1 2 3 Rating into Positive(1) or Negative(0) and 4 5 Rating into Positive(1) ----> binary classification
# i think that's good . yeah 
sia = SentimentIntensityAnalyzer()
def sentiment(text):
    score = sia.polarity_scores(text)
    sentiment_label = 0 if score['neg'] + 0.05> score['pos'] else 1
    return sentiment_label
df.loc[df['Rating'] == 3, 'Rating_copy'] = df.loc[df['Rating'] == 3, 'Review'].apply(sentiment)

df.loc[df['Rating'].isin([4, 5]), 'Rating_copy'] = 1
df.loc[df['Rating'].isin([1,2]), 'Rating_copy'] = 0

# print(df.head())
# print(df.isna().sum())
df.to_csv(r"C:\Users\97ngu\OneDrive\Desktop\course\NLP\project\sentiments\csv\Hotelprocessed_data.csv", index=False, encoding="utf-8")