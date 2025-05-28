import pandas as pd 
import string
import nltk
from nltk.tokenize import word_tokenize
#============
from nltk.corpus import stopwords 
nltk.download('stopwords')
#==================
df = pd.read_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\NLP\emailspawn\email.csv')


def replace(df_name): 
    df[df_name] = df[df_name].str.replace(r'/n','',regex = True)
    df[df_name] = df[df_name].str.replace(r'/s+','',regex = True)
    df[df_name] = df[df_name].apply(lambda x :x.translate(str.maketrans('','',string.punctuation)))
    df[df_name] = df[df_name].str.lower()

replace('Message')
# print(df.head())
#===============
stop_words = set(stopwords.words('english'))
# print(stop_words, "\n")
def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)
df['Message'] = df['Message'].apply(remove_stopwords)

#===================
from collections import Counter 
all_words = ' '.join(df['Message']).split()
word_counts = Counter(all_words)
# print(word_counts.most_common()[:-10:-1])
k = 3
rare_word = set([word for word, count in word_counts.items() if count <= k])
def remove_rare_words(text): 
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in rare_word]
    return ' '.join(filtered_words)
# df['Message'] = df['Message'].apply(remove_rare_words)
# all_words = ' '.join(df['Message']).split()
# word_counts = Counter(all_words)
# print(word_counts.most_common()[:-10:-1])
print(df.head())
#=======================================
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
df['Message'] = df['Message'].apply(lematization)
df.to_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\NLP\emailspawn\propressedDATApropressedDATA.csv', index=False, encoding='utf-8')

