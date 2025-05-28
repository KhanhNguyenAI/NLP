import pandas as pd
import string
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

df = pd.read_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\exNLP\ag_news_csv\ag_news_csv\train.csv')
df.columns = ['index','title', 'text']
#==============================================
def replace_text(df_text,text):
    df_text[text] = df_text[text].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df_text[text] = df_text[text].str.replace(r'\n', '', regex=True)
    df_text[text] = df_text[text].str.replace('  ', ' ', regex=True)
    df_text[text] = df_text[text].str.lower()
replace_text(df,'title')
replace_text(df,'text')
#===========================================================
df['text'] = df['text'].apply(word_tokenize)
df['title'] = df['title'].apply(word_tokenize)
#===========================================================
english_stopwords = set(stopwords.words('english'))
def remove_stopwords(tokens):
    return [word for word in tokens if word not in english_stopwords]
df['text'] = df['text'].apply(remove_stopwords)
df['title'] = df['title'].apply(remove_stopwords)
# print(df['title'])
# print(df['text'])
#==========================================================
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens):
    pos_tagged_sentence = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(word,wordnet_map.get(pos[0], wordnet.NOUN)) for word,pos in pos_tagged_sentence]
df['text'] = df['text'].apply(lemmatize_tokens)
df['title'] = df['title'].apply(lemmatize_tokens)
df.to_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\NLP\processed_data.csv', index=False, encoding='utf-8')