import pandas as pd 
import string
import numpy as np
import nltk 
nltk.download('punkt_tab')
nltk.download('stopwords')
# # nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

train_ds = pd.read_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\exNLP\ag_news_csv\ag_news_csv\train.csv')
test_ds = pd.read_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\exNLP\ag_news_csv\ag_news_csv\test.csv')
train_ds.columns = ['index','title', 'text']
test_ds.columns = ['index','title', 'text']

World= train_ds[train_ds['index'] == 1]
Sportsd = train_ds[train_ds['index'] == 2]

Business = train_ds[train_ds['index'] == 3]
Sci_Tech = train_ds[train_ds['index'] == 4]

# print(train_ds.shape) #(119999, 3)
# print(test_ds.shape) #(7599, 3)
def replace_text(df_text,text):
    df_text[text] = df_text[text].str.replace(r'\n', '', regex=True)
    # df_text[text] = df_text[text].str.replace(r'\s+', '', regex=True)  #cach tap xuong dong 
    df_text[text] = df_text[text].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    df_text[text] = df_text[text].str.lower()
    
replace_text(train_ds, 'title')
replace_text(train_ds, 'text')
replace_text(test_ds, 'title')
replace_text(test_ds, 'text')
# print(np.mean(train_ds['text'].str.len()))187.158326319386
# print(np.mean(test_ds['text'].str.len()))186.32872746414003
train_ds['tokens_text'] = train_ds['text'].apply(word_tokenize)
english_stopwords = set(stopwords.words('english'))
def remove_stopwords(tokens):
    return [word for word in tokens if word not in english_stopwords]
train_ds['tokens_text'] = train_ds['tokens_text'].apply(remove_stopwords)



all_words = []
for tokens_list in train_ds['tokens_text']:
    all_words.extend(tokens_list) # Gom tất cả các danh sách từ lại thành một danh sách lớn
word_counts = Counter(all_words)
# print("Các từ và tần suất xuất hiện (top 10):")
# print(word_counts.most_common(10))
# print("-" * 30)

# 4.2. Định nghĩa ngưỡng tần suất tối thiểu
min_freq_threshold = 3 # Ví dụ: Loại bỏ các từ chỉ xuất hiện 2 lần

# Tạo tập hợp các từ phổ biến (không hiếm gặp)
frequent_words = {word for word, count in word_counts.items() if count >= min_freq_threshold}

# 4.3. Định nghĩa hàm để loại bỏ từ hiếm gặp
def remove_rare_words(tokens, frequent_words_set):
    return [word for word in tokens if word in frequent_words_set]

# Áp dụng hàm remove_rare_words lên cột 'filtered_tokens'
train_ds['tokens_text'] = train_ds['tokens_text'].apply(lambda x: remove_rare_words(x, frequent_words))

# print(train_ds.columns)

###Stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]
train_ds['stemmed_tokens'] = train_ds['tokens_text'].apply(stem_tokens)
# print(train_ds['stemmed_tokens'].head())
# print(train_ds['tokens_text'].head()) # 17.0

#Lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens):
    pos_tagged_sentence = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(word,wordnet_map.get(pos[0], wordnet.NOUN)) for word,pos in pos_tagged_sentence]
train_ds['lemmatized_tokens'] = train_ds['tokens_text'].apply(lemmatize_tokens)
train_ds = train_ds.drop(columns=['text','tokens_text','stemmed_tokens'],axis = 0 )
train_ds.to_csv('text2.csv', index=False, encoding='utf-8')
