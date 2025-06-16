#jaccard similarity intersection / union
import numpy as np
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords,wordnet
import string
doc_1 = "Data is the new oil of the digital economy"
doc_2 = "Data is a new oil"


def Jaccard_Similarity(doc1, doc2):
    # List the unique words in a document
    words_doc1 = set(doc1.lower().split())
    words_doc2 = set(doc2.lower().split())

    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)

    # Calculate Jaccard similarity score
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)

stopwords_list = set(stopwords.words('english'))
def preprocessing_jaccard(doc1,doc2):
    words_doc1 = set(doc1.lower().split())
    words_doc2 = set(doc2.lower().split())
    words_doc1_re = set([x for x in words_doc1 if x not in stopwords_list])
    words_doc2_re = set([x for x in words_doc2 if x not in stopwords_list])
    intersection = words_doc1_re.intersection(words_doc2_re)
    union = words_doc1_re.union(words_doc2_re)
    return float(len(intersection)) / len(union)



lemmatizer= WordNetLemmatizer()
word_dict = []
def preprocess_text(texts):
    for text in texts : 
    
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
        word_dict.extend(tokens)
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world we live in.",
    "Deep learning techniques have greatly improved image recognition.",
    "Natural language processing allows computers to understand human language.",
    "Data science combines statistics, computer science, and domain knowledge.",
    "The weather is nice today, perfect for a walk in the park.",
    "Cats are often seen as independent and curious creatures.",
    "The stock market fluctuates based on various economic indicators.",
    "Exploring new cuisines can be an exciting culinary adventure.",
    "Machine learning algorithms can learn from data and make predictions.",
]
# Preprocess sentences
# preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
# preprocess_text(sentences)
# set_dict = set(word_dict)

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Preprocess the sentences (tokenization)
tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Access word vectors
# word_vector = model.wv['machine']  # Example: Get vector for the word 'word'
# print("Vector for 'word':", word_vector)

# Find similar words
# similar_words = model.wv.most_similar('machine', topn=5)
# print("Words similar to 'word':", similar_words)
text = 'Deep learning is the natural language processing'
words = simple_preprocess(text)

# Lấy vector của từng từ nếu có trong vocabulary của mô hình
word_vectors = [model.wv[word] for word in words if word in model.wv]

# Kết hợp các vector từ - ở đây sử dụng trung bình cộng
if word_vectors:
    text_vector = np.mean(word_vectors, axis=0)
    print("Vector của văn bản:", text_vector)
else:
    print("Không có từ nào trong mô hình Word2Vec.")
