
url='https://www.geeksforgeeks.org/nlp/text-summarization-in-nlp/'

example_text = """Deep learning (also known as deep structured learning) is part of a 
broader family of machine learning methods based on artificial neural networks with 
representation learning. Learning can be supervised, semi-supervised or unsupervised. 
Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, 
recurrent neural networks and convolutional neural networks have been applied to 
fields including computer vision, speech recognition, natural language processing, 
machine translation, bioinformatics, drug design, medical image analysis, material
inspection and board game programs, where they have produced results comparable to 
and in some cases surpassing human expert performance. Artificial neural networks
(ANNs) were inspired by information processing and distributed communication nodes
in biological systems. ANNs have various differences from biological brains. Specifically, 
neural networks tend to be static and symbolic, while the biological brain of most living organisms
is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple
layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, 
but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can.
Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, 
which permits practical application and optimized implementation, while retaining theoretical universality 
under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely 
from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, 
whence the structured part."""
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords

stop_words_list = set(stopwords.words('english'))
#frequency of occurrence of the word
freq_words = dict()
words = word_tokenize(example_text)
for word in words : 
    word = word.lower()
    if word in stop_words_list:
        continue
    if word not in freq_words: 
        freq_words[word] = 1
    else : 
        freq_words[word] +=1
# we have the list for frequency of occurrence of the word : {word : frequency}

# text tokenize to create 
text_list = []
for text in sent_tokenize(example_text) : 
    text = text.replace('\n',' ')
    text_list.append(text)
#remove the \n in example text its not benefit 
sent_dict = dict()

#create a frequency for each text (generated from the number of occurrences of words in the text)
for text in text_list : 
    for word,freq in freq_words.items():
        if word in text : 
            if text not in sent_dict : 
                sent_dict[text] = freq
            else:
                sent_dict[text] +=freq


# Giả sử 'paragraph_dict' là từ điển chứa đoạn và tần suất
top_paragraphs = {k: v for k, v in sent_dict.items() if v >= 50}
print(top_paragraphs.keys())


'''
from transformers import pipeline

# Tạo pipeline tóm tắt
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Tóm tắt từng đoạn
for paragraph in top_paragraphs:
    length = len(paragraph.split())
    max_len = min(60, max(20, int(length * 0.75)))  # Rough heuristic
    summary = summarizer(paragraph, max_length=max_len, min_length=max(10, int(max_len * 0.5)), clean_up_tokenization_spaces=True)[0]['summary_text']
    print(summary + "\n")
'''