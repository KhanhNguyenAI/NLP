#pip install transformers datasets sentencepiece
from transformers import pipeline

# Tạo pipeline tóm tắt
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Văn bản cần tóm tắt
text = """
Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods
based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised
or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks,
deep reinforcement learning, recurrent neural networks and convolutional neural networks have been applied
to fields including computer vision, speech recognition, natural language processing, machine translation,
bioinformatics, drug design, medical image analysis, material inspection and board game programs,
where they have produced results comparable to and in some cases surpassing human expert performance.
"""

# Tóm tắt văn bản
summary = summarizer(text, max_length=130, min_length=30, do_sample=False)

# Hiển thị kết quả
print("Tóm tắt:")
print(summary[0]['summary_text'])

