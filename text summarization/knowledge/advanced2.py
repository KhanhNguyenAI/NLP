from transformers import pipeline
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# !pip install git+https://github.com/PyTorchLightning/pytorch-lightning
# !pip install git+https://github.com/huggingface/transformers
# !pip install sentencepiece
# !pip install git+https://github.com/stas00/transformers
# !pip install pegasus


#https://colab.research.google.com/drive/1pNMi5BafevId5ClL_dhaVorXW2LYG9V2



# Pick model
model_name = "google/pegasus-xsum"
# Load pretrained tokenizer
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)

example_text = """
Deep learning (also known as deep structured learning) is part of a broader family of machine learning
methods based on artificial neural networks with representation learning. 
Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as 
deep neural networks, deep belief networks, deep reinforcement learning, 
recurrent neural networks and convolutional neural networks have been applied to 
fields including computer vision, speech recognition, natural language processing,
machine translation, bioinformatics, drug design, medical image analysis, 
material inspection and board game programs, where they have produced results 
comparable to and in some cases surpassing human expert performance. 
Artificial neural networks (ANNs) were inspired by information processing and 
distributed communication nodes in biological systems. ANNs have various differences 
from biological brains. Specifically, neural networks tend to be static and symbolic,
while the biological brain of most living organisms is dynamic (plastic) and analogue.
The adjective "deep" in deep learning refers to the use of multiple layers in the network.
Early work showed that a linear perceptron cannot be a universal classifier, 
but that a network with a nonpolynomial activation function with one hidden layer of 
unbounded width can. Deep learning is a modern variation which is concerned with an 
unbounded number of layers of bounded size, which permits practical application and 
optimized implementation, while retaining theoretical universality under mild conditions. 
In deep learning the layers are also permitted to be heterogeneous and to deviate widely 
from biologically informed connectionist models, for the sake of efficiency, trainability 
and understandability, whence the structured part."""

print('Original Document Size:',len(example_text))
# Define PEGASUS model
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)
# Create tokens
tokens = pegasus_tokenizer(example_text, truncation=True, padding="longest", return_tensors="pt")

# Generate the summary
encoded_summary = pegasus_model.generate(**tokens)

# Decode the summarized text
decoded_summary = pegasus_tokenizer.decode(encoded_summary[0], skip_special_tokens=True)

# Print the summary
print('Decoded Summary :',decoded_summary)

summarizer = pipeline(
    "summarization", 
    model=model_name, 
    tokenizer=pegasus_tokenizer, 
    framework="pt"
)

summary = summarizer(example_text, min_length=30, max_length=150)
summary[0]["summary_text"]