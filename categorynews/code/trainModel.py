import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
df = pd.read_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\NLP\processed_data.csv')
# string of list > list
df['text'] = df['text'].apply(ast.literal_eval)
#-------------------------------------------------------
X =df['text']
y = df['index']
#---------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform([" ".join(tokens) for tokens in X_train]) 
#VD : Kết quả của " ".join(['chính', 'phủ', 'ban', 'hành']) sẽ là chuỗi: 'chính phủ ban hành'.
X_test_vec = vectorizer.transform([" ".join(tokens) for tokens in X_test])  #[" ".join(tokens) for tokens in X_test]
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
