import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report

df = pd.read_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\NLP\emailspawn\propressedDATApropressedDATA.csv')
# print(df.columns)
X = df['Message']
y = df['Category']
print(X.shape)
# print(y.shape)
X = X.astype(str)

print(X.shape)
# === ======= = ==
y = pd.Categorical(y, categories=['ham','spam'], ordered=True).codes

# ===== shape X , y OK 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15,random_state=42)
y_train_counts = pd.Series(y_train).value_counts()
ham_count = y_train_counts.get(0, 0)
spam_count = y_train_counts.get(1, 0) 
print(ham_count,spam_count)
# # print(X_train.shape)
# # print(y_train.shape)
#--------------------------------------------------
# vectorizer = CountVectorizer(max_features=5000)

# X_train_vectorized = vectorizer.fit_transform(X_train)
# X_test_vectorized = vectorizer.transform(X_test)

# model = BernoulliNB()
# model.fit(X_train_vectorized,y_train)
# y_pre = model.predict(X_test_vectorized)
# print(classification_report(y_test,y_pre))
#================SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(max_features=5000)),
    ('classifier', BernoulliNB())
])
pipeline.fit(X_train, y_train) 
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
#========================