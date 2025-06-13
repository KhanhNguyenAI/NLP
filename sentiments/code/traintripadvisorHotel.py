import pandas as pd 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler



#access the file that preprocessed tripadvisor hotel rating
df = pd.read_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\NLP\project\sentiments\csv\Hotelprocessed_data.csv')
X= df['Review']
y= df['Rating_copy']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = Pipeline([
    ('vectorizer',CountVectorizer(max_features=10000)),
    ('over',RandomOverSampler(random_state=42)),
    ('classifier', BernoulliNB())
])
model.fit(X_train,y_train)
pre = model.predict(X_test)
print(classification_report(y_test,pre))
import tkinter as tk

def show_message():
    pre = 1  # Example prediction, replace with model.predict([entry.get()])[0]
    if pre == 1:
        message.set("Thank you! We will continue to strive for excellence.")
    else:
        message.set("We sincerely apologize for your experience! We will work to improve.")

# Create the main window
root = tk.Tk()
root.title("Customer Feedback")
root.geometry("700x400")  # Increase window size
root.configure(bg="#f0f0f0")  # Set background color

# Title Label
title_label = tk.Label(root, text="Enter your feedback:", font=("Arial", 16), bg="#f0f0f0")
title_label.pack(pady=10)

# Input Box (Increased height)
entry = tk.Text(root, width=70, height=5, font=("Arial", 14))  # Use Text instead of Entry for longer input
entry.pack(pady=10)

# Message Display (Bold text)
message = tk.StringVar()
label_message = tk.Label(root, textvariable=message, fg="blue", font=("Arial", 16, "bold"), bg="#f0f0f0")
label_message.pack(pady=20)

# Submit Button (Bigger button)
btn = tk.Button(root, text="Submit Feedback", font=("Arial", 14), bg="#0078D7", fg="white", command=show_message)
btn.pack(pady=10)

root.mainloop()