import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import string
# Simulated dataset (replace with your CSV using pd.read_csv)
data = {
    'Review': [
        'I love this product! It works great!',
        'Terrible experience. It broke after one use.',
        'Very satisfied with the purchase.',
        'Not worth the money.',
        'Absolutely fantastic! Highly recommended.',
        'Worst product ever. Do not buy.',
        'Okay for the price.',
        'Horrible. Waste of time and money.',
        'Good quality and fast shipping.',
        'Disappointed with the performance.'
    ],
    'Sentiment': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive',
                  'Negative', 'Neutral', 'Negative', 'Positive', 'Negative']
}

df = pd.DataFrame(data)
df.head()
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    return text

df['Cleaned_Review'] = df['Review'].apply(preprocess_text)
df.head()
# Convert Neutral to Negative or drop for binary classification
df = df[df['Sentiment'] != 'Neutral']

df['Sentiment'] = df['Sentiment'].map({'Positive': 1, 'Negative': 0})
tfidf = TfidfVectorizer(max_features=500)
X = tfidf.fit_transform(df['Cleaned_Review']).toarray()
y = df['Sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Optional: Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

