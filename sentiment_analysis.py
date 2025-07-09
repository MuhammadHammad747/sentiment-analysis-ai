# sentiment_analysis.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample movie review dataset
data = {
    'review': [
        'I loved the movie, it was fantastic!',
        'Worst film I have ever seen.',
        'It was okay, not great but not terrible.',
        'Absolutely wonderful movie with great acting.',
        'Terrible plot, bad acting.',
        'I enjoyed the movie a lot.',
        'I hate this film.',
        'The film was boring and too long.',
        'Amazing direction and story.',
        'Awful, I walked out halfway.'
    ],
    'sentiment': [
        'positive',
        'negative',
        'neutral',
        'positive',
        'negative',
        'positive',
        'negative',
        'negative',
        'positive',
        'negative'
    ]
}

df = pd.DataFrame(data)

# Convert labels to binary (positive=1, negative/neutral=0)
df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Prepare data
X = df['review']
y = df['label']

# Convert text to features
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Test custom review
test_review = ["This movie was awesome and I loved it"]
test_vector = vectorizer.transform(test_review)
pred = model.predict(test_vector)
print("Prediction:", "Positive" if pred[0] == 1 else "Negative or Neutral")
