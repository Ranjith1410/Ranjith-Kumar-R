import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the dataset
file_path = 'spam.csv'  # Replace with your file path
data = pd.read_csv(file_path, encoding='latin-1')

# Clean the dataset
data_cleaned = data[['class', 'message']].dropna()
data_cleaned.rename(columns={'class': 'label', 'message': 'text'}, inplace=True)

# Encode labels: 'ham' -> 0, 'spam' -> 1
label_encoder = LabelEncoder()
data_cleaned['label'] = label_encoder.fit_transform(data_cleaned['label'])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data_cleaned['text'], data_cleaned['label'], test_size=0.3, random_state=42, stratify=data_cleaned['label']
)

# Preprocess text data using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save the trained model and vectorizer
pickle.dump(model, open('spam_classifier.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(report)
