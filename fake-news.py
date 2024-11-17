# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import string

# Download the NLTK stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('news.csv')  # Assuming 'news.csv' is your dataset file

# Data Preprocessing: Removing missing values
df = df.dropna()

# Display the first few rows of the dataset
print("Dataset Sample:")
print(df.head())

# Define the features (X) and the target (y)
X = df['text']  # Assuming 'text' is the column containing the news article text
y = df['label']  # Assuming 'label' is the column containing the real/fake labels

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = vectorizer.transform(X_test)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict on the test data
y_pred = model.predict(X_test_tfidf)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy * 100:.2f}%')

# Function to preprocess user input
def preprocess_input(text):
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

# Take input from the user
user_input = input("Enter the news article: ")

# Preprocess the input
user_input_processed = preprocess_input(user_input)

# Transform the user input using the same vectorizer
user_input_tfidf = vectorizer.transform([user_input_processed])

# Predict whether the news is real or fake
prediction = model.predict(user_input_tfidf)

# Display the result
if prediction == 1:
    print("The news is REAL.")
else:
    print("The news is FAKE.")

# Optionally, display the accuracy again for context
print(f"Model accuracy: {accuracy * 100:.2f}%")
