import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

# Preprocessing steps
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# List of movie reviews for training the classifier
reviews = [
    ("Outstanding", "The movie was absolutely fantastic!"),
    ("Very Good", "I really enjoyed watching the movie."),
    ("Very Good", "Emotional"),
    ("Very Good", "charged"),
    ("Very Good", "delighted"),
    ("Very Good", "happy"),
    ("Good", "The movie had some good moments."),
    ("Normal", "The movie was average."),
    ("Bad", "I didn't like the movie at all."),
    ("Very Bad", "The movie was terrible."),
    ("Disgusting", "The movie was disgusting, I couldn't watch it.")
]

# Preprocess the reviews
def preprocess(review):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(review.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return filtered_tokens

# Feature extraction
def get_features(review):
    features = {}
    for word in word_features:
        features[word] = (word in review)
    return features

# Preprocess the reviews
processed_reviews = []
for (category, review) in reviews:
    words = preprocess(review)
    processed_reviews.append((words, category))

# Create a frequency distribution of words
all_words = []
for (words, _) in processed_reviews:
    all_words.extend(words)
word_freq = FreqDist(all_words)

# Select the most frequent words as features
word_features = list(word_freq.keys())[:100]

# Create feature sets
featuresets = [(get_features(words), category) for (words, category) in processed_reviews]

# Train the classifier
classifier = nltk.NaiveBayesClassifier.train(featuresets)

# Bot interaction
def classify_review(review):
    words = preprocess(review)
    features = get_features(words)
    return classifier.classify(features)

# Main loop
while True:
    user_input = input("Enter a movie review (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    category = classify_review(user_input)
    print("Review category:", category)
