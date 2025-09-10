import random
import spacy
import webbrowser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Training data (questions → intents)
training_sentences = [
    "hello", "hi", "good morning",
    "how are you", "what's up",
    "what is your name", "who are you",
    "how old are you", "your age",
    "what do you do", "your hobby",
    "bye", "goodbye", "see you"
]

training_labels = [
    "greeting", "greeting", "greeting",
    "feeling", "feeling",
    "name", "name",
    "age", "age",
    "hobby", "hobby",
    "goodbye", "goodbye", "goodbye"
]

# Responses for each intent
responses = {
    "greeting": ["Hello! How are you today?", "Hi there! Glad to chat with you."],
    "feeling": ["I'm doing great! Thanks for asking. How about you?", "All good here, what about you?"],
    "name": ["I'm your friendly chatbot 🤖", "You can call me ChatBuddy!"],
    "age": ["I'm timeless 😎", "Age is just a number, but I'm quite new!"],
    "hobby": ["I love chatting with people like you!", "My hobby is helping humans with information."],
    "goodbye": ["Goodbye! Have a nice day 😊", "See you later! Take care."]
}

# Convert text to features (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, training_labels)

def chatbot_response(user_input):
    # Transform input
    X_test = vectorizer.transform([user_input])
    predicted_label = clf.predict(X_test)[0]
    confidence = max(clf.predict_proba(X_test)[0])

    # If model is confident → give a response
    if confidence > 0.3:
        return random.choice(responses[predicted_label])
    else:
        # Fallback → Google search
        print("🤖 Chatbot: Sorry, I didn’t understand that. Mai aapko Google pe leke jaata hoon...")
        webbrowser.open(f"https://www.google.com/search?q={user_input}")
        return "🔎 Opening Google search for your query..."

# Chat loop
print("🤖 Chatbot: Hi! I'm your chatbot (NLP + ML). Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("🤖 Chatbot:", chatbot_response(user_input))
        break
    print("🤖 Chatbot:", chatbot_response(user_input))
