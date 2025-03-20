import json
import numpy as np
import random
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load data
words = pickle.load(open("model/words.pkl", "rb"))
classes = pickle.load(open("model/classes.pkl", "rb"))
with open("intents.json", "r") as file:
    intents = json.load(file)

model = load_model("model/chatbot_model.h5")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def get_response(text):
    bow_vector = bow(text, words)
    prediction = model.predict(np.array([bow_vector]))[0]
    max_index = np.argmax(prediction)
    
    if prediction[max_index] > 0.7:
        tag = classes[max_index]
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    
    return "Sorry, I don't understand that."

