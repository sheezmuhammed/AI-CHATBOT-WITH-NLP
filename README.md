# AI-CHATBOT-WITH-NLP

**COMPANY**:CODTECH IT SOLUTIONS
**NAME**:SHEEZ MUHAMMED C
**INTERN ID**:CT08NAH
**DOMAIN**:PYTHON
**BATCH DURATION**:January 15th/2025 to February 15th/2025

**DESCRIPTION**

AI Chatbot Using Natural Language Processing (NLP)

Introduction

This project is a simple AI chatbot that uses Natural Language Processing (NLP) to understand and respond to user queries. It leverages NLTK (Natural Language Toolkit) and SpaCy for text processing, tokenization, and stopword removal. The chatbot operates based on keyword matching and provides predefined responses to common questions.

This chatbot serves as a foundational example of how rule-based chatbots work, with the potential for future improvements using machine learning or deep learning techniques.

How the Chatbot Works

1. Preprocessing User Input

Before generating a response, the chatbot processes user input through the following steps:

✅ Lowercasing: Converts all text to lowercase for uniformity.

✅ Tokenization: Splits sentences into individual words using NLTK.

✅ Removing Punctuation: Filters out non-alphanumeric characters.

✅ Stopword Removal: Eliminates common words (e.g., “the”, “is”, “in”) using NLTK’s stopwords list.

2. Recognizing User Queries

The chatbot identifies key phrases from the processed input and matches them with predefined responses stored in a dictionary. Some examples include:
	•	“hello” → Responses like “Hi there!” or “Hello!”
	•	“how are you” → Responses like “I’m good! How about you?”
	•	“your name” → Responses like “You can call me AI Bot!”
	•	“bye” → Responses like “Goodbye!” or “Take care!”

If the chatbot does not find a matching phrase, it provides a default response such as:
“I’m not sure I understand.”

3. Generating Responses

Once the chatbot identifies a keyword, it selects a random response from the associated list using Python’s random.choice() function. This adds variety to the chatbot’s replies, making interactions feel more natural.

4. Running the Chatbot

The chatbot runs in a continuous loop, allowing users to interact until they type "exit", at which point the chatbot says goodbye and stops execution.

Project Setup & Installation

1. Install Dependencies

Before running the chatbot, install the required Python libraries:

pip install nltk spacy
python -m spacy download en_core_web_sm

2. Run the Chatbot

After installation, execute the script using:

python chatbot.py

Code Breakdown

1. Importing Required Libraries

import nltk
import spacy
import random

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

	•	NLTK: Used for tokenization and stopword removal.
	•	SpaCy: Provides additional NLP capabilities.
	•	random: Used to randomly select responses.

2. Downloading Necessary Data

nltk.download("punkt")
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")

	•	Punkt: Required for sentence tokenization.
	•	Stopwords: A list of common words to ignore during processing.
	•	SpaCy Model: Loads an English language model for advanced NLP tasks.

3. Defining Chatbot Responses

responses = {
    "hello": ["Hi there!", "Hello!", "Hey! How can I help you?"],
    "how are you": ["I'm just a bot, but I'm doing well!", "I'm good! How about you?"],
    "your name": ["I'm a chatbot!", "You can call me AI Bot!"],
    "bye": ["Goodbye!", "See you later!", "Take care!"],
    "default": ["I'm not sure I understand.", "Can you rephrase that?", "Sorry, I don't have an answer for that."]
}

This dictionary stores predefined responses for recognized keywords.

4. Processing User Input

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return tokens

	•	Lowercasing the input text.
	•	Tokenizing text into words.
	•	Removing punctuation and stopwords to focus on meaningful words.

5. Generating Responses

def chatbot_response(user_input):
    tokens = preprocess_text(user_input)
    for key in responses.keys():
        if key in user_input:
            return random.choice(responses[key])
    return random.choice(responses["default"])

	•	Checks if a keyword is present in the input.
	•	Selects a random response from the predefined list.
	•	Returns a default response if no match is found.

6. Running the Chatbot in a Loop

def chat():
    print("Chatbot: Hello! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()

	•	Continuously prompts the user for input.
	•	Generates and displays responses based on input.
	•	Ends the conversation when the user types "exit".

Example Interaction

Chatbot: Hello! Type 'exit' to end the conversation.
You: hello
Chatbot: Hi there!

You: what is your name?
Chatbot: You can call me AI Bot!

You: how are you?
Chatbot: I'm just a bot, but I'm doing well!

You: bye
Chatbot: Goodbye!

Future Enhancements

This chatbot is a basic implementation and can be extended with the following improvements:

🔹 Machine Learning: Train a chatbot using NLP and ML techniques.

🔹 Context Awareness: Maintain conversation history for better responses.

🔹 API Integration: Connect with external data sources for dynamic answers.

🔹 Voice Support: Add speech recognition for voice-based interactions.

Conclusion

This chatbot serves as a simple but effective demonstration of rule-based NLP using NLTK and SpaCy. While it relies on predefined responses, it lays the groundwork for more advanced AI-driven chatbot systems.

For future development, integrating machine learning models (e.g., transformers like GPT) can enhance the chatbot’s ability to generate human-like responses.
