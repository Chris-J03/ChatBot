#######################################################
#  Initialise AIML agent
import aiml
import wikipedia
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import requests
import random
from nltk.sem.logic import ApplicationExpression
read_expr = Expression.fromstring

# Load the CNN model
model = load_model('cod_character_classifier.h5')
best_model = load_model('best_cod_character_classifier.h5')
#model.summary()

# Define character labels as per CNN model classes
class_labels = ["Captain Price", "Woods", "Neither"]
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Initialise NLTK Inference
read_expr = Expression.fromstring

# Initialise Knowledgebase. 
kb = []
data = pd.read_csv('logical-kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]

# Create a Kernel object. 
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

# Load Q/A CSV file
qa_df = pd.read_csv('QA_Bank.csv', header=None, names=["question", "answer"])

# Vectorize the questions using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(qa_df['question'])

# News API configuration
NEWS_API_KEY = "8c84a50bb299414094095d9da191bd6f"

# Functions --

# Check if a new expression contradicts the knowledge base
def is_contradiction(expr, kb):
    try:
        # First check direct negation
        neg_expr = read_expr(f"~({str(expr)})")
        if ResolutionProver().prove(neg_expr, kb):
            return True
        
        # Check for mutual exclusions only if this is a predicate application
        if isinstance(expr, ApplicationExpression):
            # Get the predicate name safely
            pred_name = str(expr.function).lower()
            obj = str(expr.args[0])
            
            # Define mutually exclusive categories
            exclusive_map = {
                'game': ['developer', 'franchise'],
                'developer': ['game', 'franchise'], 
                'franchise': ['game', 'developer']
            }
            
            # Check if this is one of our tracked categories
            for category, exclusions in exclusive_map.items():
                if category in pred_name:
                    for other_cat in exclusions:
                        other_expr = read_expr(f"{other_cat.capitalize()}({obj})")
                        if ResolutionProver().prove(other_expr, kb):
                            return True
        
        return False
    except Exception as e:
        print(f"Contradiction check warning: {str(e)}")
        return False

# Find the best matching question from the Q/A CSV using cosine similarity
def find_best_match(user_query):
    # Vectorize the user's query
    user_tfidf = vectorizer.transform([user_query])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    # Get the index of the best match
    best_match_index = np.argmax(cosine_similarities)
    if cosine_similarities[best_match_index] > 0.3: # Must meet threshold
        return qa_df['answer'].iloc[best_match_index]
    return "I'm sorry, I don't have an answer for that."

def select_image():
    # Initialize Tkinter
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("All Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                   ("PNG Files", "*.png"),
                   ("JPEG Files", "*.jpg *.jpeg"),
                   ("Bitmap Files", "*.bmp"),
                   ("TIFF Files", "*.tiff"),
                   ("All Files", "*.*")]
    ) 
    if file_path:  # If a file is selected
        print("Selected file:", file_path)
        return file_path  # Return the file path

    return None

# Load and resize users image to match CNN training
def load_and_process_image(image_path):
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)  # Decode as JPEG
        img = tf.image.resize(img, [64, 64])  # Resize to 128x128
        img = img / 255.0  # Normalize to [0, 1]
        img = tf.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        print(f"Skipping invalid image: {image_path} due to {e}")
        return None

# Display relevant news using API
def get_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    
    # Format the news articles
    news_results = []
    for article in articles[:5]:  # Limit to 5 articles
        title = article.get("title", "No title")
        source = article.get("source", {}).get("name", "Unknown source")
        url = article.get("url", "#")
        news_results.append(f"{title} (Source: {source}) - {url}")
    return news_results


# Hangman functions ----------------

# Choose word from list
def choose_word():
    words = [
    "ModernWarfare", "BlackOps", "Price", "Ghost", "Nuketown", "Zombies", 
    "Verdansk", "M4", "AK47", "Rust", "Shipment", "Killstreak", 
    "Scorestreak", "Operator", "Warzone", "Prestige", "Camo"
    ]
    return random.choice(words)

# Display blank word
def display_word(word, guessed_letters):
    return ' '.join([letter if letter.lower() in guessed_letters else '_' for letter in word])

# Main game
def hangman():
    # Set up game
    word = choose_word()
    guessed_letters = set()
    attempts = 6 

    # Start
    print("Welcome to Hangman!")
    print(display_word(word, guessed_letters))
    
    # Run game
    while attempts > 0:
        guess = input("Guess a letter: ").lower()

        if len(guess) > 1:
            print("Input must only be one letter!")
            continue
        
        if guess in guessed_letters:
            print("You've already guessed that letter.")
            continue
    
        guessed_letters.add(guess)
    
        if guess not in word.lower() :
            attempts -= 1
            print(f"Wrong guess! You have {attempts} attempts left.")
            if attempts == 0:
                print(f"Game over! The word was '{word}'.")
                break
        else:
            print("Good guess!")
    
        current_display = display_word(word, guessed_letters)
        print(current_display)
    
        if "_" not in current_display:
            print("Congratulations! You've guessed the word!")
            break
    ''
# Welcome user
print("Welcome to this chat bot. Please feel free to ask questions from me!")

while True:     
    # Get user input
    try:
        userInput = input("> ").upper()  # Convert to uppercase to match AIML patterns
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break
    # Check if user input is empty
    if not userInput.strip():   
        print("Please enter a command or question.")
        continue
    
    # Query AIML
    aiml_response = kern.respond(userInput)
    
    # First check if we got a direct AIML response (not a command)
    if aiml_response and not aiml_response.startswith("#"):
        print(aiml_response)
        continue
        
    # Then handle commands (responses starting with #)
    if aiml_response.startswith("#"):  
        params = aiml_response[1:].split('$')  
        cmd = int(params[0])  

    # Check if AIML response is a command
    if aiml_response.startswith("#"):  
        params = aiml_response[1:].split('$')  
        cmd = int(params[0])  
        

        if cmd == 0:
            print(params[1])
            break
        
        elif cmd == 1:
            # Try fetching a Wikipedia summary
            try:
                wSummary = wikipedia.summary(params[1], sentences=3, auto_suggest=False)
                print(wSummary)
            except:
                print("Sorry, I couldn't find information on that.")
        
        # Handle news command
        elif cmd == 2:  # News command
            query = params[1]  # Extract the query from the AIML response
            news_results = get_news(query)
            if news_results:
                print("Here are some news articles:")
                for article in news_results:
                    print(f"- {article}")
            else:
                print("Sorry, I couldn't find any news articles.")
                
        # Expanding bots knowledge
        elif cmd == 31: # if input pattern is "I know that * is *"
            try:
                object, subject = params[1].split(' is ')
                object = "_".join([word.capitalize() for word in object.strip().split()])
                subject = subject.strip().capitalize()
                
                expr = read_expr(f"{subject}({object})")
                
                # First check if this exact fact already exists in KB
                knowledge_exists = any(expr == kb_expr for kb_expr in kb)
                
                if knowledge_exists:
                    print(f"This knowledge aligns with what I already know.")
                elif is_contradiction(expr, kb):
                    # Find what type it actually is
                    actual_types = []
                    for category in ['Game', 'Developer', 'Franchise']:
                        if ResolutionProver().prove(read_expr(f"{category}({object})"), kb):
                            actual_types.append(category)
                    
                    if actual_types:
                        print(f"Conflict! {object} is already: {', '.join(actual_types)}")
                    else:
                        print("This contradicts general knowledge rules")
                else:
                    kb.append(expr)
                    print(f"Learned: {object} is {subject}")
                    
            except Exception as e:
                print(f"Input error: {str(e)}")
                            
        # Utilising knowledge base
        elif cmd == 32:  # If the input pattern is "check that * is *"
            object, subject = params[1].split(' is ')  # Split the input into object and subject
            
            # Handle multiple objects by capitalizing each word, joining with an underscore
            object = "_".join([word.capitalize() for word in object.strip().split()]) 
            subject = subject.strip().capitalize()  # Capitalize the subject (e.g., Developer)
            
            try:
                # Create the expression to check if object is of subject type
                expr = read_expr(f"{subject}({object})")
                
                # Query the KB
                answer = ResolutionProver().prove(expr, kb, verbose=False) # Toggle verbose for testing
                
                # Check if we should verify the opposite 
                if not answer and subject in ['Game', 'Developer', 'Franchise']:
                    # Check if the object exists in any of the mutually exclusive categories
                    other_categories = ['Game', 'Developer', 'Franchise']
                    other_categories.remove(subject)
                    
                    for category in other_categories:
                        if ResolutionProver().prove(read_expr(f"{category}({object})"), kb):
                            print(f'Incorrect. {object} is actually a {category}.')
                            break
                    else:
                        print(f'It may not be true that {object} is a {subject}.')
                else:
                    print('Correct.' if answer else f'It may not be true that {object} is a {subject}.')
                    
            except Exception as e:
                print(f"Error processing input: {e}")
        
        # Image recognition command
        elif cmd == 33:
            try:
                character_name = params[1]  # Extract character name from AIML
        
                # Prompt user to select an image
                image_path = select_image()
                
                if image_path is None:
                    print("Error: No image selected.")
                else:
                    # Load and preprocess the image using function
                    img = load_and_process_image(image_path)
                    if img is None:
                        print("Error: Invalid image file.")
                    else:
                        # Make prediction with original model
                        prediction = model.predict(img)
                        predicted_class = np.argmax(prediction)
                        print("Prediction probabilities:", prediction)
                        
                        best_prediction = best_model.predict(img)
                        best_predicted_class = np.argmax(best_prediction)
                        print("Best prediction probabilities:", best_prediction)
                        
                        if prediction[0][predicted_class] < 0.6 or best_prediction[0][best_predicted_class] < 0.6:  # Threshold for prediction
                            print("I'm not confident in my prediction.")
                            #best_predicted_character = class_labels[best_predicted_class]
                            #print (f"The hyper parameter model identified this as: {best_predicted_character}")
                        else:
                            predicted_character = class_labels[predicted_class]
                            best_predicted_character = class_labels[best_predicted_class]
                            print(f"The model identified this as: {predicted_character}")
                            print (f"The hyper parameter model identified this as: {best_predicted_character}")

            except Exception as e:
                print(f"An error occurred: {e}")

        # Hangman option
        elif cmd == 34:
            try:
                hangman()
            except Exception as e:
                print(f"An error occurred: {e}")
                
        # If no other function fits the input   
        elif cmd in [99, 100]:
            # AIML fallback signal
            fallback_response = find_best_match(userInput)
            print(fallback_response if fallback_response else "Sorry, I do not know that.")
        else:
            print("Command not recognized.")
    # Use the QA bank if AIML does not provide a meaningful response   
    else:
        fallback_response = find_best_match(userInput)
        if fallback_response:
            print(fallback_response)
        else:
            print("I'm not sure how to respond to that.")