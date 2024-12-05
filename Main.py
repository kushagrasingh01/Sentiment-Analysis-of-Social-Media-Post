import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import requests
from io import StringIO, BytesIO
from wordcloud import WordCloud
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pytesseract
from PIL import Image
import cv2
from fer import FER

nltk.download('stopwords')

# Initialize the FER detector
detector = FER()

# Function to load data from various formats
def load_data(input_source):
    if input_source.startswith('http://') or input_source.startswith('https://'):
        response = requests.get(input_source)
        content = response.text
        if input_source.endswith('.csv'):
            df = pd.read_csv(StringIO(content))
        elif input_source.endswith('.txt'):
            data = content.splitlines()
            df = pd.DataFrame(data, columns=['text'])
            df['label'] = np.nan  # Assign NaN to labels, user needs to label data later
        else:
            raise ValueError("Unsupported file format from URL. Please provide a .csv or .txt file.")
    else:
        ext = os.path.splitext(input_source)[1]
        if ext == '.csv':
            df = pd.read_csv(input_source)
        elif ext == '.txt':
            with open(input_source, 'r', encoding='utf-8') as file:
                data = file.readlines()
            df = pd.DataFrame(data, columns=['text'])
            df['label'] = np.nan  # Assign NaN to labels, user needs to label data later
        elif ext in ['.png', '.jpg', '.jpeg']:
            text = pytesseract.image_to_string(Image.open(input_source))
            data = text.splitlines()
            df = pd.DataFrame(data, columns=['text'])
            df['label'] = np.nan  # Assign NaN to labels, user needs to label data later
        else:
            raise ValueError("Unsupported file format. Please provide a .csv, .txt, or image file (.png, .jpg, .jpeg).")
    return df

# Function to clean text data
def message_cleaning(message):
    tweet_punc_removed = [char for char in message if char not in string.punctuation]
    tweet_punc_removed_join = ''.join(tweet_punc_removed)
    tweet_punc_removed_join_clean = [word for word in tweet_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return tweet_punc_removed_join_clean

# Function to perform sentiment analysis on an image from a URL
def analyze_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        img_data = response.content
        img = Image.open(BytesIO(img_data))
        img_rgb = img.convert('RGB')
        img_rgb = np.array(img_rgb)
        result = detector.detect_emotions(img_rgb)

        if len(result) == 0:
            print("No faces detected.")
            return

        plt.imshow(img_rgb)
        for face in result:
            (x, y, w, h) = face["box"]
            emotion = face["emotions"]
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
            dominant_emotion = max(emotion, key=emotion.get)
            plt.text(x, y, dominant_emotion, fontsize=12, color='white', backgroundcolor='blue')
        plt.axis('off')
        plt.show()

        for face in result:
            emotions = face["emotions"]
            print("Emotions detected: ", emotions)
    except Exception as e:
        print(f"Error processing image from URL: {e}")

# Function to perform sentiment analysis on an image from a file
def analyze_image_from_file(image_path):
    try:
        img = Image.open(image_path)
        img_rgb = img.convert('RGB')
        img_rgb = np.array(img_rgb)
        result = detector.detect_emotions(img_rgb)

        if len(result) == 0:
            print("No faces detected.")
            return

        plt.imshow(img_rgb)
        for face in result:
            (x, y, w, h) = face["box"]
            emotion = face["emotions"]
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
            dominant_emotion = max(emotion, key=emotion.get)
            plt.text(x, y, dominant_emotion, fontsize=12, color='white', backgroundcolor='blue')
        plt.axis('off')
        plt.show()

        for face in result:
            emotions = face["emotions"]
            print("Emotions detected: ", emotions)
    except Exception as e:
        print(f"Error processing image from file: {e}")

# Function to process text data and perform sentiment analysis
def process_text_data(input_source):
    tweets_df = load_data(input_source)

    if 'text' not in tweets_df.columns or 'label' not in tweets_df.columns:
        print("Please provide the column names for the text data and labels.")
        text_column = input("Enter the column name for the text data: ")
        label_column = input("Enter the column name for the labels: ")
        tweets_df = tweets_df.rename(columns={text_column: 'text', label_column: 'label'})

    if 'label' not in tweets_df.columns or 'text' not in tweets_df.columns:
        print("The dataset does not have the required 'text' and 'label' columns.")
        return

    print(tweets_df.info())
    print(tweets_df.describe())

    if 'id' in tweets_df.columns:
        tweets_df = tweets_df.drop(['id'], axis=1)

    # Prepare the text data for training
    tweets_df_clean = tweets_df['text'].apply(message_cleaning)
    vectorizer = CountVectorizer(analyzer=lambda x: x, dtype=np.uint8)
    tweets_countvectorizer = vectorizer.fit_transform(tweets_df_clean)
    X = pd.DataFrame(tweets_countvectorizer.toarray())
    y = tweets_df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the Naive Bayes classifier
    NB_classifier = MultinomialNB()
    NB_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_predict_test = NB_classifier.predict(X_test)

    # Get the unique labels in the test set
    unique_labels = np.unique(y_test)

    # Print the classification report with the appropriate labels
    print(classification_report(y_test, y_predict_test, target_names=['Negative', 'Positive', 'Neutral'], labels=unique_labels))

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_predict_test, labels=unique_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive', 'Neutral'], yticklabels=['Negative', 'Positive', 'Neutral'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Show example predictions
    predictions = NB_classifier.predict(X_test)
    for text, prediction in zip(tweets_df.iloc[X_test.index]['text'], predictions):
        sentiment = ['Negative', 'Positive', 'Neutral'][prediction]
        print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")

# Function to prompt the user for input
def get_user_input():
    input_type = input("Enter 'file' to provide an image file path or text file path, or 'url' to provide an image URL or text URL: ").strip().lower()
    if input_type == 'file':
        file_path = input("Enter the file path: ").strip()
        if os.path.isfile(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg']:
                analyze_image_from_file(file_path)
            elif ext in ['.csv', '.txt']:
                process_text_data(file_path)
            else:
                print("Unsupported file format. Please provide a valid image (.png, .jpg, .jpeg) or text file (.csv, .txt).")
        else:
            print("The file does not exist. Please provide a valid file path.")
    elif input_type == 'url':
        url = input("Enter the URL: ").strip()
        if url.startswith('http://') or url.startswith('https://'):
            ext = os.path.splitext(url)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg']:
                analyze_image_from_url(url)
            elif ext in ['.csv', '.txt']:
                process_text_data(url)
            else:
                print("Unsupported file format. Please provide a valid image (.png, .jpg, .jpeg) or text URL (.csv, .txt).")
        else:
            print("Invalid URL. Please provide a valid URL starting with 'http://' or 'https://'.")
    else:
        print("Invalid input. Please enter 'file' or 'url'.")
        get_user_input()

# Get user input and perform sentiment analysis
get_user_input()
