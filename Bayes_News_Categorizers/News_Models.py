"""
    Andre Pont - 23164034    

    Naive Bayes Models created with much help from the lab 7 resources, 
    every model is sligtly modified just so it fits the type of data used.
    Saving and loading functions are also implemented from the lab 7 resources.

"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import pickle
import os
import matplotlib.pyplot as plt

# Get the directory where this script is located every time, (it won't work without it since it will not find the file in the other folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'parsed_news_data.csv')
data = pd.read_csv(csv_path)

#Function to simply print the dataset from the ontology
def print_dataset_shape():
    print("Dataset shape:", data.shape)
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nDataset info:")
    print(data.info())
    print("\nCategory distribution:")
    print(data['category'].value_counts())

def short_description_classifier():
    #Prepare text data and target
    X_text = data['short_description']  # Text data for classification
    y = data['category']  # Target labels

    #Split data into training and testing sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.3, random_state=42, stratify=y)

    #Vectorize the text data
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    #Initialize and train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    #Make predictions
    y_pred = model.predict(X_test_vec)

    #Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nShort Descriptions Model Results:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    #Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(data['category'].unique()), 
                yticklabels=sorted(data['category'].unique()))
    plt.title('Confusion Matrix - Short Descriptions Classifier')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return accuracy, model, vectorizer

def headline_classifier():
    #Prepare text data and target
    X_text = data['headline']  # Text data for classification
    y = data['category']  # Target labels

    #Split data into training and testing sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.3, random_state=42, stratify=y)

    #Vectorize the text data
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    #Initialize and train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    #Make predictions
    y_pred = model.predict(X_test_vec)

    #Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\Headlines Model Results:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    #Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(data['category'].unique()), 
                yticklabels=sorted(data['category'].unique()))
    plt.title('Confusion Matrix - Headline Only Classifier')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return accuracy, model, vectorizer

def combined_classifier():
    data['combined_text'] = data['headline'] + ' ' + data['short_description']  # Combine headline and description
    X_text = data['combined_text']  # Combined text data for classification
    y = data['category']  # Target labels

    #Split data into training and testing sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.3, random_state=42, stratify=y)

    #Vectorize the text data
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    #Initialize and train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    #Make predictions
    y_pred = model.predict(X_test_vec)

    #Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nCombined (Headlines + Descriptions) Model Results:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    #Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(data['category'].unique()), 
                yticklabels=sorted(data['category'].unique()))
    plt.title('Confusion Matrix - Combined (Headlines + Descriptions) Classifier')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return accuracy, model, vectorizer


#Functions to save and load models to prevent training the model each time it is needed to deliver a prediction
def save_model(model, vectorizer, model_filename, vectorizer_filename):
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(vectorizer_filename, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

def load_model(model_filename, vectorizer_filename):
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_filename, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

