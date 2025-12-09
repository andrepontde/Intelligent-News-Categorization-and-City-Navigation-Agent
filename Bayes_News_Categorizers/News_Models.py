import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('parsed_news_data.csv')

print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nDataset info:")
print(data.info())
print("\nCategory distribution:")
print(data['category'].value_counts())

def test_short_description_classifier():
    # Step 3: Prepare text data and target
    X_text = data['short_description']  # Text data for classification
    y = data['category']  # Target labels

    # Step 4: Split data into training and testing sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.3, random_state=42, stratify=y)

    # Step 5: Vectorize the text data
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # Step 6: Initialize and train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Step 7: Make predictions
    y_pred = model.predict(X_test_vec)

    # Step 8: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nShort Descriptions Model Results:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Step 9: Plot confusion matrix
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

def test_headline_classifier():
    # Step 3: Prepare text data and target
    X_text = data['headline']  # Text data for classification
    y = data['category']  # Target labels

    # Step 4: Split data into training and testing sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.3, random_state=42, stratify=y)

    # Step 5: Vectorize the text data
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # Step 6: Initialize and train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Step 7: Make predictions
    y_pred = model.predict(X_test_vec)

    # Step 8: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\Headlines Model Results:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Step 9: Plot confusion matrix
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

def test_combined_classifier():
    data['combined_text'] = data['headline'] + ' ' + data['short_description']  # Combine headline and description
    X_text = data['combined_text']  # Combined text data for classification
    y = data['category']  # Target labels

    # Step 4: Split data into training and testing sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.3, random_state=42, stratify=y)

    # Step 5: Vectorize the text data
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # Step 6: Initialize and train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Step 7: Make predictions
    y_pred = model.predict(X_test_vec)

    # Step 8: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nCombined (Headlines + Descriptions) Model Results:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Step 9: Plot confusion matrix
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
