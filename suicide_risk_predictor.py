import pandas as pd
import numpy as np
import nltk
import re
import json
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class SuicideRiskPredictor:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.history = []
        self.load_history()
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(processed_tokens)
    
    def load_data(self, filepath):
        """Load and preprocess data from CSV file"""
        print("Loading and preprocessing data...")
        df = pd.read_csv(filepath)
        # Apply preprocessing to text column
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        return df
    
    def train_model(self, df, test_size=0.2, random_state=42):
        """Train and evaluate multiple models, select the best one"""
        print("Training models...")
        
        # Split data
        X = df['processed_text']
        y = df['risk_level']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        # Create pipeline components
        vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.85)
        
        # Define models to try
        models = {
            'Random Forest': RandomForestClassifier(random_state=random_state),
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'SVM': SVC(random_state=random_state, probability=True),
            'Naive Bayes': MultinomialNB()
        }
        
        best_accuracy = 0
        best_model_name = None
        best_model = None
        results = {}
        
        # Try each model
        for name, model in models.items():
            print(f"Training {name}...")
            pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                best_model = pipeline
        
        print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")
        
        # Set the best model and vectorizer
        self.model = best_model
        self.vectorizer = best_model.named_steps['vectorizer']
        
        # Create feature importance visualization for Random Forest
        if best_model_name == 'Random Forest':
            self.visualize_feature_importance(best_model, X_train)
        
        # Evaluate on test set
        self.evaluate_model(best_model, X_test, y_test)
        
        return results
    
    def evaluate_model(self, model, X_test, y_test):
        """Detailed evaluation of the best model"""
        y_pred = model.predict(X_test)
        
        print("\nDetailed Evaluation of Best Model:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
    
    def visualize_feature_importance(self, model, X_train):
        """Visualize feature importance for Random Forest model"""
        # Get vectorizer and classifier from pipeline
        vectorizer = model.named_steps['vectorizer']
        classifier = model.named_steps['classifier']
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Transform training data
        X_train_transformed = vectorizer.transform(X_train)
        
        # Get feature importances
        importances = classifier.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Select top 20 features
        top_n = 20
        top_indices = indices[:top_n]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        # Create bar plot
        plt.figure(figsize=(10, 8))
        plt.title('Top 20 Feature Importances')
        plt.barh(range(top_n), top_importances, align='center')
        plt.yticks(range(top_n), top_features)
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        print("Feature importance visualization saved as 'feature_importance.png'")
    
    def predict_risk(self, text):
        """Predict suicide risk level for new text"""
        if self.model is None:
            print("Error: Model not trained. Please train the model first.")
            return None
        
        # Preprocess the input text
        processed_text = self.preprocess_text(text)
        
        # Make prediction
        prediction = self.model.predict([processed_text])[0]
        probabilities = self.model.predict_proba([processed_text])[0]
        
        # Get the probability for each class
        class_probs = {}
        for i, class_label in enumerate(self.model.classes_):
            class_probs[class_label] = float(probabilities[i])
        
        result = {
            'text': text,
            'processed_text': processed_text,
            'prediction': prediction,
            'probabilities': class_probs,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to history
        self.history.append(result)
        self.save_history()
        
        return result
    
    def save_model(self, model_path='suicide_risk_model.pkl'):
        """Save the trained model to a file"""
        if self.model is None:
            print("Error: No model to save. Please train the model first.")
            return
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='suicide_risk_model.pkl'):
        """Load a trained model from a file"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Extract vectorizer from pipeline
            self.vectorizer = self.model.named_steps['vectorizer']
            print(f"Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            print(f"Error: Model file {model_path} not found.")
            return False
    
    def save_history(self, filepath='risk_assessment_history.json'):
        """Save prediction history to a JSON file"""
        with open(filepath, 'w') as f:
            # Convert history to serializable format
            serializable_history = []
            for item in self.history:
                serializable_item = {k: v for k, v in item.items()}
                # Convert numpy types to Python native types
                if 'probabilities' in serializable_item:
                    serializable_item['probabilities'] = {k: float(v) for k, v in serializable_item['probabilities'].items()}
                serializable_history.append(serializable_item)
            
            json.dump(serializable_history, f, indent=2)
    
    def load_history(self, filepath='risk_assessment_history.json'):
        """Load prediction history from a JSON file"""
        try:
            with open(filepath, 'r') as f:
                self.history = json.load(f)
            print(f"Loaded {len(self.history)} historical predictions")
        except (FileNotFoundError, json.JSONDecodeError):
            self.history = []
            print("No prediction history found or invalid history file")

def main():
    predictor = SuicideRiskPredictor()
    
    # Load and train on the dataset
    df = predictor.load_data('suicide_risk_tex.csv')
    
    # Train the model
    results = predictor.train_model(df)
    
    # Save the trained model
    predictor.save_model()
    
    # Example predictions
    print("\nExample Predictions:")
    example_texts = [
        "I feel so happy today, everything is great!",
        "I don't know if I can go on anymore, everything is pointless",
        "Been feeling down lately but I'm managing",
        "What's the point of trying anymore, no one would miss me"
    ]
    
    for text in example_texts:
        result = predictor.predict_risk(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probabilities: {result['probabilities']}")

if __name__ == "__main__":
    main()