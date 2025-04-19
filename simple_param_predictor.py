import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class SimpleParameterRiskPredictor:
    """
    A simplified version of the suicide risk prediction model based on WHO parameters
    with improved probability balance and distribution.
    """
    
    def __init__(self):
        self.model = None
        self.history = []
        self.parameters = [
            'suicidal_thoughts',          # Frequency/intensity of suicidal thoughts
            'suicide_plan',               # Presence and detail of suicide plan
            'suicide_intent',             # Intent to act on suicidal thoughts
            'hopelessness',               # Feelings of hopelessness about the future
            'depression',                 # Depressive symptoms severity
            'isolation',                  # Social isolation/withdrawal
            'impulsivity',                # Impulsive behaviors/tendencies
            'substance_use',              # Alcohol/drug use
            'prior_attempts',             # History of suicide attempts
            'self_harm',                  # Non-suicidal self-harm
            'anxiety',                    # Anxiety symptoms
            'life_stressors',             # Recent significant life stressors
            'access_to_means',            # Access to lethal means
            'sleep_problems',             # Sleep disturbances
            'support_network',            # Availability of social support
        ]
        
        # Parameter weight improvements - based on clinical importance
        self.parameter_weights = {
            'suicidal_thoughts': 3.0,
            'suicide_plan': 3.0,
            'suicide_intent': 3.5,
            'hopelessness': 2.0,
            'depression': 1.5,
            'isolation': 1.0,
            'impulsivity': 1.5,
            'substance_use': 1.0,
            'prior_attempts': 3.0,
            'self_harm': 2.0,
            'anxiety': 0.5,
            'life_stressors': 1.0,
            'access_to_means': 2.5,
            'sleep_problems': 0.5,
            'support_network': -1.0  # Negative weight as this is protective
        }
        
        # Severity levels for each parameter
        self.severity_levels = {
            'none': 0,
            'mild': 1, 
            'moderate': 2,
            'severe': 3,
            'extreme': 4
        }
        
        # Load history if exists
        self.load_history()
    
    def generate_better_synthetic_data(self, n_samples=1000):
        """Generate better balanced synthetic training data"""
        print("Generating balanced synthetic training data...")
        
        # Create empty dataframe with parameter columns
        df = pd.DataFrame(columns=self.parameters + ['risk_level'])
        
        # Risk level labels and target distribution
        risk_levels = ['low', 'moderate', 'high', 'critical']
        target_distribution = {
            'low': int(n_samples * 0.25),
            'moderate': int(n_samples * 0.35),
            'high': int(n_samples * 0.25),
            'critical': int(n_samples * 0.15)
        }
        
        for risk_level, count in target_distribution.items():
            for _ in range(count):
                # Generate parameter values specifically for this risk level
                param_values = self._generate_values_for_risk_level(risk_level)
                
                # Add row to dataframe
                row_data = {**param_values, 'risk_level': risk_level}
                df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
                
        return df
    
    def _generate_values_for_risk_level(self, risk_level):
        """Generate realistic parameter values for a specific risk level"""
        param_values = {}
        
        # Base distributions for each risk level
        severity_ranges = {
            'low': (0, 1),       # none to mild
            'moderate': (1, 2),  # mild to moderate
            'high': (2, 3),      # moderate to severe
            'critical': (3, 4)   # severe to extreme
        }
        
        # Generate base values for all parameters based on risk level range
        base_range = severity_ranges[risk_level]
        for param in self.parameters:
            # For support network, invert the values (high support for low risk)
            if param == 'support_network':
                if risk_level == 'low':
                    param_values[param] = np.random.randint(3, 5)  # high support
                elif risk_level == 'moderate':
                    param_values[param] = np.random.randint(2, 4)  # moderate support
                elif risk_level == 'high':
                    param_values[param] = np.random.randint(1, 3)  # low to moderate support
                else:  # critical
                    param_values[param] = np.random.randint(0, 2)  # very low support
            else:
                # For other parameters, use the risk level's severity range
                param_values[param] = np.random.randint(base_range[0], base_range[1] + 1)
        
        # Adjust critical parameters based on risk level
        critical_params = ['suicidal_thoughts', 'suicide_plan', 'suicide_intent']
        if risk_level == 'high' or risk_level == 'critical':
            # For high/critical, ensure these key parameters are elevated
            for param in critical_params:
                if risk_level == 'critical':
                    param_values[param] = np.random.randint(3, 5)  # severe to extreme
                else:  # high
                    param_values[param] = np.random.randint(2, 4)  # moderate to severe
        
        # Add some random variation for realistic data
        self._add_random_variations(param_values)
        
        return param_values
    
    def _add_random_variations(self, param_values):
        """Add realistic random variations to parameter values"""
        # Randomly select 1-3 parameters to adjust
        num_params_to_adjust = np.random.randint(1, 4)
        params_to_adjust = np.random.choice(self.parameters, size=num_params_to_adjust, replace=False)
        
        for param in params_to_adjust:
            # Adjust up or down by 0-1 levels, ensuring we stay within 0-4 range
            adjustment = np.random.choice([-1, 0, 1])
            param_values[param] = max(0, min(4, param_values[param] + adjustment))
    
    def calculate_weighted_risk_score(self, param_values):
        """Calculate weighted risk score based on parameter values"""
        risk_score = 0
        for param, value in param_values.items():
            weight = self.parameter_weights.get(param, 1.0)
            if param == 'support_network':
                # Support network is inversely related to risk (higher support = lower risk)
                risk_score += weight * (4 - value)
            else:
                risk_score += weight * value
        
        # Normalize to 0-1 range for easier interpretation
        max_possible_score = sum(abs(w) * 4 for w in self.parameter_weights.values())
        normalized_score = risk_score / max_possible_score
        
        return normalized_score
    
    def train_model(self, df=None):
        """Train a Random Forest model on balanced data"""
        if df is None:
            df = self.generate_better_synthetic_data()
        
        print("Training model...")
        
        # Split data into features and target
        X = df[self.parameters]
        y = df['risk_level']
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        # Initialize and train Random Forest with class balancing
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced_subsample',
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Verify probability distributions with examples
        self._verify_probability_distributions()
        
        # Visualize feature importance
        self._visualize_feature_importance()
        
        return accuracy
    
    def _verify_probability_distributions(self):
        """Test the model with examples from each risk level to verify probability distributions"""
        print("\nVerifying probability distributions...")
        
        # Create clear test cases for each risk level
        test_cases = {
            'low': self._create_test_case('low'),
            'moderate': self._create_test_case('moderate'),
            'high': self._create_test_case('high'),
            'critical': self._create_test_case('critical')
        }
        
        # Test each case
        for level, case in test_cases.items():
            result = self.predict_risk(case)
            print(f"\nTest case for {level.upper()} risk:")
            print(f"Predicted: {result['prediction']}")
            print(f"Risk Score: {result['risk_score']:.2f}")
            print(f"Probability Distribution: {result['probabilities']}")
    
    def _create_test_case(self, risk_level):
        """Create a clear test case for a specific risk level"""
        case = {}
        
        # Default values based on risk level
        if risk_level == 'low':
            base_value = 0  # none
            support_value = 4  # extreme support
        elif risk_level == 'moderate':
            base_value = 2  # moderate
            support_value = 2  # moderate support
        elif risk_level == 'high':
            base_value = 3  # severe
            support_value = 1  # mild support
        else:  # critical
            base_value = 4  # extreme
            support_value = 0  # no support
        
        # Set all parameters to the base value
        for param in self.parameters:
            if param == 'support_network':
                case[param] = support_value
            else:
                case[param] = base_value
        
        return case
    
    def _visualize_feature_importance(self):
        """Visualize the importance of each parameter"""
        if self.model is None:
            print("Error: Model not trained.")
            return
        
        # Get feature importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title('Parameter Importance for Suicide Risk Assessment')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [self.parameters[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('simple_parameter_importance.png')
        plt.close()
        
        print("Parameter importance visualization saved as 'simple_parameter_importance.png'")
        
        # Also print the importance ranking
        print("\nParameter Importance Ranking:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {self.parameters[idx]}: {importances[idx]:.4f}")
    
    def predict_risk(self, parameter_values):
        """
        Predict suicide risk level with improved probability distribution
        
        Args:
            parameter_values: Dictionary with parameter names as keys and severity values
        
        Returns:
            Dictionary with prediction results and balanced probabilities
        """
        if self.model is None:
            print("Error: Model not trained. Please train the model first.")
            return None
        
        # Convert string severity levels to numeric values if needed
        numeric_values = {}
        for param in self.parameters:
            if param in parameter_values:
                value = parameter_values[param]
                if isinstance(value, str) and value.lower() in self.severity_levels:
                    numeric_values[param] = self.severity_levels[value.lower()]
                else:
                    try:
                        numeric_values[param] = int(value)
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid value '{value}' for parameter '{param}'. Using 0 (none).")
                        numeric_values[param] = 0
            else:
                # If parameter is missing, assume 'none' (0)
                numeric_values[param] = 0
        
        # Create features array in the correct order
        features = [[numeric_values[param] for param in self.parameters]]
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Get probability for each class with a minimum threshold to avoid flat distributions
        min_prob = 0.05  # Minimum probability to assign
        class_probs = {}
        for i, class_label in enumerate(self.model.classes_):
            raw_prob = float(probabilities[i])
            
            # Ensure we have a minimum difference between classes
            if raw_prob < min_prob and raw_prob > 0.01:
                class_probs[class_label] = min_prob
            else:
                class_probs[class_label] = raw_prob
        
        # Normalize probabilities to sum to 1
        total_prob = sum(class_probs.values())
        for k in class_probs:
            class_probs[k] = class_probs[k] / total_prob
        
        # Calculate weighted risk score (0-100)
        norm_score = self.calculate_weighted_risk_score(numeric_values)
        risk_score = norm_score * 100
        
        # Calculate and include key risk factors
        risk_factors = self._identify_key_risk_factors(numeric_values)
        
        result = {
            'parameters': parameter_values,
            'prediction': prediction,
            'risk_score': risk_score,
            'probabilities': class_probs,
            'risk_factors': risk_factors,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to history
        self.history.append(result)
        self.save_history()
        
        return result
    
    def _identify_key_risk_factors(self, param_values):
        """Identify key risk factors from parameter values"""
        risk_factors = {
            'high_risk_factors': [],
            'protective_factors': [],
            'warning_combinations': []
        }
        
        # Identify high risk factors (severe or extreme values)
        for param, value in param_values.items():
            if value >= 3 and param != 'support_network':
                risk_factors['high_risk_factors'].append(param)
            
            if param == 'support_network' and value >= 3:
                risk_factors['protective_factors'].append('strong_social_support')
        
        # Check for dangerous combinations
        if param_values['suicidal_thoughts'] >= 2 and param_values['suicide_plan'] >= 2:
            risk_factors['warning_combinations'].append('thoughts_with_plan')
            
        if param_values['suicide_intent'] >= 2 and param_values['access_to_means'] >= 2:
            risk_factors['warning_combinations'].append('intent_with_means')
            
        if param_values['hopelessness'] >= 3 and param_values['depression'] >= 3:
            risk_factors['warning_combinations'].append('severe_depression_hopelessness')
            
        return risk_factors
    
    def get_parameter_descriptions(self):
        """Return descriptions for each parameter to guide assessment"""
        return {
            'suicidal_thoughts': "Frequency and intensity of thoughts about suicide",
            'suicide_plan': "Presence and specificity of plan for suicide",
            'suicide_intent': "Intention or determination to act on suicidal thoughts",
            'hopelessness': "Feelings of hopelessness or pessimism about the future",
            'depression': "Severity of depressive symptoms",
            'isolation': "Degree of social isolation or withdrawal from others",
            'impulsivity': "Tendency to act on impulse without considering consequences",
            'substance_use': "Severity of alcohol or drug use",
            'prior_attempts': "History and recency of previous suicide attempts",
            'self_harm': "Non-suicidal self-injury behaviors",
            'anxiety': "Severity of anxiety symptoms",
            'life_stressors': "Recent significant stressful life events",
            'access_to_means': "Access to lethal means (firearms, medications, etc.)",
            'sleep_problems': "Severity of sleep disturbances",
            'support_network': "Availability of social support network",
        }
    
    def get_risk_level_descriptions(self):
        """Return descriptions for each risk level"""
        return {
            'low': "Low risk - No immediate danger, but monitoring may be needed.",
            'moderate': "Moderate risk - Regular monitoring and support recommended.",
            'high': "High risk - Immediate intervention and safety planning needed.",
            'critical': "Critical risk - Urgent psychiatric evaluation/hospitalization needed."
        }
    
    def save_model(self, model_path='simple_risk_model.pkl'):
        """Save the trained model to a file"""
        if self.model is None:
            print("Error: No model to save. Please train the model first.")
            return
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='simple_risk_model.pkl'):
        """Load a trained model from a file"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            print(f"Error: Model file {model_path} not found.")
            return False
    
    def save_history(self, filepath='simple_risk_history.json'):
        """Save prediction history to a JSON file"""
        with open(filepath, 'w') as f:
            # Convert history to serializable format
            serializable_history = []
            for item in self.history:
                serializable_item = {k: v for k, v in item.items()}
                # Convert any numpy types to Python native types
                if 'probabilities' in serializable_item:
                    serializable_item['probabilities'] = {k: float(v) for k, v in serializable_item['probabilities'].items()}
                serializable_history.append(serializable_item)
            
            json.dump(serializable_history, f, indent=2)
    
    def load_history(self, filepath='simple_risk_history.json'):
        """Load prediction history from a JSON file"""
        try:
            with open(filepath, 'r') as f:
                self.history = json.load(f)
            print(f"Loaded {len(self.history)} historical assessments")
        except (FileNotFoundError, json.JSONDecodeError):
            self.history = []
            print("No assessment history found or invalid history file")

def main():
    # Create and train the model
    predictor = SimpleParameterRiskPredictor()
    predictor.train_model()
    predictor.save_model()
    
    # Example with improved probability distribution
    print("\nTesting with mixed parameters case:")
    mixed_case = {
        'suicidal_thoughts': 'moderate',
        'suicide_plan': 'mild',
        'suicide_intent': 'moderate',
        'hopelessness': 'moderate', 
        'depression': 'severe',
        'isolation': 'moderate',
        'impulsivity': 'mild',
        'substance_use': 'mild',
        'prior_attempts': 'none',
        'self_harm': 'mild',
        'anxiety': 'severe',
        'life_stressors': 'moderate',
        'access_to_means': 'mild',
        'sleep_problems': 'moderate',
        'support_network': 'moderate'
    }
    
    result = predictor.predict_risk(mixed_case)
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Risk Score: {result['risk_score']:.2f}/100")
    print("Probability Distribution:")
    for level, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {level.upper()}: {prob*100:.1f}%")
    
    print(f"\nHigh Risk Factors: {result['risk_factors']['high_risk_factors']}")
    print(f"Warning Combinations: {result['risk_factors']['warning_combinations']}")

if __name__ == "__main__":
    main()
