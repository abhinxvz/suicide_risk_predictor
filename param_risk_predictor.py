import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

class SuicideRiskParameterPredictor:
    """
    A suicide risk prediction model based on WHO and clinical research parameters
    rather than text analysis. This model uses direct parameter inputs with
    severity ratings (none, mild, moderate, severe, extreme).
    """
    
    def __init__(self):
        self.model = None
        self.calibrated_model = None
        self.scaler = None
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
        
        # Parameter groups with clinical significance
        self.parameter_groups = {
            'direct_indicators': ['suicidal_thoughts', 'suicide_plan', 'suicide_intent', 'prior_attempts', 'self_harm'],
            'psychological_factors': ['hopelessness', 'depression', 'anxiety'],
            'behavioral_factors': ['isolation', 'impulsivity', 'substance_use', 'sleep_problems'],
            'circumstantial_factors': ['life_stressors', 'access_to_means', 'support_network'],
        }
        
        # Severity levels for each parameter
        self.severity_levels = {
            'none': 0,
            'mild': 1, 
            'moderate': 2,
            'severe': 3,
            'extreme': 4
        }
        
        # Risk level definitions with clinical thresholds
        self.risk_thresholds = {
            'low': 0.25,
            'moderate': 0.50,
            'high': 0.75,
            'critical': 1.0
        }
        
        # Load history if exists
        self.load_history()
    
    def generate_synthetic_data(self, n_samples=2000):
        """
        Generate improved synthetic training data based on expert knowledge of risk factors.
        This version creates more balanced data across risk levels and uses realistic clinical patterns.
        """
        print("Generating balanced synthetic training data...")
        
        # Create empty dataframe with parameter columns
        df = pd.DataFrame(columns=self.parameters + ['risk_level'])
        
        # Risk level labels and desired distribution
        risk_levels = ['low', 'moderate', 'high', 'critical']
        # Aim for a more balanced but still realistic distribution
        risk_distribution = {
            'low': 0.20,      # 20% low risk cases
            'moderate': 0.40, # 40% moderate risk cases
            'high': 0.30,     # 30% high risk cases
            'critical': 0.10  # 10% critical risk cases
        }
        
        # Clinical correlations between parameters (simplified)
        # In reality, these would be derived from research literature
        clinical_correlations = {
            # Key parameters that strongly indicate high/critical risk when elevated together
            'high_risk_cluster': ['suicidal_thoughts', 'suicide_plan', 'suicide_intent', 'hopelessness'],
            # Parameters that are often elevated together
            'depression_cluster': ['depression', 'hopelessness', 'sleep_problems', 'isolation'],
            # Parameters that have protective effects
            'protective_factors': ['support_network']
        }
        
        for risk_level in risk_levels:
            # Calculate samples per risk level based on distribution
            samples_for_level = int(n_samples * risk_distribution[risk_level])
            
            for _ in range(samples_for_level):
                # Generate correlated parameter values based on risk level
                param_values = self._generate_correlated_parameters(risk_level, clinical_correlations)
                
                # Add row to dataframe
                row_data = {**param_values, 'risk_level': risk_level}
                df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
        
        return df
    
    def _generate_correlated_parameters(self, risk_level, clinical_correlations):
        """Generate clinically correlated parameter values based on risk level"""
        param_values = {}
        
        # Base probabilities for each severity level by risk category
        severity_probs = {
            'low': [0.6, 0.3, 0.1, 0.0, 0.0],       # Mostly none/mild
            'moderate': [0.2, 0.4, 0.3, 0.1, 0.0],  # Mostly mild/moderate
            'high': [0.0, 0.2, 0.3, 0.4, 0.1],      # Mostly moderate/severe
            'critical': [0.0, 0.0, 0.2, 0.4, 0.4]   # Mostly severe/extreme
        }
        
        # Set base values for all parameters based on risk level
        for param in self.parameters:
            # Sample severity from distribution appropriate for this risk level
            severity = np.random.choice(5, p=severity_probs[risk_level])
            param_values[param] = severity
        
        # Adjust the high risk cluster based on risk level
        if risk_level in ['high', 'critical']:
            # For high/critical, elevate the high risk cluster parameters
            for param in clinical_correlations['high_risk_cluster']:
                # For critical, make these more likely to be extreme
                if risk_level == 'critical':
                    param_values[param] = np.random.choice([3, 4], p=[0.3, 0.7])  # Severe or extreme
                else:  # high
                    param_values[param] = np.random.choice([2, 3, 4], p=[0.2, 0.6, 0.2])  # Moderate to extreme
        
        # Depression cluster shows correlation
        if param_values['depression'] >= 2:  # If depression is moderate or higher
            # Related symptoms tend to be elevated together
            for param in clinical_correlations['depression_cluster']:
                if param != 'depression':  # Skip depression itself
                    # Elevate but keep within reasonable range
                    param_values[param] = min(param_values[param] + np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3]), 4)
        
        # Protective factors - Support network is inversely related to risk level
        if risk_level == 'low':
            param_values['support_network'] = np.random.choice([2, 3, 4], p=[0.2, 0.4, 0.4])  # Moderate to extreme support
        elif risk_level == 'moderate':
            param_values['support_network'] = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])  # Mild to moderate support
        elif risk_level == 'high':
            param_values['support_network'] = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])  # None to mild support
        else:  # critical
            param_values['support_network'] = np.random.choice([0, 1], p=[0.8, 0.2])  # Mostly no support
        
        # Add clinical edge cases and variations for realism
        self._add_clinical_variations(param_values, risk_level)
        
        return param_values
    
    def _add_clinical_variations(self, param_values, risk_level):
        """Add clinical edge cases and variations for more realistic data"""
        
        # Edge case: High impulsivity with access to means dramatically increases risk
        if param_values['impulsivity'] >= 3 and param_values['access_to_means'] >= 3:
            # Increase suicide intent if impulsivity and access to means are both high
            param_values['suicide_intent'] = min(param_values['suicide_intent'] + np.random.choice([1, 2]), 4)
        
        # Edge case: Prior attempts strongly predict future attempts
        if param_values['prior_attempts'] >= 3:
            # Increase suicide intent and planning if there's history of attempts
            param_values['suicide_intent'] = min(param_values['suicide_intent'] + 1, 4)
            param_values['suicide_plan'] = min(param_values['suicide_plan'] + 1, 4)
        
        # Edge case: Some low risk cases still have moderate depression
        if risk_level == 'low' and np.random.random() < 0.3:
            param_values['depression'] = 2  # Moderate depression
            # But strong support network compensates
            param_values['support_network'] = np.random.choice([3, 4])  # Strong support
        
        # Edge case: Some critical cases don't express suicidal thoughts clearly
        if risk_level == 'critical' and np.random.random() < 0.15:
            # Hidden suicidality - intent and planning high but expression low
            param_values['suicidal_thoughts'] = np.random.choice([0, 1])
            # But other indicators are strongly elevated
            param_values['hopelessness'] = 4
            param_values['isolation'] = 4
        
        # Random variation for realism
        if np.random.random() < 0.05:  # 5% random variation
            # Pick random parameter to adjust
            param = np.random.choice(self.parameters)
            # Adjust randomly up or down by 1 (keeping within bounds)
            adjust = np.random.choice([-1, 1])
            param_values[param] = max(0, min(4, param_values[param] + adjust))
            
    def train_model(self, df=None, test_size=0.2, random_state=42):
        """Train and calibrate the model using the provided or synthetic dataset"""
        if df is None:
            df = self.generate_synthetic_data()
        
        print("Training model...")
        
        # Split data
        X = df[self.parameters]
        y = df['risk_level']
        
        # Initialize scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train Random Forest model with tuned hyperparameters
        base_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced_subsample',  # Handle class imbalance
            random_state=random_state
        )
        
        # Calibrate probabilities for better estimates
        self.model = CalibratedClassifierCV(
            estimator=base_model,
            method='sigmoid',  # Platt scaling
            cv=5,
            n_jobs=-1
        )
        
        # Train the calibrated model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        
        # Generate balanced examples to test probability distribution
        self._validate_probability_distribution()
        
        # Visualize feature importance
        self.visualize_feature_importance(base_model)
        
        return {'accuracy': accuracy}
    
    def _validate_probability_distribution(self):
        """Test probability distribution on synthetic examples from each risk level"""
        print("\nValidating probability distributions across risk levels...")
        
        # Create example cases for each risk level with clear risk patterns
        examples = {
            'low': {param: 0 for param in self.parameters},        # All parameters none
            'moderate': {param: 2 for param in self.parameters},   # All parameters moderate
            'high': {param: 3 for param in self.parameters},       # All parameters severe
            'critical': {param: 4 for param in self.parameters}    # All parameters extreme
        }
        
        # Set support network inverse to risk (high support for low risk)
        examples['low']['support_network'] = 4
        examples['moderate']['support_network'] = 2
        examples['high']['support_network'] = 1
        examples['critical']['support_network'] = 0
        
        # Test each example
        for risk_level, params in examples.items():
            result = self.predict_risk(params)
            print(f"\nExample {risk_level.upper()} risk case:")
            print(f"Predicted: {result['prediction'].upper()}")
            print(f"Probability distribution: {result['probabilities']}")
    
    def visualize_feature_importance(self, base_model=None):
        """Visualize the importance of each parameter in risk prediction"""
        if base_model is None and self.model is None:
            print("Error: Model not trained. Please train the model first.")
            return
        
        if base_model is None:
            # If using calibrated model, extract base estimator
            if hasattr(self.model, 'estimators_'):
                # Get the first calibrated estimator
                base_model = self.model.estimators_[0].base_estimator
            elif hasattr(self.model, 'estimator'):
                # Direct access to the base estimator
                base_model = self.model.estimator
            else:
                print("Error: Cannot extract feature importances from the calibrated model.")
                return
        
        # Get feature importances
        importances = base_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title('Parameter Importance for Suicide Risk Assessment')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [self.parameters[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('parameter_importance.png')
        plt.close()
        
        print("Parameter importance visualization saved as 'parameter_importance.png'")
        
        # Print importance values
        print("\nParameter Importance Ranking:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {self.parameters[idx]}: {importances[idx]:.4f}")
    
    def predict_risk(self, parameter_values):
        """
        Predict suicide risk based on parameter values with improved probability estimates
        
        Args:
            parameter_values: Dictionary with parameter names as keys and severity levels as values
                             (either string levels or numeric values 0-4)
        
        Returns:
            Dictionary with prediction results and calibrated probabilities
        """
        if self.model is None:
            print("Error: Model not trained. Please train the model first.")
            return None
        
        # Convert severity levels from strings to numeric values if needed
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
        features = np.array([[numeric_values[param] for param in self.parameters]])
        
        # Scale features if scaler exists
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Get probability for each class
        class_probs = {}
        for i, class_label in enumerate(self.model.classes_):
            class_probs[class_label] = float(probabilities[i])
        
        # Calculate clinical risk score (0-100) using weighted probabilities
        risk_score = self._calculate_clinical_risk_score(class_probs)
        
        # Calculate risk factors summary
        risk_factor_summary = self._summarize_risk_factors(numeric_values)
        
        result = {
            'parameters': parameter_values,
            'prediction': prediction,
            'risk_score': risk_score,
            'probabilities': class_probs,
            'risk_factors': risk_factor_summary,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to history
        self.history.append(result)
        self.save_history()
        
        return result
    
    def _calculate_clinical_risk_score(self, class_probs):
        """Calculate a clinically meaningful risk score (0-100) from class probabilities"""
        # Weight the risk levels
        risk_weights = {
            'low': 0,
            'moderate': 33,
            'high': 67,
            'critical': 100
        }
        
        # Calculate weighted average
        risk_score = sum(class_probs.get(level, 0) * weight for level, weight in risk_weights.items())
        
        return risk_score
    
    def _summarize_risk_factors(self, numeric_values):
        """Summarize key risk factors from the parameter values"""
        summary = {
            'highest_risk_factors': [],
            'protective_factors': [],
            'warning_signs': []
        }
        
        # Find highest risk factors (parameters with severe/extreme values)
        for param, value in numeric_values.items():
            if value >= 3:  # Severe or extreme
                summary['highest_risk_factors'].append(param)
            elif value <= 1 and param == 'support_network':  # Low support network is a risk
                summary['warning_signs'].append('limited_support')
            elif value >= 4 and param == 'support_network':  # High support is protective
                summary['protective_factors'].append('strong_support')
        
        # Add specific clinical warning signs
        if numeric_values['prior_attempts'] >= 2 and numeric_values['suicide_intent'] >= 2:
            summary['warning_signs'].append('history_with_intent')
        
        if numeric_values['impulsivity'] >= 3 and numeric_values['access_to_means'] >= 2:
            summary['warning_signs'].append('impulsivity_with_means')
            
        if all(numeric_values[p] >= 2 for p in ['hopelessness', 'depression', 'isolation']):
            summary['warning_signs'].append('depression_triad')
        
        return summary
    
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
        
    def save_model(self, model_path='suicide_risk_parameter_model.pkl'):
        """Save the trained model to a file"""
        if self.model is None:
            print("Error: No model to save. Please train the model first.")
            return
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'parameters': self.parameters
            }, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='suicide_risk_parameter_model.pkl'):
        """Load a trained model from a file"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data.get('scaler')
                loaded_params = model_data.get('parameters')
                
                # Verify parameters match
                if loaded_params and loaded_params != self.parameters:
                    print("Warning: Loaded model uses different parameters. Using model's parameters.")
                    self.parameters = loaded_params
                    
            print(f"Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            print(f"Error: Model file {model_path} not found.")
            return False
        except (pickle.UnpicklingError, KeyError):
            print(f"Error: Invalid model file format.")
            return False
    
    def save_history(self, filepath='risk_assessment_param_history.json'):
        """Save prediction history to a JSON file"""
        with open(filepath, 'w') as f:
            # Convert history to serializable format
            serializable_history = []
            for item in self.history:
                serializable_item = {k: v for k, v in item.items()}
                # Convert numpy types to Python native types if present
                if 'probabilities' in serializable_item:
                    serializable_item['probabilities'] = {k: float(v) for k, v in serializable_item['probabilities'].items()}
                serializable_history.append(serializable_item)
            
            json.dump(serializable_history, f, indent=2)
    
    def load_history(self, filepath='risk_assessment_param_history.json'):
        """Load prediction history from a JSON file"""
        try:
            with open(filepath, 'r') as f:
                self.history = json.load(f)
            print(f"Loaded {len(self.history)} historical parameter-based assessments")
        except (FileNotFoundError, json.JSONDecodeError):
            self.history = []
            print("No parameter assessment history found or invalid history file")

def main():
    predictor = SuicideRiskParameterPredictor()
    
    # Train the model on synthetic data
    predictor.train_model()
    
    # Save the trained model
    predictor.save_model()
    
    # Example parameters for prediction
    print("\nExample Predictions:")
    
    example_params = [
        # Low risk example
        {
            'suicidal_thoughts': 'none',
            'suicide_plan': 'none',
            'suicide_intent': 'none',
            'hopelessness': 'mild',
            'depression': 'mild',
            'isolation': 'mild',
            'impulsivity': 'none',
            'substance_use': 'none',
            'prior_attempts': 'none',
            'self_harm': 'none',
            'anxiety': 'mild',
            'life_stressors': 'moderate',
            'access_to_means': 'none',
            'sleep_problems': 'mild',
            'support_network': 'moderate'
        },
        # High risk example
        {
            'suicidal_thoughts': 'severe',
            'suicide_plan': 'moderate',
            'suicide_intent': 'moderate',
            'hopelessness': 'severe',
            'depression': 'severe',
            'isolation': 'severe',
            'impulsivity': 'moderate',
            'substance_use': 'moderate',
            'prior_attempts': 'mild',
            'self_harm': 'moderate',
            'anxiety': 'moderate',
            'life_stressors': 'severe',
            'access_to_means': 'severe',
            'sleep_problems': 'moderate',
            'support_network': 'mild'
        }
    ]
    
    for params in example_params:
        result = predictor.predict_risk(params)
        print(f"\nPredicted Risk Level: {result['prediction'].upper()}")
        print(f"Risk Score (0-100): {result['risk_score']:.1f}")
        print(f"Probability Breakdown: {result['probabilities']}")
        print(f"Key Risk Factors: {result['risk_factors']['highest_risk_factors']}")

if __name__ == "__main__":
    main()
