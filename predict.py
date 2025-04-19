import argparse
from suicide_risk_predictor import SuicideRiskPredictor

def main():
    parser = argparse.ArgumentParser(description='Predict suicide risk from text')
    parser.add_argument('--text', '-t', type=str, help='Text to analyze for suicide risk')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()
    
    # Initialize predictor and load model
    predictor = SuicideRiskPredictor()
    model_loaded = predictor.load_model()
    
    if not model_loaded:
        print("Model not found. Please run suicide_risk_predictor.py first to train the model.")
        return
    
    if args.interactive:
        print("\n===== Suicide Risk Prediction Tool =====")
        print("Type 'quit' or 'exit' to end the session")
        print("=======================================\n")
        
        while True:
            text = input("\nEnter text to analyze (or 'quit' to exit): ")
            if text.lower() in ['quit', 'exit']:
                break
                
            if text.strip():
                result = predictor.predict_risk(text)
                print_result(result)
            else:
                print("Please enter some text to analyze.")
    
    elif args.text:
        result = predictor.predict_risk(args.text)
        print_result(result)
    
    else:
        parser.print_help()

def print_result(result):
    """Print prediction results in a formatted way"""
    if not result:
        return
        
    print("\n----- PREDICTION RESULT -----")
    
    # Define risk level descriptions
    risk_descriptions = {
        'high': 'HIGH RISK: Immediate attention recommended. The text contains concerning indicators of potential self-harm or suicidal ideation.',
        'medium': 'MEDIUM RISK: Monitoring recommended. The text shows signs of distress that should be addressed.',
        'low': 'LOW RISK: Some concerning elements present. Supportive follow-up may be beneficial.',
        'none': 'NO APPARENT RISK: No clear indicators of suicidal ideation detected in this text.'
    }
    
    # Get the risk level and probability
    risk_level = result['prediction']
    probability = result['probabilities'][risk_level] * 100
    
    # Print the main prediction with probability
    print(f"Risk Level: {risk_level.upper()} ({probability:.1f}%)")
    
    # Print the description for this risk level
    if risk_level in risk_descriptions:
        print(f"\n{risk_descriptions[risk_level]}")
    
    # Print all probabilities
    print("\nProbability Breakdown:")
    for level, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {level.upper()}: {prob*100:.1f}%")
        
    print("\nNOTE: This is an automated analysis tool and should not replace professional clinical judgment.")
    print("If you or someone you know is in crisis, please contact a mental health professional or crisis service immediately.")
    print("--------------------------")

if __name__ == "__main__":
    main()
