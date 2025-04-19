import argparse
import json
from simple_param_predictor import SimpleParameterRiskPredictor

def main():
    parser = argparse.ArgumentParser(description='Parameter-based suicide risk assessment tool')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--train', '-t', action='store_true', help='Train a new model before assessment')
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SimpleParameterRiskPredictor()
    
    # Train or load model
    if args.train:
        predictor.train_model()
        predictor.save_model()
    else:
        model_loaded = predictor.load_model()
        if not model_loaded:
            print("Model not found. Training a new model...")
            predictor.train_model()
            predictor.save_model()
    
    if args.interactive:
        interactive_assessment(predictor)
    else:
        parser.print_help()

def interactive_assessment(predictor):
    """Interactive risk assessment using parameter inputs"""
    print("\n===== WHO Parameter-Based Suicide Risk Assessment Tool =====")
    print("This tool assesses suicide risk based on clinical parameters.")
    print("Type 'quit' or 'exit' at any time to end the session.")
    print("============================================================\n")
    
    # Get parameter descriptions for guidance
    param_descriptions = predictor.get_parameter_descriptions()
    risk_level_descriptions = predictor.get_risk_level_descriptions()
    
    while True:
        print("\nNew Risk Assessment\n" + "-" * 20)
        
        # Ask if user wants to proceed
        response = input("Begin a new assessment? (yes/no): ")
        if response.lower() in ['no', 'n', 'quit', 'exit']:
            break
        
        # Get parameter values
        parameter_values = {}
        severity_options = ['none', 'mild', 'moderate', 'severe', 'extreme']
        
        for param in predictor.parameters:
            # Print parameter description
            print(f"\n{param.replace('_', ' ').title()}: {param_descriptions[param]}")
            
            # Show severity options
            print("Options: " + " | ".join(severity_options))
            
            # Get user input with validation
            while True:
                value = input(f"Enter severity level (0-4 or {'/'.join(severity_options)}): ")
                
                if value.lower() in ['quit', 'exit']:
                    return
                
                # Check if input is one of the severity options or a valid number
                if value.lower() in severity_options:
                    parameter_values[param] = value.lower()
                    break
                elif value.isdigit() and 0 <= int(value) <= 4:
                    parameter_values[param] = int(value)
                    break
                else:
                    print("Invalid input. Please enter a valid severity level.")
        
        # Make prediction
        result = predictor.predict_risk(parameter_values)
        
        # Display results
        display_result(result, risk_level_descriptions)
        
        # Ask if user wants to save the assessment
        save = input("\nDo you want to save this assessment (yes/no)? ")
        if save.lower() in ['yes', 'y']:
            filename = f"assessment_{result['timestamp'].replace(' ', '_').replace(':', '-')}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Assessment saved to {filename}")

def display_result(result, risk_level_descriptions):
    """Display prediction results in a formatted way"""
    if not result:
        return
    
    prediction = result['prediction']
    risk_score = result['risk_score']
    
    print("\n" + "=" * 50)
    print("RISK ASSESSMENT RESULTS")
    print("=" * 50)
    
    # Define color codes for risk levels (for terminals that support ANSI)
    colors = {
        'low': '\033[92m',      # Green
        'moderate': '\033[93m',  # Yellow
        'high': '\033[91m',      # Red
        'critical': '\033[31;1m' # Bright Red
    }
    reset = '\033[0m'
    
    # Print risk level with color if possible
    print(f"\nRisk Level: {colors.get(prediction, '')}{prediction.upper()}{reset}")
    
    # Print risk score
    print(f"Risk Score: {risk_score:.1f}/100")
    
    # Print risk level description
    if prediction in risk_level_descriptions:
        print(f"\n{risk_level_descriptions[prediction]}")
    
    # Print probability breakdown
    print("\nProbability Breakdown:")
    for level, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {level.upper()}: {prob*100:.1f}%")
    
    print("\nRECOMMENDATIONS:")
    if prediction == 'low':
        print("- Consider follow-up assessment if circumstances change")
        print("- Provide resources for mental health support")
        print("- Educate about warning signs and when to seek help")
    elif prediction == 'moderate':
        print("- Regular follow-up assessments recommended")
        print("- Consider referral to mental health services")
        print("- Discuss safety planning and coping strategies")
        print("- Involve support network with consent")
    elif prediction == 'high':
        print("- Immediate intervention recommended")
        print("- Implement comprehensive safety plan")
        print("- Consider referral to specialized mental health services")
        print("- Evaluate need for more intensive monitoring")
        print("- Remove access to means if possible")
    elif prediction == 'critical':
        print("- URGENT ACTION REQUIRED")
        print("- Immediate psychiatric evaluation recommended")
        print("- Consider hospitalization or crisis services")
        print("- Continuous monitoring and support essential")
        print("- Immediate removal of access to potential means")
    
    print("\nIMPORTANT: This tool provides guidance only and does not replace")
    print("professional clinical judgment. When in doubt, always consult")
    print("with qualified mental health professionals and follow established")
    print("clinical protocols for suicide risk management.")
    print("=" * 50)

if __name__ == "__main__":
    main()
