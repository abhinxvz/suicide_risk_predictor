import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from simple_param_predictor import SimpleParameterRiskPredictor


st.set_page_config(
    page_title="Suicide Risk Assessment Tool",
    page_icon="🧠",
    layout="wide"
)


if 'predictor' not in st.session_state:
    st.session_state.predictor = SimpleParameterRiskPredictor()
    # Try to load existing model
    try:
        st.session_state.predictor.load_model()
    except:
        st.session_state.predictor.train_model()
        st.session_state.predictor.save_model()

# Reset parameters if not in session state
if 'parameters' not in st.session_state:
    st.session_state.parameters = {}

# Main title
st.title("Suicide Risk Assessment Tool")
st.markdown("### Rate all factors to predict suicide risk level")

# Get parameter descriptions
param_descriptions = st.session_state.predictor.get_parameter_descriptions()

# Create a simple severity selector
def create_severity_scale(key, param_name, description):
    options = ["none", "mild", "moderate", "severe", "extreme"]
    value = st.select_slider(
        f"{description}",
        options=options,
        key=key
    )
    st.session_state.parameters[param_name] = value
    return value

# Create three columns for parameters
col1, col2, col3 = st.columns(3)

# Split parameters among columns
with col1:
    st.subheader("Suicidal Behavior")
    create_severity_scale("st_slider", "suicidal_thoughts", "How frequent and intense are thoughts about suicide?")
    create_severity_scale("sp_slider", "suicide_plan", "Has the person developed a specific plan for suicide?")
    create_severity_scale("si_slider", "suicide_intent", "How strong is the person's intention to act on suicidal thoughts?")
    create_severity_scale("pa_slider", "prior_attempts", "Has the person attempted suicide in the past?")
    create_severity_scale("sh_slider", "self_harm", "Does the person engage in self-harming behaviors?")

with col2:
    st.subheader("Psychological & Behavioral")
    create_severity_scale("hp_slider", "hopelessness", "Does the person feel hopeless about their future?")
    create_severity_scale("dp_slider", "depression", "How severe are the person's depressive symptoms?")
    create_severity_scale("ax_slider", "anxiety", "To what extent does anxiety affect the person's daily life?")
    create_severity_scale("is_slider", "isolation", "How socially isolated or withdrawn is the person?")
    create_severity_scale("im_slider", "impulsivity", "Does the person act impulsively without considering consequences?")

with col3:
    st.subheader("Other Factors")
    create_severity_scale("su_slider", "substance_use", "Is there evidence of alcohol or drug use problems?")
    create_severity_scale("sl_slider", "sleep_problems", "How severe are the person's sleep disturbances?")
    create_severity_scale("ls_slider", "life_stressors", "Are there significant stressful events affecting the person?")
    create_severity_scale("am_slider", "access_to_means", "Does the person have access to lethal means?")
    create_severity_scale("sn_slider", "support_network", "How strong is the person's social support system?")

# Predict button
if st.button("Predict Risk Level", type="primary"):
    # Make prediction
    with st.spinner("Analyzing risk factors..."):
        result = st.session_state.predictor.predict_risk(st.session_state.parameters)
        
        # Show results
        st.markdown("---")
        st.subheader("Assessment Results")
        
        # Display risk level
        prediction = result['prediction']
        risk_score = result['risk_score']
        
        # Simple risk level display
        if prediction == 'low':
            risk_color = "green"
        elif prediction == 'moderate':
            risk_color = "orange"
        elif prediction == 'high':
            risk_color = "red"
        else:  # critical
            risk_color = "darkred"
            
        st.markdown(f"<h2 style='color: {risk_color};'>{prediction.upper()} RISK</h2>", unsafe_allow_html=True)
        st.markdown(f"**Risk Score:** {risk_score:.1f}/100")
        
        # Show probability breakdown
        st.subheader("Probability Breakdown")
        for level, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"- **{level.upper()}**: {prob*100:.1f}%")
        
        # Show key risk factors if present
        if 'risk_factors' in result and result['risk_factors']['high_risk_factors']:
            st.subheader("Key Risk Factors")
            for factor in result['risk_factors']['high_risk_factors']:
                st.markdown(f"- {factor.replace('_', ' ').title()}")
                
        # Show warning combinations if present
        if 'risk_factors' in result and result['risk_factors']['warning_combinations']:
            st.subheader("Warning Combinations")
            for warning in result['risk_factors']['warning_combinations']:
                st.markdown(f"- {warning.replace('_', ' ').title()}")
                
        # Disclaimer
        st.markdown("---")
        st.caption("This tool is for educational purposes only. It is not a substitute for professional clinical judgment or care.")
        
        # Adding Indian helpline information and resources
        st.markdown("---")
        st.subheader("Helplines and Resources")
        
        # Helplines section
        st.markdown("### 🆘 Indian Suicide Prevention Helplines")
        helplines = """
        - **AASRA**: 91-9820466726 (24/7) | [aasra.info](http://www.aasra.info/)
        - **Lifeline Foundation**: +91 33 24637401/24637432 | [lifelinefoundation.org](https://lifelinefoundation.org/)
        - **Sneha Foundation India**: +91-44-24640050 (24/7) | [snehaindia.org](https://www.snehaindia.org/)
        - **Vandrevala Foundation**: 1860-2662-345 / +91-1860-2662-345 (24/7)
        - **iCall**: +91 22 2556 3291 (Mon-Sat, 8 AM-10 PM)
        - **NIMHANS**: 080-26995000 | Toll-free: 1800-599-0019
        """
        st.markdown(helplines)
        
        # Ways to avoid suicidal thoughts
        st.markdown("### 🌱 Ways to Take Care of Mental Health")
        prevention = """
        1. **Seek professional help**: Talk to a psychiatrist, psychologist, or counselor
        2. **Connect with others**: Maintain relationships with trusted friends and family
        3. **Develop a safety plan**: Identify warning signs and emergency contacts
        4. **Practice self-care**: Ensure adequate sleep, nutrition, and physical activity
        5. **Mindfulness and meditation**: Try apps like Headspace or Calm
        6. **Avoid alcohol and drugs**: These can worsen suicidal thoughts
        7. **Set small goals**: Focus on manageable daily achievements
        8. **Identify triggers**: Be aware of situations that may worsen mental health
        """
        st.markdown(prevention)
        
        # Healing resources section
        st.markdown("### 🎮 Apps and Games for Mental Wellness")
        resources = """
        - **SuperBetter**: A game that builds resilience and positive mental health
        - **Happify**: Science-based activities and games to reduce stress
        - **Calm**: Meditation and mindfulness app with sleep stories
        - **Headspace**: Guided meditation and mindfulness exercises
        - **Youper**: AI-powered emotional health assistant
        - **Wysa**: AI chatbot for mental health support
        - **Habitica**: Turn your life into an RPG to build healthy habits
        """
        st.markdown(resources)
        
        # Websites for mental health
        st.markdown("### 🌐 Websites for Mental Health Support")
        websites = """
        - **[It's OK To Talk](https://itsoktotalk.in/)**: Indian youth mental health platform
        - **[The Live Love Laugh Foundation](https://www.thelivelovelaughfoundation.org/)**: Mental health awareness
        - **[MINDS Foundation](https://www.mindsfoundation.org/)**: Mental health education and treatment
        - **[Anxiety and Depression Association of India (ADAI)](https://www.nimh.nih.gov/health/topics/anxiety-disorders)**: Resources for anxiety disorders
        - **[Manastha](https://www.manastha.com/)**: Counseling and mental health services in India
        - **[Talkspace](https://www.talkspace.com/)**: Online therapy services
        - **[7 Cups](https://www.7cups.com/)**: Free emotional support from trained listeners
        """
        st.markdown(websites)
        
        # Add attribution with GitHub link at the bottom
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; margin-top: 30px; padding: 10px;">
            <p>Made by <a href="https://github.com/abhinxvz" target="_blank">abhinxvz</a></p>
        </div>
        """, unsafe_allow_html=True)
