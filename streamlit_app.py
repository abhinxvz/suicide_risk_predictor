import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import time
from simple_param_predictor import SimpleParameterRiskPredictor

# Configure page settings
st.set_page_config(
    page_title="WHO Suicide Risk Assessment Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up session state for theme and view mode if not exists
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'simple_view' not in st.session_state:
    st.session_state.simple_view = False

# Load custom CSS
def local_css():
    # Choose CSS based on theme
    if st.session_state.dark_mode:
        base_bg = "#121212"
        text_color = "#f0f0f0"
        container_bg = "#1e1e1e"
        sidebar_bg = "#262626"
        accent_color = "#4f8bf9"
        secondary_bg = "#333333"
        success_color = "#4CAF50"
        warning_color = "#FFC107"
        danger_color = "#FF5252"
    else:
        base_bg = "#f8f9fa"
        text_color = "#333333"
        container_bg = "#ffffff"
        sidebar_bg = "#f5f5f5"
        accent_color = "#4e89e8"
        secondary_bg = "#e9ecef"
        success_color = "#28a745"
        warning_color = "#ffc107"
        danger_color = "#dc3545"
        
    # Define risk level colors based on theme
    if st.session_state.dark_mode:
        risk_low_bg = "rgba(0, 128, 0, 0.3)"
        risk_low_border = "darkgreen"
        risk_moderate_bg = "rgba(255, 165, 0, 0.3)"
        risk_moderate_border = "darkorange"
        risk_high_bg = "rgba(255, 0, 0, 0.3)"
        risk_high_border = "darkred"
        risk_critical_bg = "rgba(139, 0, 0, 0.4)" 
        risk_critical_border = "rgb(139, 0, 0)"
    else:
        risk_low_bg = "rgba(0, 128, 0, 0.2)"
        risk_low_border = "darkgreen"
        risk_moderate_bg = "rgba(255, 165, 0, 0.2)"
        risk_moderate_border = "darkorange"
        risk_high_bg = "rgba(255, 0, 0, 0.2)"
        risk_high_border = "darkred"
        risk_critical_bg = "rgba(139, 0, 0, 0.3)"
        risk_critical_border = "rgb(139, 0, 0)"
    
    st.markdown(f"""
    <style>
        /* Base theme */
        .stApp {{
            background-color: {base_bg};
            color: {text_color};
        }}
        .stApp > header {{
            background-color: transparent !important;
        }}
        
        /* Sidebar styling */
        .css-1d391kg, [data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
        }}
        
        /* Clean container style */
        .clean-container {{
            background-color: {container_bg};
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        /* Risk boxes */
        .risk-box {{
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .risk-box-low {{
            background-color: {risk_low_bg};
            border: 2px solid {risk_low_border};
        }}
        .risk-box-moderate {{
            background-color: {risk_moderate_bg};
            border: 2px solid {risk_moderate_border};
        }}
        .risk-box-high {{
            background-color: {risk_high_bg};
            border: 2px solid {risk_high_border};
        }}
        .risk-box-critical {{
            background-color: {risk_critical_bg};
            border: 3px solid {risk_critical_border};
        }}
        
        /* Gauge and chart containers */
        .gauge-wrapper {{
            text-align: center;
            margin-bottom: 20px;
        }}
        
        /* Risk factors box */
        .risk-factors-box {{
            background-color: {secondary_bg};
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }}
        
        /* Disclaimer */
        .disclaimer {{
            font-size: 0.8rem;
            color: {text_color};
            opacity: 0.8;
            font-style: italic;
            background-color: {secondary_bg};
            padding: 10px;
            border-radius: 5px;
        }}
        
        /* Parameter card */
        .parameter-card {{
            background-color: {container_bg};
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }}
        
        /* Help text */
        .help-text {{
            font-size: 0.9rem;
            color: {text_color};
            opacity: 0.7;
        }}
        
        /* Buttons */
        .stButton > button {{
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        /* Sliders */
        .stSlider {{
            padding-top: 10px;
            padding-bottom: 10px;
        }}
        
        /* Theme toggle container */
        .theme-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin-bottom: 20px;
            background-color: {secondary_bg};
            border-radius: 10px;
        }}
        
        /* Simple view adaptations */
        .simple-view h1 {{
            font-size: 1.8rem;
        }}
        .simple-view h2 {{
            font-size: 1.5rem;
        }}
        .simple-view h3 {{
            font-size: 1.2rem;
        }}
        .simple-view p, .simple-view li {{
            font-size: 1rem;
        }}
        .simple-view .clean-container {{
            padding: 15px;
            margin-bottom: 15px;
        }}
        .simple-view .parameter-card {{
            padding: 10px;
            margin-bottom: 15px;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Apply simple view class if enabled
    if st.session_state.simple_view:
        st.markdown("""
        <style>
            .main {
                max-width: 800px;
                margin: 0 auto;
            }
        </style>
        """, unsafe_allow_html=True)
        # Add simple-view class to body
        st.markdown('<div class="simple-view">', unsafe_allow_html=True)

# Theme toggle functions
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode
    
def toggle_simple_view():
    st.session_state.simple_view = not st.session_state.simple_view

# Set up session state
def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'welcome'
    if 'parameters' not in st.session_state:
        st.session_state.parameters = {}
    if 'assessment_complete' not in st.session_state:
        st.session_state.assessment_complete = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'predictor' not in st.session_state:
        st.session_state.predictor = SimpleParameterRiskPredictor()
        # Try to load existing model
        try:
            st.session_state.predictor.load_model()
        except:
            pass

# Helper functions
def create_severity_scale(key, value_key, help_text=None):
    """Create severity selector with labels and visual indicators"""
    options = ["none", "mild", "moderate", "severe", "extreme"]
    labels = {
        "none": "None (0)",
        "mild": "Mild (1)",
        "moderate": "Moderate (2)",
        "severe": "Severe (3)",
        "extreme": "Extreme (4)"
    }
    
    # Get current value from session state if exists
    current_val = st.session_state.parameters.get(value_key, "none")
    
    col1, col2 = st.columns([3, 7])
    with col1:
        st.write("Severity:")
    with col2:
        selected = st.select_slider(
            key=key,
            label="",
            options=options,
            format_func=lambda x: labels[x],
            value=current_val
        )
        st.session_state.parameters[value_key] = selected
    
    if help_text:
        st.markdown(f"<div class='help-text'>{help_text}</div>", unsafe_allow_html=True)
    
    return selected

def display_gauge(value, title="Risk Score"):
    """Display a semi-circular gauge for the risk score"""
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})
    
    # Adjust colors based on theme
    if st.session_state.dark_mode:
        background_color = "#1e1e1e"
        text_color = "#f0f0f0"
        gauge_bg_color = "#333333"
        gauge_bg_alpha = 0.7
    else:
        background_color = "white"
        text_color = "black"
        gauge_bg_color = "lightgray"
        gauge_bg_alpha = 0.5
    
    # Set figure background color
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    
    # Define the angles for the gauge (semi-circle)
    theta = np.linspace(0, np.pi, 100)
    
    # Define the colormap
    cmap = plt.cm.RdYlGn_r  # Red-Yellow-Green reversed
    norm = plt.Normalize(0, 100)
    
    # Plot the background
    ax.plot(theta, [1] * 100, color=gauge_bg_color, linewidth=20, alpha=gauge_bg_alpha)
    
    # Calculate angle for the value marker
    value_angle = np.pi * (value / 100)
    
    # Plot the value
    ax.plot([0, value_angle], [0, 0.9], color=text_color, linewidth=2.5)
    
    # Add color segmentation
    for i in range(100):
        angle = np.pi * (i / 100)
        ax.plot([angle, angle], [0.8, 1.0], color=cmap(norm(i)), linewidth=3)
    
    # Add labels with theme-appropriate colors
    ax.text(np.pi/6, 0.6, "Low", ha='center', va='center', fontsize=10, color=text_color)
    ax.text(np.pi/2, 0.6, "Moderate", ha='center', va='center', fontsize=10, color=text_color)
    ax.text(5*np.pi/6, 0.6, "High", ha='center', va='center', fontsize=10, color=text_color)
    
    # Add the value in the center
    ax.text(np.pi/2, 0.2, f"{value:.1f}/100", ha='center', va='center', fontsize=18, fontweight='bold', color=text_color)
    ax.text(np.pi/2, 0.1, title, ha='center', va='center', fontsize=12, color=text_color)
    
    # Configure the chart appearance
    ax.set_rticks([])  # Remove radial ticks
    ax.set_xticks([])  # Remove angular ticks
    ax.spines['polar'].set_visible(False)  # Remove the outer circle
    
    # Set the limits
    ax.set_ylim(0, 1.2)
    
    # Remove grid
    ax.grid(False)
    
    return fig

def reset_assessment():
    """Reset the current assessment"""
    st.session_state.parameters = {}
    st.session_state.assessment_complete = False
    st.session_state.page = 'assessment'

def navigation():
    """Navigation sidebar"""
    with st.sidebar:
        st.title("Navigation")
        
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = 'welcome'
        
        if st.button("📝 New Assessment", use_container_width=True):
            reset_assessment()
            st.session_state.page = 'assessment'
        
        if st.button("📊 Assessment History", use_container_width=True):
            st.session_state.page = 'history'
        
        if st.button("ℹ️ About", use_container_width=True):
            st.session_state.page = 'about'
        
        st.markdown("---")
        
        # Theme toggle
        with st.container():
            st.markdown('<div class="theme-container">', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 7])
            with col1:
                st.write("Theme:")
            with col2:
                theme = st.selectbox("", ["Light", "Dark"], key="theme")
                if theme == "Dark":
                    toggle_dark_mode()
                else:
                    st.session_state.dark_mode = False
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Simple view toggle
        with st.container():
            st.markdown('<div class="theme-container">', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 7])
            with col1:
                st.write("View:")
            with col2:
                view = st.selectbox("", ["Standard", "Simple"], key="view")
                if view == "Simple":
                    toggle_simple_view()
                else:
                    st.session_state.simple_view = False
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown(
            """
            <div class="disclaimer">
            This tool is intended for educational purposes only and should not replace professional 
            clinical judgment. Always consult with qualified healthcare providers for assessment
            and treatment decisions.
            </div>
            """, 
            unsafe_allow_html=True
        )

# Page functions
def welcome_page():
    """Welcome page with introduction to the tool"""
    st.title("WHO Parameter-Based Suicide Risk Assessment Tool")
    
    # Create a clean main container
    with st.container():
        st.markdown('<div class="clean-container">', unsafe_allow_html=True)
        st.markdown(
            """
            ### Welcome to the Interactive Suicide Risk Assessment Tool

            This application uses clinically relevant parameters based on WHO research to help assess 
            suicide risk levels. It implements a machine learning model trained on clinical patterns
            to provide an objective assessment of risk.

            **Key features:**
            - Comprehensive assessment using 15 clinical parameters
            - Evidence-based risk evaluation
            - Visualization of risk factors and warning signs
            - Private and secure assessment (all data remains on your computer)
            """
        )
        
        # Call-to-action button
        if st.button("Begin New Assessment", type="primary", use_container_width=True):
            st.session_state.page = 'assessment'
            st.experimental_rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Information boxes in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="clean-container">', unsafe_allow_html=True)
        st.subheader("How It Works")
        st.markdown("""
            1. Answer questions about 15 clinical parameters
            2. The tool analyzes your responses using a machine learning model
            3. View your personalized risk assessment with recommendations
            4. Track changes in risk profile over time
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="clean-container">', unsafe_allow_html=True)
        st.subheader("Clinical Parameters")
        st.markdown("""
            The assessment evaluates key factors including:
            - Suicidal thoughts, plans, and intent
            - Psychological factors (depression, anxiety)
            - Behavioral indicators
            - Social and environmental circumstances
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Important notice
    st.markdown('<div class="clean-container">', unsafe_allow_html=True)
    st.warning("⚠️ **Disclaimer**: This tool is for educational purposes only. It is not a substitute for professional clinical judgment or care. Always consult qualified healthcare providers for assessment and treatment.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Close simple-view div if enabled
    if st.session_state.simple_view:
        st.markdown('</div>', unsafe_allow_html=True)

def assessment_page():
    """Assessment page with parameter input"""
    st.title("Suicide Risk Assessment Tool")
    st.markdown("### Rate all factors to predict suicide risk level")

    # Get parameter descriptions
    param_descriptions = st.session_state.predictor.get_parameter_descriptions()

    # Create a simple severity selector
    def create_severity_scale(key, param_name):
        options = ["none", "mild", "moderate", "severe", "extreme"]
        value = st.select_slider(
            f"{param_name.replace('_', ' ').title()}",
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
        create_severity_scale("st_slider", "suicidal_thoughts")
        create_severity_scale("sp_slider", "suicide_plan")
        create_severity_scale("si_slider", "suicide_intent")
        create_severity_scale("pa_slider", "prior_attempts")
        create_severity_scale("sh_slider", "self_harm")

    with col2:
        st.subheader("Psychological & Behavioral")
        create_severity_scale("hp_slider", "hopelessness")
        create_severity_scale("dp_slider", "depression")
        create_severity_scale("ax_slider", "anxiety")
        create_severity_scale("is_slider", "isolation")
        create_severity_scale("im_slider", "impulsivity")

    with col3:
        st.subheader("Other Factors")
        create_severity_scale("su_slider", "substance_use")
        create_severity_scale("sl_slider", "sleep_problems")
        create_severity_scale("ls_slider", "life_stressors")
        create_severity_scale("am_slider", "access_to_means")
        create_severity_scale("sn_slider", "support_network")

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

    # Close simple-view div if enabled
    if st.session_state.simple_view:
        st.markdown('</div>', unsafe_allow_html=True)

def results_page():
    """Results page showing assessment results"""
    if not hasattr(st.session_state, 'result'):
        st.warning("No assessment results to display. Please complete an assessment first.")
        if st.button("Start New Assessment"):
            reset_assessment()
        return
    
    result = st.session_state.result
    prediction = result['prediction']
    risk_score = result['risk_score']
    probabilities = result['probabilities']
    
    # Get risk level descriptions
    risk_descriptions = st.session_state.predictor.get_risk_level_descriptions()
    
    # Main container with header
    st.markdown('<div class="clean-container">', unsafe_allow_html=True)
    
    # Title and risk level display
    st.title("Risk Assessment Results")
    
    # Display risk level with appropriate styling
    risk_box_class = f"risk-box risk-box-{prediction.lower()}"
    st.markdown(f"""
        <div class="{risk_box_class}">
            <h2>{prediction.upper()} RISK</h2>
            <p>{risk_descriptions.get(prediction, '')}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Two columns for risk score and probability breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        # Container for risk score
        st.markdown('<div class="clean-container">', unsafe_allow_html=True)
        st.subheader("Risk Score")
        fig = display_gauge(risk_score)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Container for probability breakdown
        st.markdown('<div class="clean-container">', unsafe_allow_html=True)
        st.subheader("Probability Breakdown")
        
        # Sort probabilities by value in descending order
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Create a horizontal bar chart for probabilities
        labels = [item[0].upper() for item in sorted_probs]
        values = [item[1] * 100 for item in sorted_probs]
        
        # Define colors for each risk level
        colors = {
            'LOW': 'green',
            'MODERATE': 'orange',
            'HIGH': 'red',
            'CRITICAL': 'darkred'
        }
        
        # Create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 3))
        bars = ax.barh(labels, values, color=[colors.get(label, 'gray') for label in labels])
        ax.set_xlabel('Probability (%)')
        ax.set_xlim(0, 100)
        
        # Add percentage labels on bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 1
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                    va='center')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Container for risk factors and warnings
    st.markdown('<div class="clean-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Risk Factors")
        if 'risk_factors' in result and 'high_risk_factors' in result['risk_factors']:
            high_risk_factors = result['risk_factors']['high_risk_factors']
            if high_risk_factors:
                for factor in high_risk_factors:
                    st.markdown(f"- **{factor.replace('_', ' ').title()}**")
            else:
                st.write("No severe or extreme risk factors identified.")
        else:
            st.write("Risk factor analysis not available.")
    
    with col2:
        st.subheader("Warning Signs")
        if 'risk_factors' in result and 'warning_combinations' in result['risk_factors']:
            warnings = result['risk_factors']['warning_combinations']
            if warnings:
                for warning in warnings:
                    # Format warning
                    if warning == 'thoughts_with_plan':
                        st.markdown("- ⚠️ **Suicidal thoughts with specific plan**")
                    elif warning == 'intent_with_means':
                        st.markdown("- ⚠️ **Suicidal intent with access to means**")
                    elif warning == 'severe_depression_hopelessness':
                        st.markdown("- ⚠️ **Severe depression with hopelessness**")
                    else:
                        st.markdown(f"- ⚠️ **{warning.replace('_', ' ').title()}**")
            else:
                st.write("No specific warning combinations identified.")
        else:
            st.write("Warning sign analysis not available.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Container for recommendations
    st.markdown('<div class="clean-container">', unsafe_allow_html=True)
    
    # Recommendations based on risk level
    st.subheader("Recommendations")
    
    if prediction == 'low':
        st.markdown("""
            - Consider follow-up assessment if circumstances change
            - Provide resources for mental health support
            - Educate about warning signs and when to seek help
        """)
    elif prediction == 'moderate':
        st.markdown("""
            - Regular follow-up assessments recommended
            - Consider referral to mental health services
            - Discuss safety planning and coping strategies
            - Involve support network with consent
        """)
    elif prediction == 'high':
        st.markdown("""
            - **Immediate intervention recommended**
            - Implement comprehensive safety plan
            - Consider referral to specialized mental health services
            - Evaluate need for more intensive monitoring
            - Remove access to means if possible
        """)
    elif prediction == 'critical':
        st.markdown("""
            - **URGENT ACTION REQUIRED**
            - Immediate psychiatric evaluation recommended
            - Consider hospitalization or crisis services
            - Continuous monitoring and support essential
            - Immediate removal of access to potential means
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Parameter summary in expandable section
    with st.expander("View Parameter Details"):
        # Create a dataframe for parameter values
        param_data = []
        for param, value in result['parameters'].items():
            # Convert numeric values to severity labels
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
                value_int = int(value) if isinstance(value, int) else int(value)
                severity_levels = {0: "none", 1: "mild", 2: "moderate", 3: "severe", 4: "extreme"}
                value_str = severity_levels.get(value_int, value)
            else:
                value_str = value
                
            param_data.append({
                "Parameter": param.replace('_', ' ').title(),
                "Severity": value_str.title()
            })
        
        df = pd.DataFrame(param_data)
        st.dataframe(df, use_container_width=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("New Assessment", use_container_width=True):
            reset_assessment()
            st.experimental_rerun()
    
    with col2:
        if st.button("Back to Home", use_container_width=True):
            st.session_state.page = 'welcome'
            st.experimental_rerun()
    
    # Disclaimer
    st.markdown("""
        <div class="disclaimer">
        <strong>Important:</strong> This tool provides guidance only and does not replace 
        professional clinical judgment. When in doubt, always consult with qualified mental 
        health professionals and follow established clinical protocols for suicide risk management.
        </div>
    """, unsafe_allow_html=True)
    
    # Close simple-view div if enabled
    if st.session_state.simple_view:
        st.markdown('</div>', unsafe_allow_html=True)

def history_page():
    """Display assessment history"""
    st.title("Assessment History")
    
    if not st.session_state.history:
        st.info("No assessment history available. Complete an assessment to see it here.")
        return
    
    # Show history in reverse chronological order
    for i, result in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Assessment {i+1} - {result.get('timestamp', 'No Date')}"):
            st.markdown(f"**Risk Level:** {result['prediction'].upper()}")
            st.markdown(f"**Risk Score:** {result['risk_score']:.1f}/100")
            
            # Format probabilities
            st.markdown("**Probability Breakdown:**")
            probs_md = ""
            for level, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                probs_md += f"- {level.upper()}: {prob*100:.1f}%\n"
            st.markdown(probs_md)
            
            # Display parameters
            st.markdown("**Parameters:**")
            param_data = []
            for param, value in result['parameters'].items():
                param_data.append([param.replace('_', ' ').title(), value])
            
            param_df = pd.DataFrame(param_data, columns=["Parameter", "Severity"])
            st.dataframe(param_df)
    
    # Close simple-view div if enabled
    if st.session_state.simple_view:
        st.markdown('</div>', unsafe_allow_html=True)

def about_page():
    """About page with information about the tool"""
    st.title("About the WHO Parameter-Based Suicide Risk Assessment Tool")
    
    st.markdown("""
        ### Tool Overview
        
        This application implements a machine learning-based suicide risk assessment tool that uses 
        15 clinically relevant parameters based on WHO research. It is designed to help identify 
        potential suicide risk levels through a structured parameter assessment.
        
        ### Key Features
        
        - **Evidence-based parameters**: Draws from WHO and clinical research on suicide risk factors
        - **Machine learning model**: Uses Random Forest algorithm trained on patterns from clinical literature
        - **Probability calibration**: Provides realistic probability distribution across risk levels
        - **Risk factor identification**: Highlights key risk factors and dangerous combinations
        - **Clinical recommendations**: Offers guidance based on assessed risk level
        
        ### Parameters Assessed
        
        The tool evaluates 15 key parameters grouped into four categories:
        
        1. **Suicidal Behavior**
           - Suicidal thoughts
           - Suicide plan
           - Suicide intent
           - Prior attempts
           - Self-harm
        
        2. **Psychological Factors**
           - Hopelessness
           - Depression
           - Anxiety
        
        3. **Behavioral Factors**
           - Social isolation
           - Impulsivity
           - Substance use
           - Sleep problems
        
        4. **Circumstances**
           - Life stressors
           - Access to means
           - Support network
        
        ### Risk Levels
        
        The tool categorizes risk into four levels:
        
        - **Low Risk**: No immediate danger, but monitoring may be needed
        - **Moderate Risk**: Regular monitoring and support recommended
        - **High Risk**: Immediate intervention and safety planning needed
        - **Critical Risk**: Urgent psychiatric evaluation/hospitalization needed
        
        ### Technical Information
        
        The backend uses a Random Forest classifier trained on synthetic data that reflects 
        clinical patterns from suicide risk research. The model has been tuned to provide 
        balanced and clinically meaningful probability estimates across risk categories.
        
        ### Limitations and Disclaimer
        
        This tool is for educational purposes only and has significant limitations:
        
        - It is not a validated clinical instrument
        - It should never replace professional assessment
        - It has been trained on synthetic data, not real patient data
        - Risk assessment requires comprehensive clinical judgment
        
        Always consult qualified mental health professionals for proper assessment and care.
    """)
    
    st.info("This application was developed as an educational tool to demonstrate how machine learning can be applied to mental health assessment, but it should never be used as a substitute for professional clinical care.")
    
    # Close simple-view div if enabled
    if st.session_state.simple_view:
        st.markdown('</div>', unsafe_allow_html=True)

# Main app
def main():
    local_css()
    init_session_state()
    navigation()
    
    # Check if model needs training
    if not hasattr(st.session_state.predictor, 'model') or st.session_state.predictor.model is None:
        model_path = 'simple_risk_model.pkl'
        
        if os.path.exists(model_path):
            try:
                st.session_state.predictor.load_model(model_path)
            except:
                with st.spinner("Training initial model (first run only)..."):
                    st.session_state.predictor.train_model()
                    st.session_state.predictor.save_model(model_path)
        else:
            with st.spinner("Training initial model (first run only)..."):
                st.session_state.predictor.train_model()
                st.session_state.predictor.save_model(model_path)
    
    # Display the appropriate page
    if st.session_state.page == 'welcome':
        welcome_page()
    elif st.session_state.page == 'assessment':
        assessment_page()
    elif st.session_state.page == 'results':
        results_page()
    elif st.session_state.page == 'history':
        history_page()
    elif st.session_state.page == 'about':
        about_page()
        
    # Close simple-view div if enabled
    if st.session_state.simple_view:
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
