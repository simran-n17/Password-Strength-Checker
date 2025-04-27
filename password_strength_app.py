import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import re
from password_dataset import generate_password_dataset

# Set page configuration
st.set_page_config(
    page_title="Password Strength Predictor",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_dataset(n_samples=1000):
    """Load or generate the password dataset"""
    file_path = "password_strength_dataset.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        df = generate_password_dataset(n_samples)
        df.to_csv(file_path, index=False)
        return df

def train_model(X_train, y_train, model_type="Random Forest"):
    """Train a machine learning model for password strength prediction"""
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:  # Decision Tree
        model = DecisionTreeClassifier(random_state=42)
    
    model.fit(X_train, y_train)
    return model

def extract_password_features(password):
    """
    Extract features from a password string
    
    Returns:
        Dictionary of password features
    """
    features = {
        'Length': len(password),
        'Has_Lowercase': 1 if re.search(r'[a-z]', password) else 0,
        'Has_Uppercase': 1 if re.search(r'[A-Z]', password) else 0,
        'Has_Digit': 1 if re.search(r'\d', password) else 0,
        'Has_Special': 1 if re.search(r'[!@#$%^&*()_\-+=\[\]{}|:;,.<>?/~]', password) else 0,
        'Lowercase_Count': sum(c.islower() for c in password),
        'Uppercase_Count': sum(c.isupper() for c in password),
        'Digit_Count': sum(c.isdigit() for c in password),
        'Special_Count': sum(not c.isalnum() for c in password)
    }
    return features

def suggest_stronger_password(password, strength):
    """
    Suggest improvements to make a weak or medium password stronger
    
    Args:
        password: Original password
        strength: Predicted strength ('Weak', 'Medium', 'Strong')
        
    Returns:
        List of suggestions and improved password
    """
    if strength == "Strong":
        return ["Your password is already strong!"], password
    
    suggestions = []
    improved = password
    
    # Get current features
    features = extract_password_features(password)
    
    # Check length
    if features['Length'] < 8:
        suggestions.append("Increase length to at least 8 characters")
        # Add more characters to reach minimum length
        additional_chars_needed = max(0, 8 - features['Length'])
        improved += "!A1" * (additional_chars_needed // 3 + 1)
    elif features['Length'] < 12:
        suggestions.append("Consider increasing length to 12 or more characters")
    
    # Check character types
    if features['Has_Lowercase'] == 0:
        suggestions.append("Add lowercase letters")
        improved += 'abc'
    
    if features['Has_Uppercase'] == 0:
        suggestions.append("Add uppercase letters")
        improved += 'XYZ'
    
    if features['Has_Digit'] == 0:
        suggestions.append("Add numeric digits")
        improved += '123'
    
    if features['Has_Special'] == 0:
        suggestions.append("Add special characters (e.g., !@#$%^)")
        improved += '!@#'
    
    # Additional suggestions for weak passwords
    if strength == "Weak":
        if features['Length'] < 10:
            suggestions.append("Aim for at least 10 characters for better security")
        
        # Check for common patterns
        if password.lower() in ["password", "123456", "qwerty", "admin", "welcome"]:
            suggestions.append("Avoid common password words")
        
        if improved == password:  # No changes made yet
            improved = password + "!A1b2C3"
            
    return suggestions, improved

def display_password_strength_meter(strength):
    """Display a visual strength meter"""
    if strength == "Weak":
        color = "red"
        width = 33
    elif strength == "Medium":
        color = "orange"
        width = 67
    else:  # Strong
        color = "green"
        width = 100
    
    st.markdown(
        f"""
        <div style="border: 1px solid #ccc; width: 100%; height: 20px; border-radius: 5px; overflow: hidden;">
            <div style="background-color: {color}; width: {width}%; height: 100%;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

def visualize_data(df):
    """Create visualizations for the password dataset"""
    st.subheader("Password Dataset Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of password strengths
        fig = px.pie(df, names='Strength', title='Distribution of Password Strengths',
                    color='Strength', color_discrete_map={'Weak': 'red', 'Medium': 'orange', 'Strong': 'green'})
        st.plotly_chart(fig)
    
    with col2:
        # Password length by strength category
        fig = px.box(df, x='Strength', y='Length', color='Strength', 
                    title='Password Length by Strength Category',
                    color_discrete_map={'Weak': 'red', 'Medium': 'orange', 'Strong': 'green'})
        st.plotly_chart(fig)
    
    # Feature correlation
    feature_cols = ['Length', 'Has_Lowercase', 'Has_Uppercase', 'Has_Digit', 'Has_Special',
                    'Lowercase_Count', 'Uppercase_Count', 'Digit_Count', 'Special_Count']
    
    corr = df[feature_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                   title='Feature Correlation Matrix')
    st.plotly_chart(fig)
    
    # Character type distribution by strength
    char_types = pd.melt(df, id_vars=['Strength'], value_vars=['Lowercase_Count', 'Uppercase_Count', 
                                                             'Digit_Count', 'Special_Count'],
                        var_name='Character Type', value_name='Count')
    
    fig = px.box(char_types, x='Character Type', y='Count', color='Strength',
                title='Character Type Distribution by Password Strength',
                color_discrete_map={'Weak': 'red', 'Medium': 'orange', 'Strong': 'green'})
    st.plotly_chart(fig)

def main():
    st.title("🔒 Password Strength Predictor")
    st.markdown("### Predict and improve your password security using machine learning")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Introduction", "Dataset Exploration", "Model Training", "Password Prediction"]
    )
    
    # Load dataset
    dataset = load_dataset()
    
    # Introduction
    if page == "Introduction":
        st.header("Introduction to Password Security")
        
        st.markdown("""
        ### Why Password Security Matters
        
        In today's digital world, strong passwords are your first line of defense against unauthorized access.
        A weak password can be cracked within seconds to minutes, while a strong password might take years or be virtually impossible to crack.
        
        ### How We Measure Password Strength
        
        This application uses machine learning to classify passwords into three categories:
        
        - **Weak**: Easy to guess or crack (e.g., common words, simple patterns, short length)
        - **Medium**: Better security but could be improved (e.g., longer but lacking variety)
        - **Strong**: High security with length and complexity (e.g., mix of characters, sufficient length)
        
        ### Key Factors in Password Strength
        
        - **Length**: Longer passwords are generally stronger
        - **Complexity**: Mix of uppercase, lowercase, numbers, and special characters
        - **Unpredictability**: Avoiding common words and patterns
        
        Use the sidebar to explore the dataset, train models, and test your own passwords!
        """)
        
        st.info("**Did you know?** A 12-character random password with a mix of character types would take about 34,000 years to crack using current technology.")
        
        # Show examples
        st.subheader("Example Password Strengths")
        examples = pd.DataFrame({
            "Password": ["123456", "Summer2023", "P@ssw0rd!2023"],
            "Strength": ["Weak", "Medium", "Strong"],
            "Why": ["Short, only numbers, extremely common", 
                   "Has mixed case and numbers, but uses a common word pattern", 
                   "Good length, mix of all character types, not a common pattern"]
        })
        st.table(examples)
    
    # Dataset Exploration
    elif page == "Dataset Exploration":
        st.header("Password Dataset Exploration")
        
        st.markdown(f"""
        This dataset contains {len(dataset)} passwords with various features extracted from them.
        These features help our machine learning model learn patterns that distinguish between weak, medium, and strong passwords.
        """)
        
        # Display dataset statistics
        st.subheader("Dataset Preview")
        st.dataframe(dataset.head())
        
        # Show feature descriptions
        st.subheader("Feature Descriptions")
        feature_desc = pd.DataFrame({
            "Feature": ["Length", "Has_Lowercase", "Has_Uppercase", "Has_Digit", "Has_Special", 
                       "Lowercase_Count", "Uppercase_Count", "Digit_Count", "Special_Count"],
            "Description": [
                "Total number of characters in the password",
                "Whether the password contains lowercase letters (0=No, 1=Yes)",
                "Whether the password contains uppercase letters (0=No, 1=Yes)",
                "Whether the password contains digits (0=No, 1=Yes)",
                "Whether the password contains special characters (0=No, 1=Yes)",
                "Number of lowercase letters in the password",
                "Number of uppercase letters in the password",
                "Number of digits in the password",
                "Number of special characters in the password"
            ]
        })
        st.table(feature_desc)
        
        # Display visualizations
        visualize_data(dataset)
    
    # Model Training
    elif page == "Model Training":
        st.header("Machine Learning Model Training")
        
        st.markdown("""
        Train a machine learning model to predict password strength based on the features we've extracted.
        You can choose between different algorithms and see how well they perform.
        """)
        
        # Select features
        st.subheader("Feature Selection")
        
        feature_options = ["Length", "Has_Lowercase", "Has_Uppercase", "Has_Digit", "Has_Special", 
                           "Lowercase_Count", "Uppercase_Count", "Digit_Count", "Special_Count"]
        
        selected_features = st.multiselect(
            "Select features to use for training",
            options=feature_options,
            default=feature_options
        )
        
        if not selected_features:
            st.warning("Please select at least one feature")
            return
        
        # Select algorithm
        algorithm = st.selectbox(
            "Select machine learning algorithm",
            options=["Random Forest", "Decision Tree"]
        )
        
        # Train-test split ratio
        test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
        
        if st.button("Train Model"):
            # Prepare data
            X = dataset[selected_features]
            y = dataset['Strength']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            st.info(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
            
            # Train model
            with st.spinner("Training model..."):
                model = train_model(X_train, y_train, model_type=algorithm)
                
                # Make predictions on test set
                y_pred = model.predict(X_test)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                st.success(f"Model trained! Accuracy: {accuracy:.2f}")
                
                # Display classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                
                # Display confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, 
                                x=['Predicted Weak', 'Predicted Medium', 'Predicted Strong'],
                                y=['Actual Weak', 'Actual Medium', 'Actual Strong'],
                                text_auto=True, 
                                color_continuous_scale='Blues')
                st.plotly_chart(fig)
                
                # Display feature importance
                if algorithm == "Random Forest" or algorithm == "Decision Tree":
                    st.subheader("Feature Importance")
                    importances = model.feature_importances_
                    feat_imp = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                                title='Feature Importance')
                    st.plotly_chart(fig)
                
                # Save model
                joblib.dump(model, f"password_strength_{algorithm.lower().replace(' ', '_')}.pkl")
                joblib.dump(selected_features, "selected_features.pkl")
                st.info(f"Model saved as password_strength_{algorithm.lower().replace(' ', '_')}.pkl")
    
    # Password Prediction
    elif page == "Password Prediction":
        st.header("Predict Password Strength")
        
        # Check if models exist
        rf_model_path = "password_strength_random_forest.pkl"
        dt_model_path = "password_strength_decision_tree.pkl"
        features_path = "selected_features.pkl"
        
        models_available = []
        if os.path.exists(rf_model_path):
            models_available.append("Random Forest")
        if os.path.exists(dt_model_path):
            models_available.append("Decision Tree")
        
        if not models_available:
            st.warning("No trained models found. Please go to the 'Model Training' section to train a model first.")
            
            # Train a default model
            st.info("Training a default Random Forest model...")
            
            # Prepare data
            feature_cols = ['Length', 'Has_Lowercase', 'Has_Uppercase', 'Has_Digit', 'Has_Special',
                           'Lowercase_Count', 'Uppercase_Count', 'Digit_Count', 'Special_Count']
            X = dataset[feature_cols]
            y = dataset['Strength']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = train_model(X_train, y_train, model_type="Random Forest")
            joblib.dump(model, rf_model_path)
            joblib.dump(feature_cols, features_path)
            
            models_available = ["Random Forest"]
            
            st.success("Default model trained successfully!")
        
        # Select model
        selected_model = st.selectbox(
            "Select model for prediction",
            options=models_available
        )
        
        # Load model
        model_path = rf_model_path if selected_model == "Random Forest" else dt_model_path
        model = joblib.load(model_path)
        
        # Load selected features
        if os.path.exists(features_path):
            selected_features = joblib.load(features_path)
        else:
            selected_features = ['Length', 'Has_Lowercase', 'Has_Uppercase', 'Has_Digit', 'Has_Special',
                               'Lowercase_Count', 'Uppercase_Count', 'Digit_Count', 'Special_Count']
        
        # Password input
        password = st.text_input("Enter a password to check its strength", type="password")
        
        show_password = st.checkbox("Show password")
        if show_password and password:
            st.text(f"Password: {password}")
        
        if password and st.button("Check Strength"):
            # Extract features
            features = extract_password_features(password)
            
            # Create a DataFrame with the features
            features_df = pd.DataFrame([features])
            
            # Only use selected features for prediction
            features_df = features_df[selected_features]
            
            # Make prediction
            strength = model.predict(features_df)[0]
            
            # Display result
            st.subheader(f"Predicted Strength: {strength}")
            
            # Show strength meter
            display_password_strength_meter(strength)
            
            # Password improvement suggestions
            st.subheader("Password Improvement Suggestions")
            
            suggestions, improved = suggest_stronger_password(password, strength)
            
            if strength != "Strong":
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
                
                st.markdown("### Improved Password Suggestion:")
                st.info(improved)
                
                # Show strength comparison
                improved_features = extract_password_features(improved)
                improved_df = pd.DataFrame([improved_features])
                improved_df = improved_df[selected_features]
                improved_strength = model.predict(improved_df)[0]
                
                st.markdown(f"The improved password would be rated as: **{improved_strength}**")
                st.markdown("### Improved Password Strength Meter:")
                display_password_strength_meter(improved_strength)
            else:
                st.success("Your password is already strong!")
            
            # Compare with common patterns
            if len(password) < 8:
                st.warning("Password is too short. Cybersecurity experts recommend at least 8 characters.")
            
            common_passwords = ["123456", "password", "qwerty", "admin", "welcome", "123456789", "12345"]
            if password.lower() in common_passwords:
                st.error("This is an extremely common password! It would be cracked instantly.")
            
            # Display feature values
            st.subheader("Password Features")
            feature_df = pd.DataFrame({
                'Feature': list(features.keys()),
                'Value': list(features.values())
            })
            st.table(feature_df)

if __name__ == "__main__":
    main()