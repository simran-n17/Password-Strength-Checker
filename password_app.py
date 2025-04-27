import streamlit as st
import pandas as pd
import joblib
import re
import string
from password_dataset import PasswordDatasetGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# App Configuration
st.set_page_config(
    page_title="Password Strength Checker",
    layout="wide"
)

class PasswordAnalyzer:
    """Core password analysis functionality"""
    
    @staticmethod
    def extract_features(password: str) -> dict:
        """Extract security features from password"""
        return {
            'length': len(password),
            'has_lower': int(bool(re.search(r'[a-z]', password))),
            'has_upper': int(bool(re.search(r'[A-Z]', password))),
            'has_digit': int(bool(re.search(r'\d', password))),
            'has_special': int(bool(re.search(r'[^a-zA-Z0-9]', password)))
        }
    
    @staticmethod
    def generate_strong_password(length: int = 12) -> str:
        """Generate a random strong password"""
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        while True:
            password = ''.join(random.choices(chars, k=length))
            features = PasswordAnalyzer.extract_features(password)
            if (features['has_lower'] and features['has_upper'] 
                and features['has_digit'] and features['has_special']):
                return password

class PasswordApp:
    """Streamlit application class"""
    
    def __init__(self):
        self.model = None
        self.dataset = None
    
    def load_model(self):
        """Load or train model"""
        try:
            self.model = joblib.load("password_model.pkl")
        except:
            st.warning("Training new model...")
            self.train_model()
    
    def train_model(self):
        """Train and save new model"""
        generator = PasswordDatasetGenerator()
        df = generator.generate_dataset(5000)
        
        X = df[['length', 'has_lower', 'has_upper', 'has_digit', 'has_special']]
        y = df['strength']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, "password_model.pkl")
        st.success("Model trained successfully!")
    
    def run(self):
        """Main application flow"""
        st.title("Password Strength Checker")
        
        tab1, tab2, tab3 = st.tabs(["Analyzer", "Generator", "Security Tips"])
        
        with tab1:
            self.analyzer_tab()
        with tab2:
            self.generator_tab()
        with tab3:
            self.tips_tab()
    
    def analyzer_tab(self):
        """Password analysis interface"""
        st.header("Password Strength Analysis")
        password = st.text_input("Enter a password:", type="password")
        
        if password:
            features = pd.DataFrame([PasswordAnalyzer.extract_features(password)])
            strength = self.model.predict(features)[0]
            
            # Visual feedback
            color = {"Weak": "red", "Medium": "orange", "Strong": "green"}[strength]
            st.markdown(f"### Strength: <span style='color:{color}'>{strength}</span>", 
                      unsafe_allow_html=True)
            
            # Feature breakdown
            st.subheader("Security Features")
            st.dataframe(features.T.rename(columns={0: "Value"}))
            
            # Improvement suggestions
            if strength != "Strong":
                st.warning("Improvement suggestions:")
                if features['length'][0] < 12:
                    st.write("- Use at least 12 characters")
                if not features['has_special'][0]:
                    st.write("- Add special characters (!@#$)")
    
    def generator_tab(self):
        """Password generator interface"""
        st.header("Strong Password Generator")
        length = st.slider("Password length", 8, 32, 16)
        
        if st.button("Generate Password"):
            password = PasswordAnalyzer.generate_strong_password(length)
            st.code(password)
            st.download_button(
                "Download Password",
                data=password,
                file_name="generated_password.txt"
            )
    
    def tips_tab(self):
        """Security recommendations"""
        st.header("Password Security Tips")
        tips = [
            "🔒 Use at least 12-16 characters",
            "🔀 Mix uppercase, lowercase, numbers and symbols",
            "🚫 Avoid personal information or common words",
            "🔄 Use unique passwords for each account",
            "📌 Consider using a password manager"
        ]
        for tip in tips:
            st.markdown(f"- {tip}")

if __name__ == "__main__":
    app = PasswordApp()
    app.load_model()
    app.run()