# Password Strength Checker

A practical tool to evaluate and improve password security using machine learning.

## What It Does

- Analyzes passwords for security weaknesses
- Classifies strength as Weak, Medium, or Strong
- Provides instant feedback and improvement tips
- Includes a secure password generator

## How It Works

1. Enter any password
2. The system checks multiple security factors:
   - Length (8+ characters recommended)
   - Mix of uppercase/lowercase letters
   - Numbers and special characters
   - Common pattern detection
3. Get immediate strength assessment
4. Receive customized improvement suggestions

## Technical Details

**Backend:**
- Python 3
- Random Forest machine learning model
- Regular expressions for pattern checking

**Frontend:**
- Streamlit web interface
- Interactive strength meter
- Clean, responsive design

## Setup Instructions

1. Install requirements:
```bash
pip install streamlit pandas scikit-learn
```

2. Run the application:
```bash
streamlit run password_app.py
```

## Features

✔️ Real-time strength analysis  
✔️ Detailed security breakdown  
✔️ One-click password generation  
✔️ No data collection - runs locally  

## Why Use This?

Unlike basic checkers, this tool:
- Uses actual machine learning
- Provides specific improvement advice
- Works completely offline
- Generates truly secure passwords

---
