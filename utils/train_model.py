import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv('data/passwords.csv')
X = np.vstack(df['password'].apply(extract_features))
y = df['strength']

# Train
model = RandomForestClassifier()
model.fit(X, y)

# Save
joblib.dump(model, 'models/password_model.pkl')