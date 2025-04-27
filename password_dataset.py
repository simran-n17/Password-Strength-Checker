import pandas as pd
import numpy as np
import random
import string
import re
from typing import List, Dict

class PasswordDatasetGenerator:
    """Generates synthetic password datasets with strength labels"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def _generate_weak_passwords(self, n: int) -> List[str]:
        """Generate common weak password patterns"""
        weak_passwords = []
        common_words = ['password', 'welcome', 'admin', 'qwerty', 'abc123']
        
        for _ in range(n):
            pattern = random.choice([
                # Short numeric
                ''.join(random.choices(string.digits, k=random.randint(4, 6))),
                # Common word with simple suffix
                random.choice(common_words) + random.choice(['', '1', '123', '!']),
                # Name with digits
                random.choice(['john', 'mary', 'david']) + ''.join(random.choices(string.digits, k=2))
            ])
            weak_passwords.append(pattern)
        return weak_passwords
    
    def _generate_medium_passwords(self, n: int) -> List[str]:
        """Generate medium strength passwords"""
        medium_passwords = []
        for _ in range(n):
            length = random.randint(7, 10)
            if random.random() > 0.5:
                # Mixed characters
                password = ''.join(random.choices(string.ascii_lowercase, k=length-2))
                password += ''.join(random.choices(string.digits, k=2))
                if random.random() > 0.5:
                    password = password.capitalize()
            else:
                # Word with numbers
                word = random.choice(['sunshine', 'winter', 'coffee'])
                password = word + str(random.randint(100, 9999))
            medium_passwords.append(password)
        return medium_passwords
    
    def _generate_strong_passwords(self, n: int) -> List[str]:
        """Generate cryptographically strong passwords"""
        strong_passwords = []
        special_chars = '!@#$%^&*'
        
        for _ in range(n):
            length = random.randint(12, 16)
            components = [
                ''.join(random.choices(string.ascii_lowercase, k=4)),
                ''.join(random.choices(string.ascii_uppercase, k=3)),
                ''.join(random.choices(string.digits, k=3)),
                ''.join(random.choices(special_chars, k=2))
            ]
            password = ''.join(components)
            password = ''.join(random.sample(password, len(password)))  # Shuffle
            strong_passwords.append(password)
        return strong_passwords
    
    def generate_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate complete password dataset"""
        n_weak = int(n_samples * 0.3)
        n_medium = int(n_samples * 0.4)
        n_strong = n_samples - n_weak - n_medium
        
        passwords = (
            self._generate_weak_passwords(n_weak) +
            self._generate_medium_passwords(n_medium) +
            self._generate_strong_passwords(n_strong)
        )
        
        strengths = ['Weak']*n_weak + ['Medium']*n_medium + ['Strong']*n_strong
        
        return self._add_features(pd.DataFrame({
            'password': passwords,
            'strength': strengths
        }))
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add password features to dataframe"""
        df['length'] = df['password'].str.len()
        df['has_lower'] = df['password'].str.contains(r'[a-z]').astype(int)
        df['has_upper'] = df['password'].str.contains(r'[A-Z]').astype(int)
        df['has_digit'] = df['password'].str.contains(r'\d').astype(int)
        df['has_special'] = df['password'].str.contains(r'[^a-zA-Z0-9]').astype(int)
        return df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

def save_dataset(filepath: str = "password_dataset.csv", n_samples: int = 5000):
    """Generate and save dataset"""
    generator = PasswordDatasetGenerator()
    df = generator.generate_dataset(n_samples)
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath} with {n_samples} samples")
    return df

if __name__ == "__main__":
    save_dataset()