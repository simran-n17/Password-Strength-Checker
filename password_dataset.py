import pandas as pd
import numpy as np
import random
import string
import re

def generate_password_dataset(n_samples=1000, seed=42):
    """
    Generate a dataset of passwords with features and strength labels
    
    Args:
        n_samples: Number of password samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with passwords and features
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Lists to store data
    passwords = []
    strengths = []
    
    # Generate weak passwords (30% of dataset)
    n_weak = int(n_samples * 0.3)
    for _ in range(n_weak):
        # Common weak patterns
        weak_types = [
            # Short numeric
            ''.join(random.choices(string.digits, k=random.randint(4, 6))),
            # Common words with simple modifications
            random.choice(['password', 'welcome', 'admin', 'qwerty', 'abc123', '123456', 'football', 'baseball', 'dragon']) + 
                random.choice(['', '1', '123', '!']),
            # First name + digits
            random.choice(['john', 'mary', 'david', 'susan', 'mike', 'sarah', 'james', 'linda', 'robert', 'emily']) + 
                ''.join(random.choices(string.digits, k=random.randint(0, 2)))
        ]
        password = random.choice(weak_types)
        passwords.append(password)
        strengths.append('Weak')
    
    # Generate medium passwords (40% of dataset)
    n_medium = int(n_samples * 0.4)
    for _ in range(n_medium):
        # Medium strength patterns
        length = random.randint(7, 10)
        
        # Mix of letters and numbers, maybe one capital
        if random.choice([True, False]):
            password = ''.join(random.choices(string.ascii_lowercase, k=length-2))
            password += ''.join(random.choices(string.digits, k=2))
            if random.choice([True, False]):
                # Capitalize one letter
                idx = random.randint(0, len(password)-1)
                password = password[:idx] + password[idx].upper() + password[idx+1:]
        else:
            # Word + number + maybe special
            word = random.choice(['sunset', 'mountain', 'tiger', 'eagle', 'winter', 'summer', 'office', 'computer', 'coffee', 'music'])
            digits = ''.join(random.choices(string.digits, k=random.randint(2, 4)))
            special = random.choice(['', '!', '@', '#']) if random.random() > 0.5 else ''
            password = word + digits + special
            
        passwords.append(password)
        strengths.append('Medium')
    
    # Generate strong passwords (30% of dataset)
    n_strong = n_samples - n_weak - n_medium
    for _ in range(n_strong):
        # Strong password patterns
        length = random.randint(10, 16)
        
        # Complex pattern with uppercase, lowercase, digits, and special chars
        lowercase_chars = random.randint(max(1, length - 9), length - 3)
        uppercase_chars = random.randint(1, length - lowercase_chars - 2)
        digit_chars = random.randint(1, length - lowercase_chars - uppercase_chars - 1)
        special_chars = length - lowercase_chars - uppercase_chars - digit_chars
        
        password = (
            ''.join(random.choices(string.ascii_lowercase, k=lowercase_chars)) +
            ''.join(random.choices(string.ascii_uppercase, k=uppercase_chars)) +
            ''.join(random.choices(string.digits, k=digit_chars)) +
            ''.join(random.choices('!@#$%^&*()-_=+', k=special_chars))
        )
        
        # Shuffle the password characters
        password_list = list(password)
        random.shuffle(password_list)
        password = ''.join(password_list)
        
        passwords.append(password)
        strengths.append('Strong')
    
    # Create DataFrame
    df = pd.DataFrame({
        'Password': passwords,
        'Strength': strengths
    })
    
    # Add features
    df['Length'] = df['Password'].apply(len)
    df['Has_Lowercase'] = df['Password'].apply(lambda x: 1 if re.search(r'[a-z]', x) else 0)
    df['Has_Uppercase'] = df['Password'].apply(lambda x: 1 if re.search(r'[A-Z]', x) else 0)
    df['Has_Digit'] = df['Password'].apply(lambda x: 1 if re.search(r'\d', x) else 0)
    df['Has_Special'] = df['Password'].apply(lambda x: 1 if re.search(r'[!@#$%^&*()_\-+=\[\]{}|:;,.<>?/~]', x) else 0)
    df['Lowercase_Count'] = df['Password'].apply(lambda x: sum(c.islower() for c in x))
    df['Uppercase_Count'] = df['Password'].apply(lambda x: sum(c.isupper() for c in x))
    df['Digit_Count'] = df['Password'].apply(lambda x: sum(c.isdigit() for c in x))
    df['Special_Count'] = df['Password'].apply(lambda x: sum(not c.isalnum() for c in x))
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df

def save_password_dataset(n_samples=1000, filename="password_strength_dataset.csv"):
    """
    Generate and save a password dataset to CSV
    
    Args:
        n_samples: Number of samples to generate
        filename: Output CSV filename
    """
    df = generate_password_dataset(n_samples)
    df.to_csv(filename, index=False)
    print(f"Dataset with {n_samples} samples saved to {filename}")
    return df

if __name__ == "__main__":
    # Generate and save a dataset with 1000 samples
    save_password_dataset(1000)