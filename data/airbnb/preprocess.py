import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Load the CSV files
sessions = pd.read_csv('original/sessions.csv').dropna()
users = pd.read_csv('original/users.csv').dropna()

# Drop specified columns from users
users = users.drop(columns=['date_account_created', 'timestamp_first_active', 'date_first_booking'])

# Rename 'id' to 'user_id' in users
users = users.rename(columns={'id': 'user_id'})

# Select 3000 random users
np.random.seed(42)  # For reproducibility
sampled_users = users.sample(n=min(10000, len(users)), random_state=42)
sampled_user_ids = sampled_users['user_id'].unique()

# Filter sessions to only include the sampled users, then sample up to 10 sessions per user
sessions_filtered = sessions[sessions['user_id'].isin(sampled_user_ids)]
sessions_sampled = sessions_filtered.groupby('user_id').apply(
    lambda x: x.sample(n=min(10, len(x)), random_state=42)
).reset_index(drop=True)

# **KEY CHANGE: Filter users to only those with sessions**
users_with_sessions = sessions_sampled['user_id'].unique()
users = sampled_users[sampled_users['user_id'].isin(users_with_sessions)]

# Update sessions to match the filtered users
sessions = sessions_sampled

# Convert string user_id to numeric while preserving joins
unique_user_ids = pd.concat([users['user_id'], sessions['user_id']]).unique()
user_id_mapping = {str_id: num_id for num_id, str_id in enumerate(unique_user_ids, start=1)}

# Apply the mapping to both tables
users['user_id'] = users['user_id'].map(user_id_mapping)
sessions['user_id'] = sessions['user_id'].map(user_id_mapping)

# Add session_id as row number to sessions
sessions.insert(0, 'session_id', range(1, len(sessions) + 1))

# Ordinal encode string columns in both tables
def ordinal_encode_strings(df):
    """Apply ordinal encoding to all string/object columns in a dataframe"""
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if string_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[string_cols] = encoder.fit_transform(df[string_cols])
    
    return df

# Apply ordinal encoding
users = ordinal_encode_strings(users)
sessions = ordinal_encode_strings(sessions)

# Save the transformed data
users.to_csv('user.csv', index=False)
sessions.to_csv('session.csv', index=False)

print("Transformation complete!")
print(f"\nUsers shape: {users.shape}")
print(f"Sessions shape: {sessions.shape}")
print(f"\nUsers columns: {users.columns.tolist()}")
print(f"Sessions columns: {sessions.columns.tolist()}")
print(f"\nUsers dtypes:\n{users.dtypes}")
print(f"\nSessions dtypes:\n{sessions.dtypes}")

# Verify each user has at least one session
print(f"\n✓ All {len(users)} users have at least one session")
print(f"✓ Average sessions per user: {len(sessions) / len(users):.2f}")