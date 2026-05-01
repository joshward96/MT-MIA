import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def generate_t1_child_data(
    num_t1s=500,              # Number of t1 records
    t1_dims=8,                # Dimensions for t1 data
    child_dims=8,                 # Dimensions for child data
    min_children=1,               # Minimum children per t1
    max_children=1,               # Maximum children per t1
    t1_mean=None,             # Mean vector for t1 distribution
    t1_cov=None,              # Covariance matrix for t1 distribution
    child_mean=None,              # Mean vector for child distribution
    child_cov=None,               # Covariance matrix for child distribution
    output_t1_file='t1_data.csv',  # Output file for t1 data
    output_child_file='child_data.csv'     # Output file for child data
):
    """
    Generate multivariate Gaussian data for t1 and child tables.
    
    Parameters:
    -----------
    num_t1s : int
        Number of t1 records to generate
    t1_dims : int
        Number of dimensions (features) for t1 data
    child_dims : int
        Number of dimensions (features) for child data
    min_children, max_children : int
        Range for uniform distribution of number of children per t1
    t1_mean, child_mean : array-like or None
        Mean vectors for t1 and child distributions
        If None, zeros will be used
    t1_cov, child_cov : array-like or None
        Covariance matrices for t1 and child distributions
        If None, identity matrices will be used
    output_t1_file, output_child_file : str
        Filenames for output CSV files
    
    Returns:
    --------
    tuple : (t1_df, child_df)
        DataFrames containing the generated data
    """
    # Set default means and covariances if not provided
    if t1_mean is None:
        t1_mean = np.zeros(t1_dims)
    if t1_cov is None:
        t1_cov = np.eye(t1_dims)
    if child_mean is None:
        child_mean = np.ones(child_dims)
    if child_cov is None:
        child_cov = np.eye(child_dims)
    
    # Generate t1 data from multivariate Gaussian
    t1_data = multivariate_normal.rvs(
        mean=t1_mean, 
        cov=t1_cov, 
        size=num_t1s
    )
    
    # Create t1 DataFrame
    t1_df = pd.DataFrame(
        t1_data, 
        columns=[f't1_feature_{i+1}' for i in range(t1_dims)]
    )
    t1_df['t1_id'] = np.arange(1, num_t1s + 1)
    
    # Generate child data
    child_rows = []
    child_id = 1
    
    for t1_id in t1_df['t1_id']:
        # Random number of children from uniform distribution
        num_children = np.random.randint(min_children, max_children + 1)
        
        # Generate children data from multivariate Gaussian
        children_data = multivariate_normal.rvs(
            mean=child_mean, 
            cov=child_cov, 
            size=num_children
        )
        
        # Handle case where num_children = 1 to ensure proper 2D shape
        if num_children == 1:
            children_data = children_data.reshape(1, -1)
        
        # Add each child to the list of child rows
        for i in range(num_children):
            child_row = {'child_id': child_id, 't1_id': t1_id}
            for j in range(child_dims):
                child_row[f'child_feature_{j+1}'] = children_data[i, j]
            child_rows.append(child_row)
            child_id += 1
    
    # Create child DataFrame
    child_df = pd.DataFrame(child_rows).drop(columns='child_id')
    child_df['t2_id'] = child_df.index

    # Save to CSV files
    #t1_df.to_csv(output_t1_file, index=False)
    #child_df.to_csv(output_child_file, index=False)
    
    return t1_df, child_df
t1_train,child_train =generate_t1_child_data(min_children=100,               # Minimum children per t1
                                                     max_children=100,              )
t1_holdout,child_holdout =generate_t1_child_data()
t1_ref,child_ref =generate_t1_child_data()
t1_synth,child_synth =generate_t1_child_data(min_children=100,               # Minimum children per t1
                                                     max_children=100,              )

dirs = [
    "data/toy/mem",
    "data/toy/non_mem",
    "data/toy/ref",
    "data/toy/synth",
]

for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
t1_train.to_csv("data/toy/mem/t1.csv", index=False)
t1_holdout.to_csv("data/toy/non_mem/t1.csv", index=False)
t1_ref.to_csv("data/toy/ref/t1.csv", index=False)
t1_synth.to_csv("data/toy/synth/t1.csv", index=False)

child_train.to_csv("data/toy/mem/t2.csv", index=False)
child_holdout.to_csv("data/toy/non_mem/t2.csv", index=False)
child_ref.to_csv("data/toy/ref/t2.csv", index=False)
child_synth.to_csv("data/toy/synth/t2.csv", index=False)