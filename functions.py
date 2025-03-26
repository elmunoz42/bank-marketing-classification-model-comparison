import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve, auc, classification_report
import requests
import json
import os
from dotenv import load_dotenv

def plot_dataframe_missing_values(dataframe, filepath, custom_na_values=None, custom_title='Missing Value Proportion per Column'):
    """
    Create and save a bar plot showing the proportion of missing values in each column.
    Handles both explicit NaN values and custom missing value indicators.
    
    Parameters:
    dataframe (pd.DataFrame): The dataframe to analyze
    filepath (str): Path where the plot should be saved
    custom_na_values (dict, optional): Dictionary mapping column names to values that 
                                      should be considered as missing, e.g., {'pdays': [999]}
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = dataframe.copy()
    
    # Replace custom NA values with NaN
    if custom_na_values:
        for col, values in custom_na_values.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].replace(values, np.nan)
    
    # Calculate missing values
    missing_values = df_copy.isna().mean().sort_values(ascending=False)
    
    # Only keep columns with at least some missing values
    missing_values = missing_values[missing_values > 0]
    
    # Check if we have any missing values to plot
    if len(missing_values) == 0:
        print("No missing values detected in the dataframe.")
        return
    
    # Create the bar plot
    plt.figure(figsize=(12, 6))
    missing_plot = missing_values.plot(kind='bar')
    
    # Add value labels on top of each bar
    for i, v in enumerate(missing_values):
        plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
    
    # Customize the plot
    plt.xlabel('Columns')
    plt.ylabel('Proportion of Missing Values')
    plt.title(custom_title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot first (before showing it)
    plt.savefig(filepath)
    
    # Show the plot
    plt.show()
    
    # Print summary information
    print(f"Missing value analysis saved to {filepath}")
    print(f"Total columns with missing values: {len(missing_values)}")
    if len(missing_values) > 0:
        print(f"Column with most missing values: {missing_values.index[0]} ({missing_values.iloc[0]:.2%})")

def calculate_percentage_difference(value1, value2):
    """
    Calculate the percentage difference between two values using NumPy.
    
    Args:
    value1 (float): First value
    value2 (float): Second value
    
    Returns:
    float: Percentage difference between the two values
    """
    return np.abs(value1 - value2) / np.mean([value1, value2]) * 100

def calculate_percentage_change(original_value, new_value):
    """
    Calculate the percentage change between two values using NumPy.
    
    Args:
    original_value (float): The original value
    new_value (float): The new value
    
    Returns:
    float: Percentage change from the original value to the new value
    """
    return (new_value - original_value) / original_value * 100

def identify_feature_types(df):
    # Identify numeric features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Identify categorical features
    # This includes object dtype and any integer column with low cardinality
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Check for integer columns that might be categorical (e.g., ordinal data)
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].nunique() < 10:  # Adjust this threshold as needed
            categorical_features.append(col)
            numeric_features.remove(col)
    
    # Remove the target variable if it's in either list
    target_variable = 'price'  # Adjust this to your target variable name
    if target_variable in numeric_features:
        numeric_features.remove(target_variable)
    if target_variable in categorical_features:
        categorical_features.remove(target_variable)
    
    return numeric_features, categorical_features

def mse_for_different_degrees(X,y,range_low, range_stop):
    mses = []
    for i in range(range_low,range_stop):
    #for 1, 2, 3, ...

        #create pipeline
        pipeline = Pipeline([
            ('features', PolynomialFeatures(degree=i, include_bias=False)),
            ('model', LinearRegression())
        ])
        #fit pipeline
        pipeline.fit(X,y)
        #make predictions
        predictions = pipeline.predict(X)
        #compute mse
        mse = mean_squared_error(y,predictions)
        #append mse to mses
        mses.append(mse)
        
    return mses


def describe_dataset_with_claude(df, api_key, model='claude-3-opus-20240229', 
                                version='2023-06-01', custom_prompt=None, 
                                column_description=None, column_info_file=None):
    """
    Function to analyze a dataset using pandas describe() and then interpret the results using Claude API.
    
    Parameters:
    df (pandas.DataFrame): The dataset to analyze
    api_key (str): Anthropic API key
    model (str, optional): Claude model to use (default: 'claude-3-opus-20240229')
    version (str, optional): Anthropic API version (default: '2023-06-01')
    custom_prompt (str, optional): Custom instructions for Claude
    column_description (str, optional): Description of columns to provide context to Claude
    column_info_file (str, optional): Path to a text file containing column descriptions
    
    Returns:
    str: Claude's interpretation of the dataset statistics
    """
    # Get basic statistics
    stats = df.describe(include='all')
    
    # Get column data types
    dtypes = pd.DataFrame(df.dtypes, columns=['Data Type'])
    
    # Calculate null values
    null_counts = pd.DataFrame(df.isnull().sum(), columns=['Null Count'])
    null_percentages = pd.DataFrame(df.isnull().mean() * 100, columns=['Null Percentage'])
    
    # Count unique values for each column
    unique_counts = pd.DataFrame(df.nunique(), columns=['Unique Values'])
    
    # Combine all information
    dataset_info = {
        'statistics': stats.to_dict(),
        'data_types': dtypes.to_dict(),
        'null_info': {
            'counts': null_counts.to_dict(),
            'percentages': null_percentages.to_dict()
        },
        'unique_values': unique_counts.to_dict(),
        'sample_data': df.head(5).to_dict()
    }
    
    # Default prompt if none is provided
    if custom_prompt is None:
        custom_prompt = """
        You are an expert data analyst. Based on the dataset statistics provided, give me a concise, 
        human-readable interpretation of the key characteristics of this dataset. Focus on:
        
        1. The typical values and ranges for numerical columns
        2. The distribution of categorical columns
        3. Any potential issues with the data (e.g., missing values, outliers)
        4. Any interesting patterns or insights
        
        Format your response as bullet points that are easy to read and understand.
        Make your insights actionable for a business context.
        """
    
    # Get column description from file if specified
    if column_info_file and os.path.exists(column_info_file):
        with open(column_info_file, 'r') as f:
            column_description = f.read()
    
    # Add column description to the prompt if provided
    column_info = ""
    if column_description:
        column_info = f"""
        Here is additional information about the columns in this dataset:
        {column_description}
        
        Please use this information to better understand the context and meaning of each column.
        """
    
    # Prepare the prompt for Claude
    prompt = f"""
    {custom_prompt}
    
    {column_info}
    
    Here is the statistical summary of the dataset:
    {json.dumps(dataset_info, default=str, indent=2)}
    """
    
    # API endpoint for Claude
    api_url = "https://api.anthropic.com/v1/messages"
    
    # Prepare the request
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": version
    }
    
    data = {
        "model": model,
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    # Make the API request
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        
        # Extract Claude's response
        claude_response = response.json()
        interpretation = claude_response['content'][0]['text']
        
        return interpretation
    
    except Exception as e:
        return f"Error calling Claude API: {str(e)}"
