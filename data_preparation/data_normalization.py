import pandas as pd
import os
import re
from datetime import datetime


def normalize_data(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file, encoding='ANSI')

    # Function to clean numeric values (remove commas and convert to numeric)
    def clean_numeric(val):
        if pd.isna(val):
            return None
        if isinstance(val, str):
            return pd.to_numeric(val.replace(',', ''), errors='coerce')
        return val

    # Function to normalize time format
    def normalize_time(time_str):
        if pd.isna(time_str):
            return None

        # Handle different time formats
        if isinstance(time_str, str):
            # Format like "00:30:41"
            if re.match(r'\d+:\d+:\d+', time_str):
                parts = time_str.split(':')
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2])
                return hours * 3600 + minutes * 60 + seconds

            # Format like "0:08:12"
            elif re.match(r'\d+:\d+:\d+', time_str):
                parts = time_str.split(':')
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2])
                return hours * 3600 + minutes * 60 + seconds

            # Format like "0:01:06"
            elif re.match(r'\d+:\d+:\d+', time_str):
                parts = time_str.split(':')
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2])
                return hours * 3600 + minutes * 60 + seconds

        return clean_numeric(time_str)

    # Function to normalize dates
    def normalize_date(date_str):
        if pd.isna(date_str) or date_str == 'N/A':
            return None

        try:
            # Try to parse date in MM/DD/YYYY format
            return datetime.strptime(date_str, '%m/%d/%Y').strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            return date_str

    # Normalize numeric columns
    numeric_columns = [
        'Video Views',
        'Total Channel Subcribers',
        'Total Chanel Views',
        'Duration in Seconds',
        'No of Likes',
        'No of Comments',
        'No of Videos the Channel',
        'No of Playlist'
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    # Normalize time columns
    df['Duration of Video'] = df['Duration of Video'].apply(normalize_time)

    # Make sure Duration in Seconds is consistent
    df['Duration in Seconds'] = df['Duration in Seconds'].apply(clean_numeric)

    # Normalize date columns
    date_columns = ['Date of Video Upload', 'Date of the Last Comment']
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_date)

    # Convert boolean-like columns to actual booleans
    boolean_columns = ['Subtitle', 'Video Description', 'Premiered or Not']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': True, 'No': False, 'N/A': None})

    # Save the normalized data to a new CSV file
    df.to_csv(output_file, index=False, encoding='utf-8')

    return df


if __name__ == "__main__":
    # Get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define file paths
    input_file = os.path.join(script_dir, 'data.csv')
    output_file = os.path.join(script_dir, 'normalized_data.csv')

    # Normalize the data
    normalized_df = normalize_data(input_file, output_file)

    print(f"Data normalization complete. Normalized data saved to {output_file}")
    print(f"First 5 rows of normalized data:")
    print(normalized_df.head())