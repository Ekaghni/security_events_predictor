import pandas as pd
import numpy as np

# Replace 'your_file.csv' with the actual path to your CSV file
input_file_path = 'events_info.csv'
output_file_path = 'output_file.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file_path)

# Replace empty cells in the "Severity Level" column with random numbers
df['Severity Level'] = df['Severity Level'].apply(lambda x: np.random.uniform(0.0, 10.0) if pd.isna(x) else x)

# Save the modified DataFrame to a new CSV file
df.to_csv(output_file_path, index=False)

print(f"Empty cells in 'Severity Level' column replaced with random numbers and saved to {output_file_path}.")
