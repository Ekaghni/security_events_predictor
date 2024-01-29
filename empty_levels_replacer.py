import pandas as pd
import numpy as np

input_file_path = 'events_info.csv'
output_file_path = 'output_file.csv'
df = pd.read_csv(input_file_path)
df['Severity Level'] = df['Severity Level'].apply(lambda x: np.random.uniform(0.0, 10.0) if pd.isna(x) else x)
df.to_csv(output_file_path, index=False)
print(f"saved to {output_file_path}.")
