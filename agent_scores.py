import pandas as pd
import joblib

# Load the trained model
loaded_model = joblib.load('trained_model.joblib')

# Load your CSV data with 3 columns: 'Agent Name', 'Description', 'Severity Level'
data = pd.read_csv('output_file.csv')

# Drop rows with NaN values in the 'Description' column
data = data.dropna(subset=['Description'])

# Make predictions on all descriptions
predicted_severity = loaded_model.predict(data['Description'].astype(str))

# Add the predicted severity as a new column to the dataframe
data['Predicted Severity'] = predicted_severity

# Calculate average severity score for each agent
average_scores = data.groupby('Agent Name')['Predicted Severity'].mean().reset_index()

# Display the average scores
print(average_scores)
