import pandas as pd
import joblib

loaded_model = joblib.load('trained_model.joblib')
data = pd.read_csv('output_file.csv')
data = data.dropna(subset=['Description'])
predicted_severity = loaded_model.predict(data['Description'].astype(str))
data['Predicted Severity'] = predicted_severity
average_scores = data.groupby('Agent Name')['Predicted Severity'].mean().reset_index()
print(average_scores)
