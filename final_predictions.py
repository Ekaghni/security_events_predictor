import joblib
loaded_model = joblib.load('trained_model.joblib')
new_descriptions = ["Audit: Command execution failed.", "New dpkg installed.", "File deleted.","Wazuh agent started.","Host-based anomaly detection event (rootcheck)."]
predicted_severity = loaded_model.predict(new_descriptions)
for desc, severity in zip(new_descriptions, predicted_severity):
    print(f'Description: {desc}\t Predicted Severity: {severity:.2f}')
