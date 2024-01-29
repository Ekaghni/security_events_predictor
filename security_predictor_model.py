import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm
import joblib

# Load your CSV data
data = pd.read_csv('output_file.csv')

# Drop rows with NaN values in the 'Description' column
data = data.dropna(subset=['Description'])

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Define tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Tokenize the input data
def tokenize_data(data, max_length=128):
    input_ids = []
    attention_masks = []

    for description in tqdm(data['Description']):
        encoded_text = tokenizer.encode_plus(
            str(description),  # Convert to string to handle NaN values
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_text['input_ids'])
        attention_masks.append(encoded_text['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

# Tokenize training and testing data
train_input_ids, train_attention_masks = tokenize_data(train_data)
test_input_ids, test_attention_masks = tokenize_data(test_data)

# Create DataLoader for training and testing
train_dataset = TensorDataset(train_input_ids, train_attention_masks, torch.tensor(train_data['Severity Level'].values))
test_dataset = TensorDataset(test_input_ids, test_attention_masks)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Fine-tune the model (example using linear regression, you can use other regression models)
model_regressor = make_pipeline(TfidfVectorizer(), LinearRegression())
model_regressor.fit(train_data['Description'].astype(str), train_data['Severity Level'])

# Save the trained model
joblib.dump(model_regressor, 'trained_model.joblib')

# Evaluate the model
score = model_regressor.score(test_data['Description'].astype(str), test_data['Severity Level'])
print(f'R-squared Score: {score:.2f}')

# Make predictions on new descriptions
new_descriptions = ["Audit: Command execution failed.", "New dpkg installed.", "File deleted.","Wazuh agent started."]
predicted_severity = model_regressor.predict(new_descriptions)

for desc, severity in zip(new_descriptions, predicted_severity):
    print(f'Description: {desc}\t Predicted Severity: {severity:.2f}')
