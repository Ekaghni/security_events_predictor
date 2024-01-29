import pandas as pd

def add_rating_to_csv(csv_file, search_string, rating):
    df = pd.read_csv(csv_file)
    mask = df['Description'].str.contains(search_string, case=False, na=False, regex=False)
    df.loc[mask, 'Severity Level'] = rating
    df.to_csv(csv_file, index=False)
csv_file = 'events_info.csv'
search_string = 'Audit: Command: /usr/bin/echo.'
rating = 5.1

add_rating_to_csv(csv_file, search_string, rating)
