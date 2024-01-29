import pandas as pd

def add_rating_to_csv(csv_file, search_string, rating):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Search for the specified string in the "Description" column
    mask = df['Description'].str.contains(search_string, case=False, na=False, regex=False)

    # Add the rating to the "Severity Level" column for matching rows
    df.loc[mask, 'Severity Level'] = rating

    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)

# Example usage
csv_file = 'events_info.csv'
search_string = 'Audit: Command: /usr/bin/echo.'
rating = 5.1

add_rating_to_csv(csv_file, search_string, rating)
