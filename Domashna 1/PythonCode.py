import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import locale
import os

# Set locale for formatting
locale.setlocale(locale.LC_ALL, 'mk_MK.UTF-8')

def date_ranges(start_date, end_date, days=364):
    """Generate ranges from start to end date with given day intervals."""
    start = datetime.strptime(start_date, "%d.%m.%Y")
    end = datetime.strptime(end_date, "%d.%m.%Y")
    while start < end:
        next_end = min(start + timedelta(days=days), end)
        yield start.strftime("%d.%m.%Y"), next_end.strftime("%d.%m.%Y")
        start = next_end + timedelta(days=1)

def get_all_symbols():
    """Retrieve all valid issuer symbols from the North Macedonia Stock Exchange site."""
    url = "https://www.mse.mk/mk/stats/symbolhistory/REPL"
    response = requests.get(url)
    symbols = []

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        dropdown = soup.find("select", {"name": "Code"})
        options = dropdown.find_all("option")

        for option in options:
            symbol = option['value']
            if not any(char.isdigit() for char in symbol):
                symbols.append(symbol)

    return symbols

def load_existing_data():
    """Load existing data from CSV and find the last date for each issuer."""
    filename = "all_historical_data_final.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename, dayfirst=True)
        df['Датум'] = pd.to_datetime(df['Датум'], errors='coerce', format='%d.%m.%Y')
        df = df.dropna(subset=['Датум'])
        return df
    else:
        return pd.DataFrame()

def scrape_data_for_symbol(symbol, start_date, end_date):
    """Scrape data for a single symbol between the given date range."""
    base_url = "https://www.mse.mk/mk/stats/symbolhistory/REPL"
    all_data = []

    for from_date, to_date in date_ranges(start_date, end_date):
        print(f"Scraping {symbol} from {from_date} to {to_date}...")

        payload = {
            "FromDate": from_date,
            "ToDate": to_date,
            "Code": symbol
        }
        response = requests.post(base_url, data=payload)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find('table', {'id': 'resultsTable'})
            if table:
                for row in table.find_all('tr')[1:]:
                    cols = [col.text.strip() for col in row.find_all('td')]
                    all_data.append([symbol] + cols)
                print(f"Data found for {from_date} to {to_date}")
                time.sleep(1)
            else:
                print(f"No data found in table for {from_date} to {to_date}.")
        else:
            print(f"Failed to retrieve data for {from_date} to {to_date}. Status code: {response.status_code}")

    return all_data

def save_data(all_data, existing_df):
    """Combine and save all issuer data to a single CSV file."""
    df_new = pd.DataFrame(all_data, columns=[
        'Код на издавач', 'Датум', 'Цена на последна трансакција', 'Мак.', 'Мин.',
        'Просечна цена', '%пром.', 'Количина', 'Промет во БЕСТ во денари',
        'Вкупен промет во денари'
    ])
    df_new['Датум'] = pd.to_datetime(df_new['Датум'], format='%d.%m.%Y')
    combined_df = pd.concat([existing_df, df_new], ignore_index=True).drop_duplicates(
        subset=['Код на издавач', 'Датум']).sort_values(['Код на издавач', 'Датум'])

    for col in ['Промет во БЕСТ во денари', 'Вкупен промет во денари']:
        combined_df[col] = combined_df[col].apply(format_value)

    combined_df.to_csv("all_historical_data_final.csv", index=False, encoding='utf-8-sig')
    print("Data saved to all_historical_data_final.csv")

def format_value(x):
    if isinstance(x, str) and '.' in x:
        return locale.format_string('%.2f', float(x.replace('.', '')), grouping=True)
    elif isinstance(x, str):
        return locale.format_string('%d', int(x.replace('.', '')), grouping=True)
    return x

# New function to store last run date
def save_last_run_date(date):
    with open("last_run_date.txt", "w") as file:
        file.write(date)

# New function to load last run date
def load_last_run_date():
    if os.path.exists("last_run_date.txt"):
        with open("last_run_date.txt", "r") as file:
            return file.read().strip()
    return None

# Main Process
start_time = time.time()  # Start timer

# Load the last run date or set default start date
last_run_date = load_last_run_date()
start_date = last_run_date if last_run_date else "03.11.2014"  # Default start date

end_date = datetime.today().strftime("%d.%m.%Y")
symbols = get_all_symbols()
existing_df = load_existing_data()
all_data = []

if 'Код на издавач' not in existing_df.columns:
    print("Warning: 'Код на издавач' column is missing from the existing data. Please check your CSV file.")

for symbol in symbols:
    symbol_data = existing_df[existing_df['Код на издавач'] == symbol] if 'Код на издавач' in existing_df.columns else pd.DataFrame()

    if not symbol_data.empty:
        last_date = symbol_data['Датум'].max().strftime("%d.%m.%Y")
        start_date_for_symbol = (datetime.strptime(last_date, "%d.%m.%Y") + timedelta(days=1)).strftime("%d.%m.%Y")
    else:
        start_date_for_symbol = start_date

    new_data = scrape_data_for_symbol(symbol, start_date_for_symbol, end_date)
    if new_data:
        all_data.extend(new_data)

if all_data:
    save_data(all_data, existing_df)

# Save the current date as the last run date
save_last_run_date(datetime.today().strftime("%d.%m.%Y"))

end_time = time.time()  # End timer
execution_time = end_time - start_time  # Calculate execution time
print(f"Execution time: {execution_time:.2f} seconds")