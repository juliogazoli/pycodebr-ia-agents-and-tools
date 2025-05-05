import requests
import sqlite3
from bs4 import BeautifulSoup


url = 'https://www.idealsoftwares.com.br/indices/ipca_ibge.html'

response = requests.get(url)
html_content = response.content

soup = BeautifulSoup(html_content, 'html.parser')

table = soup.find_all(
    name='table',
    attrs={'class': 'table table-bordered table-striped text-center'},
)[1]

ipca_data = []
for row in table.find_all('tr')[1:]:
    cols = row.find_all('td')
    if cols:
        month_year = cols[0].text.strip()
        value = cols[1].text.strip()\
                            .replace(',', '.')\
                            .replace(' ', '').replace('\n', '')
        if value:
            month, year = month_year.split('/')
            ipca_data.append((float(value), month, int(year)))


conn = sqlite3.connect('ipca.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS IPCA (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    value REAL,
    month TEXT,
    year INTEGER,
    UNIQUE(month, year)
)
''')

for data in ipca_data:
    value, month, year = data
    cursor.execute('''
    INSERT OR IGNORE INTO IPCA (value, month, year)
    VALUES (?, ?, ?)
    ''', (value, month, year))

conn.commit()
conn.close()

print('Dados históricos do IPCA salvos com sucesso!')
