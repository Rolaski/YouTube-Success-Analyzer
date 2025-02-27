from pyArango.connection import Connection
from dotenv import load_dotenv
import os

# Załaduj zmienne z pliku .env
load_dotenv()

# Pobierz dane z pliku .env
username = os.getenv("ARANGO_USERNAME")
password = os.getenv("ARANGO_PASSWORD")
database_name = os.getenv("ARANGO_DATABASE")

# Nawiązanie połączenia z bazą danych ArangoDB
conn = Connection(username=username, password=password)
db = conn[database_name]

# Pobranie wszystkich kolekcji
collections = db.collections

# Usunięcie każdej kolekcji, która nie jest systemowa
for collection_name in collections:
    collection = db[collection_name]
    if not collection.properties()["isSystem"]:
        collection.delete()
        print(f"Collection removed: {collection_name}")

print("All collections removed.")