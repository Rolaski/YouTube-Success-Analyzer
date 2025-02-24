import csv
from pyArango.connection import *
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

# Dostęp do istniejących kolekcji
videos = db["Videos"]
creators = db["Creators"]
has_creator = db["HasCreator"]

# Wczytywanie danych z pliku CSV
with open('data.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)

    for row in csv_reader:
        # Tworzenie dokumentu dla twórcy
        creator = creators.createDocument()
        creator["Name"] = row["Creator Name"]
        creator["Gender"] = row["Creator Gender"]
        creator["Total Channel Subscribers"] = row["Total Channel Subcribers"]
        creator["Total Channel Views"] = row["Total Chanel Views"]
        creator["Channel URL"] = row["Channel URL"]
        creator["Intern Who Collected the Data"] = row["Intern Who Collected the Data"]
        creator.save()

        # Tworzenie dokumentu dla filmu
        video = videos.createDocument()
        video["Link"] = row["Video Link"]
        video["Views"] = row["Video Views"]
        video["Title"] = row["Video Title"]
        video["Duration"] = row["Duration of Video"]
        video["Date of Upload"] = row["Date of Video Upload"]
        video["Language"] = row["Language of the Video"]
        video["Likes"] = row["No of Likes"]
        video["Comments"] = row["No of Comments"]
        video["Hashtags"] = row["Hashtags"]
        video["Max Quality"] = row["Maximum Quality of the Video"]
        video["Premiered"] = row["Premiered or Not"]
        video.save()

        # Dodanie relacji (edge) między twórcą a filmem
        edge = has_creator.createDocument()
        edge._from = creator._id
        edge._to = video._id
        edge.save()

print("Dane zostały załadowane do ArangoDB.")
