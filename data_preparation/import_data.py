import csv
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


# Funkcja tworząca kolekcje, jeśli nie istnieją
def ensure_collections():
    document_collections = ["Videos", "Creators", "Hashtags", "Languages", "Playlists", "UploadDates", "VideoQuality"]
    edge_collections = ["HasCreator", "HasLanguage", "HasHashtag", "BelongsToPlaylist", "UploadedOn", "HasQuality"]

    for col in document_collections:
        if not db.hasCollection(col):
            db.createCollection(name=col)
            print(f"Created document collection: {col}")

    for col in edge_collections:
        if not db.hasCollection(col):
            db.createCollection(name=col, className="Edges")
            print(f"Created edge collection: {col}")


ensure_collections()
print("All collections created")

# Pobranie kolekcji dokumentów
videos = db["Videos"]
creators = db["Creators"]
hashtags = db["Hashtags"]
languages = db["Languages"]
playlists = db["Playlists"]
upload_dates = db["UploadDates"]
video_quality = db["VideoQuality"]

# Pobranie kolekcji edge
has_creator = db["HasCreator"]
has_language = db["HasLanguage"]
has_hashtag = db["HasHashtag"]
belongs_to_playlist = db["BelongsToPlaylist"]
uploaded_on = db["UploadedOn"]
has_quality = db["HasQuality"]

# Wczytywanie danych z pliku CSV
with open('data.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)

    for row in csv_reader:
        # Utworzenie dokumentu dla twórcy
        creator = creators.createDocument()
        creator["Name"] = row["Creator Name"]
        creator["Gender"] = row["Creator Gender"]
        creator["Total Channel Subscribers"] = row["Total Channel Subcribers"]
        creator["Total Channel Views"] = row["Total Chanel Views"]
        creator["Channel URL"] = row["Channel URL"]
        creator.save()

        # Utworzenie dokumentu dla filmu
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
        video["Community Engagement"] = row["Community Engagement (Posts per week)"]
        video.save()

        # Tworzymy krawędź HasCreator – relacja między twórcą a filmem.
        has_creator_edge = has_creator.createDocument()
        has_creator_edge["_from"] = creator._id
        has_creator_edge["_to"] = video._id
        has_creator_edge.save()

        # Utworzenie dokumentu dla języka
        language_doc = languages.createDocument()
        language_doc["Language"] = row["Language of the Video"]
        language_doc.save()

        # Krawędź HasLanguage – łączenie filmu z językiem
        has_language_edge = has_language.createDocument()
        has_language_edge["_from"] = video._id
        has_language_edge["_to"] = language_doc._id
        has_language_edge.save()

        # Utworzenie krawędzi dla hashtagów
        hashtags_list = row["Hashtags"].split()
        for tag in hashtags_list:
            hashtag_doc = hashtags.createDocument()
            hashtag_doc["Hashtag"] = tag
            hashtag_doc.save()
            has_hashtag_edge = has_hashtag.createDocument()
            has_hashtag_edge["_from"] = video._id
            has_hashtag_edge["_to"] = hashtag_doc._id
            has_hashtag_edge.save()

        # Utworzenie krawędzi BelongsToPlaylist, jeśli kolumna "No of Playlist" nie jest pusta
        if row["No of Playlist"]:
            playlist_doc = playlists.createDocument()
            playlist_doc["Playlist"] = row["No of Playlist"]
            playlist_doc.save()
            belongs_edge = belongs_to_playlist.createDocument()
            belongs_edge["_from"] = video._id
            belongs_edge["_to"] = playlist_doc._id
            belongs_edge.save()

        # Utworzenie dokumentu dla daty publikacji
        upload_date_doc = upload_dates.createDocument()
        upload_date_doc["Upload Date"] = row["Date of Video Upload"]
        upload_date_doc.save()
        uploaded_on_edge = uploaded_on.createDocument()
        uploaded_on_edge["_from"] = video._id
        uploaded_on_edge["_to"] = upload_date_doc._id
        uploaded_on_edge.save()

        # Utworzenie dokumentu dla jakości filmu
        quality_doc = video_quality.createDocument()
        quality_doc["Quality"] = row["Maximum Quality of the Video"]
        quality_doc.save()
        has_quality_edge = has_quality.createDocument()
        has_quality_edge["_from"] = video._id
        has_quality_edge["_to"] = quality_doc._id
        has_quality_edge.save()

print("Data loaded to database")
