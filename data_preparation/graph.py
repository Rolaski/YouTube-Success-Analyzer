import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import re
import hashlib
from arango import ArangoClient
import os
from dotenv import load_dotenv

# Załaduj plik .env z folderu /database
env_path = os.path.join(os.path.dirname(__file__), '../database', '.env')

# Ładowanie zmiennych środowiskowych
load_dotenv(env_path)
ARANGO_USERNAME = os.getenv("ARANGO_USERNAME")
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD")
ARANGO_DATABASE = os.getenv("ARANGO_DATABASE")

# Stała ścieżka do zapisanego grafu HTML
SAVED_GRAPH_PATH = "saved_graph.html"


# Pobranie danych z bazy ArangoDB
def get_graph_data():
    try:
        client = ArangoClient(hosts='http://localhost:8529')
        db = client.db(ARANGO_DATABASE, username=ARANGO_USERNAME, password=ARANGO_PASSWORD)

        query_creators = "FOR c IN creators RETURN c"
        creators = [doc for doc in db.aql.execute(query_creators)]
        df_creators = pd.DataFrame(creators)

        query_videos = "FOR v IN videos RETURN v"
        videos = [doc for doc in db.aql.execute(query_videos)]
        df_videos = pd.DataFrame(videos)

        query_edges = "FOR e IN video_by_creator RETURN e"
        edges = [doc for doc in db.aql.execute(query_edges)]
        df_edges = pd.DataFrame(edges)

        return df_creators, df_videos, df_edges

    except Exception as e:
        st.error(f"Błąd pobierania danych: {str(e)}")
        return None, None, None


def generate_and_save_graph(df_creators, df_videos, df_edges):
    G = nx.Graph()

    # Dodawanie węzłów twórców
    for _, row in df_creators.iterrows():
        G.add_node(row['_key'],
                   label=row['name'],
                   title=f"Twórca: {row['name']}\nSubskrypcje: {row.get('total_subscribers', 'N/A')}",
                   color="#FFA500",
                   size=25)

    # Dodawanie węzłów filmów z skróconymi tytułami
    for _, row in df_videos.iterrows():
        # Skróć długie tytuły
        short_title = row['title'][:20] + "..." if len(row['title']) > 20 else row['title']

        G.add_node(row['_key'],
                   label=short_title,
                   title=f"Film: {row['title']}\nWyświetlenia: {row.get('views', 'N/A')}",
                   color="#87CEEB",
                   size=12)

    # Dodawanie krawędzi
    for _, row in df_edges.iterrows():
        G.add_edge(row['_from'].split('/')[-1], row['_to'].split('/')[-1])

    # Konfiguracja sieci
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)

    # Ustawienie opcji fizyki bezpośrednio
    net.barnes_hut(gravity=-5000, central_gravity=0.2, spring_length=200, spring_strength=0.1, damping=0.09)
    net.repulsion(node_distance=200, central_gravity=0.2, spring_length=200, spring_strength=0.1, damping=0.09)

    # Zwiększamy czytelność węzłów
    net.options.nodes = {
        "font": {
            "size": 14,
            "strokeWidth": 3,
            "strokeColor": "#222222"
        }
    }

    # Włączamy podpowiedzi (tooltips)
    net.options.interaction = {
        "hover": True,
        "tooltipDelay": 200
    }

    # Pokazujemy przyciski kontroli fizyki
    net.show_buttons(filter_=["physics"])

    # Zapisz graf do pliku HTML
    net.save_graph(SAVED_GRAPH_PATH)


# Funkcja do wyświetlania grafu w Streamlit
def show_graph_page():
    st.header("📌 Graf Relacji Twórców i Wideo")

    # Inicjalizacja zmiennych sesji
    if "last_data_state" not in st.session_state:
        st.session_state["last_data_state"] = (None, None, None)
        st.session_state["data_changed"] = True

    # Pobierz aktualne dane
    current_data = get_graph_data()

    if current_data[0] is None or current_data[1] is None or current_data[2] is None:
        st.error("Nie udało się pobrać danych z bazy.")
        return

    # Wyświetl zapisany graf
    try:
        if os.path.exists(SAVED_GRAPH_PATH):
            with open(SAVED_GRAPH_PATH, "r", encoding="utf-8") as file:
                graph_html = file.read()
                components.html(graph_html, height=600)
        else:
            # Generowanie grafu jeśli nie istnieje
            generate_and_save_graph(current_data[0], current_data[1], current_data[2])
            # Następnie wyświetl go
            with open(SAVED_GRAPH_PATH, "r", encoding="utf-8") as file:
                graph_html = file.read()
                components.html(graph_html, height=600)
    except Exception as e:
        st.error(f"Błąd podczas wyświetlania grafu: {str(e)}")