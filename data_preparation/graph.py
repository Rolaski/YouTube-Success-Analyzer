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


# Funkcja do sprawdzania, czy dane się zmieniły
def has_data_changed(old_data, new_data):
    if any(df is None for df in old_data) or any(df is None for df in new_data):
        return True

    # Porównaj liczbę wierszy we wszystkich ramkach danych
    return (len(old_data[0]) != len(new_data[0]) or
            len(old_data[1]) != len(new_data[1]) or
            len(old_data[2]) != len(new_data[2]))


# Funkcja do dodawania twórcy do bazy
def add_creator_to_arango(name, url, subs, views, videos):
    try:
        client = ArangoClient(hosts='http://localhost:8529')
        db = client.db(ARANGO_DATABASE, username=ARANGO_USERNAME, password=ARANGO_PASSWORD)

        creator_doc = {
            '_key': re.sub(r'[^a-zA-Z0-9_]', '_', name),  # Tworzenie poprawnego klucza
            'name': name,
            'channel_url': url,
            'total_subscribers': subs,
            'total_views': views,
            'video_count': videos
        }

        db.collection('creators').insert(creator_doc)
        # Ustaw flagę, że dane się zmieniły
        if "last_data_state" in st.session_state:
            st.session_state["data_changed"] = True
        return True

    except Exception as e:
        st.error(f"Błąd dodawania twórcy: {str(e)}")
        return False


# Funkcja do dodawania nowego wideo
def add_video_to_arango(title, url, views, duration, creator_key):
    try:
        client = ArangoClient(hosts='http://localhost:8529')
        db = client.db(ARANGO_DATABASE, username=ARANGO_USERNAME, password=ARANGO_PASSWORD)

        video_doc = {
            '_key': hashlib.md5(title.encode()).hexdigest()[:10],  # Generowanie unikalnego klucza
            'title': title,
            'url': url,
            'views': views,
            'duration_seconds': duration
        }

        # Dodanie wideo do kolekcji
        db.collection('videos').insert(video_doc)

        # Stworzenie relacji twórca-wideo
        edge_doc = {
            '_from': f'creators/{creator_key}',
            '_to': f'videos/{video_doc["_key"]}'
        }
        db.collection('video_by_creator').insert(edge_doc)

        # Ustaw flagę, że dane się zmieniły
        if "last_data_state" in st.session_state:
            st.session_state["data_changed"] = True
        return True

    except Exception as e:
        st.error(f"Błąd dodawania wideo: {str(e)}")
        return False


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

    # --- Formularz do dodawania nowego twórcy ---
    st.subheader("➕ Dodaj nowego twórcę")
    with st.form("add_creator_form"):
        creator_name = st.text_input("Nazwa twórcy")
        creator_url = st.text_input("URL kanału")
        creator_subs = st.number_input("Liczba subskrybentów", min_value=0, step=1)
        creator_views = st.number_input("Liczba wyświetleń kanału", min_value=0, step=1)
        creator_videos = st.number_input("Liczba filmów na kanale", min_value=0, step=1)
        submitted = st.form_submit_button("Dodaj twórcę")

        if submitted and creator_name:
            success = add_creator_to_arango(creator_name, creator_url, creator_subs, creator_views, creator_videos)
            if success:
                st.success(f"✅ Twórca {creator_name} został dodany!")
                st.session_state["data_changed"] = True
                st.rerun()
            else:
                st.error("❌ Nie udało się dodać twórcy.")

    # --- Formularz do dodawania nowego filmu ---
    st.subheader("🎬 Dodaj nowe wideo")
    with st.form("add_video_form"):
        video_title = st.text_input("Tytuł filmu")
        video_url = st.text_input("URL filmu")
        video_views = st.number_input("Liczba wyświetleń", min_value=0, step=1)
        video_duration = st.number_input("Czas trwania (sekundy)", min_value=0, step=1)

        df_creators = current_data[0]
        if df_creators is not None and not df_creators.empty:
            creator_key = st.selectbox("Twórca filmu", df_creators['_key'].tolist())
            submitted_video = st.form_submit_button("Dodaj wideo")

            if submitted_video and video_title:
                success = add_video_to_arango(video_title, video_url, video_views, video_duration, creator_key)
                if success:
                    st.success(f"✅ Wideo {video_title} zostało dodane!")
                    st.session_state["data_changed"] = True
                    st.rerun()
                else:
                    st.error("❌ Nie udało się dodać wideo.")
        else:
            st.write("Najpierw dodaj twórców, aby móc dodać wideo.")
            st.form_submit_button("Dodaj wideo", disabled=True)
