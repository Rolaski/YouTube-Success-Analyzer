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


def generate_and_save_graph(df_creators, df_videos, df_edges, max_nodes=10, selected_creator=None):
    G = nx.Graph()

    # Filtrowanie danych na podstawie wybranego twórcy
    if selected_creator and not selected_creator == "Wszyscy":
        # Znajdź klucz wybranego twórcy
        creator_key = df_creators[df_creators['name'] == selected_creator]['_key'].values[0]

        # Filtruj krawędzie dla wybranego twórcy
        filtered_edges = df_edges[df_edges['_from'].str.contains(f"creators/{creator_key}")]

        # Znajdź filmy powiązane z wybranym twórcą
        related_video_keys = [edge['_to'].split('/')[-1] for _, edge in filtered_edges.iterrows()]

        # Filtruj DataFrame z filmami
        filtered_videos = df_videos[df_videos['_key'].isin(related_video_keys)]

        # Używaj tylko danych dla wybranego twórcy
        selected_creators = df_creators[df_creators['_key'] == creator_key]
    else:
        filtered_videos = df_videos
        filtered_edges = df_edges
        selected_creators = df_creators

    # Ograniczenie liczby węzłów z równoważeniem między twórcami i filmami
    total_available = len(selected_creators) + len(filtered_videos)
    if total_available > max_nodes:
        # Oblicz proporcjonalną liczbę twórców i filmów
        if len(selected_creators) == 1:
            # Jeśli mamy tylko jednego twórcę, pokazujemy go i resztę to filmy
            creators_to_show = 1
            videos_to_show = max_nodes - 1
        else:
            # Inaczej równoważymy między twórcami i filmami
            creators_ratio = min(0.3, len(selected_creators) / total_available)  # Maksymalnie 30% to twórcy
            creators_to_show = min(len(selected_creators), max(1, int(max_nodes * creators_ratio)))
            videos_to_show = max_nodes - creators_to_show

        # Ogranicz odpowiednio twórców i filmy
        selected_creators = selected_creators.head(creators_to_show)
        filtered_videos = filtered_videos.head(videos_to_show)

        # Zaktualizuj krawędzie
        creator_keys = selected_creators['_key'].tolist()
        video_keys = filtered_videos['_key'].tolist()

        filtered_edges = filtered_edges[
            filtered_edges['_from'].apply(lambda x: x.split('/')[-1] in creator_keys) &
            filtered_edges['_to'].apply(lambda x: x.split('/')[-1] in video_keys)
            ]

    # Dodawanie węzłów twórców
    for _, row in selected_creators.iterrows():
        G.add_node(row['_key'],
                   label=row['name'],
                   title=f"Twórca: {row['name']}\nSubskrypcje: {row.get('total_subscribers', 'N/A')}",
                   color="#FFA500",
                   size=25)

    # Dodawanie węzłów filmów z skróconymi tytułami
    for _, row in filtered_videos.iterrows():
        # Skróć długie tytuły
        short_title = row['title'][:20] + "..." if len(row['title']) > 20 else row['title']

        G.add_node(row['_key'],
                   label=short_title,
                   title=f"Film: {row['title']}\nWyświetlenia: {row.get('views', 'N/A')}",
                   color="#87CEEB",
                   size=12)

    # Dodawanie krawędzi
    for _, row in filtered_edges.iterrows():
        from_key = row['_from'].split('/')[-1]
        to_key = row['_to'].split('/')[-1]
        if G.has_node(from_key) and G.has_node(to_key):
            G.add_edge(from_key, to_key)

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

    # Zwróć faktyczną liczbę węzłów dla interfejsu
    return len(selected_creators), len(filtered_videos)


# Funkcja do wyświetlania grafu w Streamlit
def show_graph_page():
    st.header("📌 Graf Relacji Twórców i Wideo")

    # Pobierz dane
    df_creators, df_videos, df_edges = get_graph_data()

    if df_creators is None or df_videos is None or df_edges is None:
        st.error("Nie udało się pobrać danych z bazy.")
        return

    # Ustal maksymalną liczbę węzłów
    total_nodes = len(df_creators) + len(df_videos)

    # Inicjalizacja stanu
    if 'graph_generated' not in st.session_state:
        st.session_state.graph_generated = False

    # Przenosimy opcje filtrowania na stronę główną (nie w sidebarze)
    st.subheader("Opcje filtrowania")

    # Tworzymy dwie kolumny dla opcji filtrowania
    col1, col2 = st.columns(2)

    with col1:
        # Losowo wybieramy jednego twórcę jako domyślny
        if len(df_creators) > 0:
            default_creator_index = 1  # Pierwszy twórca po "Wszyscy"
            creator_names = ["Wszyscy"] + df_creators['name'].tolist()
            selected_creator = st.selectbox(
                "Filtruj według twórcy",
                options=creator_names,
                index=default_creator_index,
                help="Wybierz twórcę, aby zobaczyć jego filmy"
            )
        else:
            selected_creator = "Wszyscy"

    with col2:
        # Domyślnie pokazujemy bardzo małą liczbę węzłów (1 twórca + jego filmy)
        if selected_creator != "Wszyscy":
            # Oblicz domyślną liczbę węzłów dla jednego twórcy i kilku filmów
            creator_key = df_creators[df_creators['name'] == selected_creator]['_key'].values[0]
            related_videos_count = df_edges[df_edges['_from'].str.contains(f"creators/{creator_key}")].shape[0]
            default_nodes = max(2, min(10, 1 + related_videos_count))  # Minimum 2 węzły, maksimum 10
        else:
            default_nodes = 10  # Domyślna wartość

        # Dodaj możliwość wpisania liczby węzłów
        max_nodes = st.number_input(
            "Maksymalna liczba węzłów",
            min_value=2,  # Zmieniono minimalną wartość na 2 (minimalnie 1 twórca + 1 film)
            max_value=total_nodes,
            value=default_nodes,
            step=5,
            help="Wpisz lub wybierz liczbę węzłów do wyświetlenia"
        )

        # Ostrzeżenie przy dużej liczbie węzłów
        if max_nodes > 400:
            show_large_graph = st.checkbox(
                "Jestem świadomy, że generowanie dużego grafu może zająć do minuty czasu. Chcę kontynuować.",
                value=False,
                help="Generowanie dużego grafu może znacząco obciążyć przeglądarkę"
            )
            if not show_large_graph:
                max_nodes = 400
                st.warning(
                    "Liczba węzłów została ograniczona do 400. Zaznacz pole powyżej, aby wygenerować większy graf.")

    # Przycisk do regeneracji grafu
    if st.button("Generuj graf", help="Kliknij, aby wygenerować lub odświeżyć graf"):
        with st.spinner("Generowanie grafu..."):
            # Generuj graf na podstawie wybranych opcji
            creators_count, videos_count = generate_and_save_graph(df_creators, df_videos, df_edges, max_nodes,
                                                                   selected_creator)
            st.session_state.graph_generated = True
            st.success(f"Graf został wygenerowany! Wyświetlono {creators_count} twórców i {videos_count} filmów.")

    # Wizualizacja grafu
    if not st.session_state.graph_generated and not os.path.exists(SAVED_GRAPH_PATH):
        # Przy pierwszym załadowaniu, generujemy minimalny graf (10 węzłów)
        with st.spinner("Generowanie początkowego grafu..."):
            generate_and_save_graph(df_creators, df_videos, df_edges, 10, selected_creator)
            st.session_state.graph_generated = True

    # Wyświetl legend
    st.markdown("""
    ### Legenda:
    - 🟠 **Pomarańczowy węzeł**: Twórca
    - 🔵 **Niebieski węzeł**: Film
    """)

    # Wyświetl zapisany graf
    try:
        with open(SAVED_GRAPH_PATH, "r", encoding="utf-8") as file:
            graph_html = file.read()
            components.html(graph_html, height=600)
    except Exception as e:
        st.error(f"Błąd podczas wyświetlania grafu: {str(e)}")

    # Dodaj pomocnicze informacje
    st.info("""
    **Wskazówki:**
    - Możesz przeciągać węzły, aby lepiej zobaczyć relacje
    - Zbliż kursor do węzła, aby zobaczyć więcej informacji
    - Użyj przycisków w prawym dolnym rogu do sterowania fizyką grafu
    - Kliknij dwukrotnie na węzeł, aby wycentrować widok
    """)