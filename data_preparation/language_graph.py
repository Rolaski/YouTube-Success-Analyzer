import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import plotly.express as px
from arango import ArangoClient
from dotenv import load_dotenv

# Załaduj plik .env z folderu /database
env_path = os.path.join(os.path.dirname(__file__), '../database', '.env')
load_dotenv(env_path)
ARANGO_USERNAME = os.getenv("ARANGO_USERNAME")
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD")
ARANGO_DATABASE = os.getenv("ARANGO_DATABASE")

# Stała ścieżka do zapisanego grafu HTML
LANGUAGE_GRAPH_PATH = "language_graph.html"


# Pobranie danych o językach z bazy ArangoDB
def get_language_data():
    try:
        client = ArangoClient(hosts='http://localhost:8529')
        db = client.db(ARANGO_DATABASE, username=ARANGO_USERNAME, password=ARANGO_PASSWORD)

        # Zapytanie do pobrania filmów i ich języków
        query_videos = """
        FOR v IN videos
            FILTER v.language != null && v.language != ""
            LET creator = (
                FOR c, e IN 1..1 INBOUND v video_by_creator
                    RETURN c
            )[0]
            RETURN {
                video_key: v._key,
                video_title: v.title,
                language: v.language,
                views: v.views,
                upload_date: v.upload_date,
                creator_key: creator ? creator._key : null,
                creator_name: creator ? creator.name : null
            }
        """
        videos = list(db.aql.execute(query_videos))

        # Zapytanie o twórców z filmami w wielu językach
        query_creators = """
        FOR c IN creators
            LET languages = (
                FOR v, e IN 1..1 OUTBOUND CONCAT('creators/', c._key) video_by_creator
                    FILTER v.language != null && v.language != ""
                    RETURN DISTINCT v.language
            )
            FILTER LENGTH(languages) > 1
            RETURN {
                creator_key: c._key,
                creator_name: c.name,
                languages: languages
            }
        """
        creators_multilang = list(db.aql.execute(query_creators))

        return videos, creators_multilang

    except Exception as e:
        st.error(f"Błąd pobierania danych: {str(e)}")
        return None, None


# Generowanie grafu zależności językowych
def generate_language_graph(videos, creators_multilang, min_videos=5):
    G = nx.Graph()

    # Zliczanie filmów w każdym języku
    language_counts = {}
    for video in videos:
        lang = video['language']
        if lang not in language_counts:
            language_counts[lang] = 0
        language_counts[lang] += 1

    # Dodawanie węzłów języków (filtrowanie języków z małą liczbą filmów)
    for lang, count in language_counts.items():
        if count >= min_videos:
            G.add_node(lang, label=lang, title=f"Język: {lang}\nFilmy: {count}",
                       color="#3498db", size=min(25 + count // 10, 50))

    # Tworzenie krawędzi między językami na podstawie twórców, którzy tworzą treści w wielu językach
    for creator in creators_multilang:
        languages = creator['languages']
        # Filtruj języki, które są w naszym grafie
        languages = [lang for lang in languages if lang in G.nodes]

        # Utwórz krawędzie między wszystkimi parami języków
        for i in range(len(languages)):
            for j in range(i + 1, len(languages)):
                lang1, lang2 = languages[i], languages[j]
                # Dodaj krawędź lub zwiększ wagę, jeśli już istnieje
                if G.has_edge(lang1, lang2):
                    G[lang1][lang2]['weight'] += 1
                    G[lang1][lang2]['title'] = f"{G[lang1][lang2]['weight']} twórców używa obu języków"
                    # Zwiększ grubość krawędzi dla lepszej wizualizacji
                    G[lang1][lang2]['width'] = 1 + G[lang1][lang2]['weight']
                else:
                    G.add_edge(lang1, lang2, weight=1, title="1 twórca używa obu języków", width=1)

    # Konfiguracja wizualizacji sieci
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)

    # Konfiguracja fizyki
    net.barnes_hut(gravity=-5000, central_gravity=0.2, spring_length=200, spring_strength=0.1, damping=0.09)

    # Poprawa czytelności węzłów
    net.options.nodes = {
        "font": {
            "size": 14,
            "strokeWidth": 3,
            "strokeColor": "#222222"
        }
    }

    # Włączenie podpowiedzi (tooltips)
    net.options.interaction = {
        "hover": True,
        "tooltipDelay": 200
    }

    # Pokazywanie przycisków sterowania fizyką
    net.show_buttons(filter_=["physics"])

    # Zapisanie grafu do pliku HTML
    net.save_graph(LANGUAGE_GRAPH_PATH)

    return G


# Funkcja do wyświetlania twórców według języka
def show_creators_by_language(selected_language, videos):
    if not selected_language:
        return

        # Filtrowanie filmów według wybranego języka
    language_videos = [v for v in videos if v['language'] == selected_language]

    # Grupowanie według twórcy i zliczanie filmów
    creator_videos = {}
    for video in language_videos:
        creator_name = video.get('creator_name', 'Nieznany')
        if creator_name not in creator_videos:
            creator_videos[creator_name] = []

        # Bezpieczna konwersja views na liczbę
        try:
            views_value = video.get('views', 0)
            if isinstance(views_value, str):
                views_value = views_value.replace(',', '')
                views_value = float(views_value)
            elif views_value is None:
                views_value = 0
        except (ValueError, TypeError):
            views_value = 0

        creator_videos[creator_name].append({
            'title': video['video_title'],
            'views': views_value
        })

    # Obliczanie statystyk dla każdego twórcy
    creator_stats = []
    for creator, videos in creator_videos.items():
        # Bezpieczne obliczenie sumy wyświetleń
        total_views = 0
        for v in videos:
            try:
                if v['views']:
                    total_views += float(v['views'])
            except (ValueError, TypeError):
                pass

        avg_views = total_views / len(videos) if len(videos) > 0 else 0
        creator_stats.append({
            'creator': creator,
            'videos': len(videos),
            'total_views': total_views,
            'avg_views': avg_views
        })

    # Sortowanie według liczby filmów (malejąco)
    creator_stats.sort(key=lambda x: x['videos'], reverse=True)

    # Wyświetlanie twórców i ich statystyk
    st.subheader(f"Twórcy w języku {selected_language}")

    if not creator_stats:
        st.info(f"Nie znaleziono twórców dla języka {selected_language}")
        return

    # Tworzenie DataFrame dla lepszego wyświetlania
    df = pd.DataFrame(creator_stats)
    df.columns = ['Twórca', 'Liczba Filmów', 'Całkowita Liczba Wyświetleń', 'Średnia Liczba Wyświetleń na Film']

    # Formatowanie liczb dla lepszej czytelności
    df['Całkowita Liczba Wyświetleń'] = df['Całkowita Liczba Wyświetleń'].apply(lambda x: f"{int(x):,}")
    df['Średnia Liczba Wyświetleń na Film'] = df['Średnia Liczba Wyświetleń na Film'].apply(lambda x: f"{int(x):,}")

    # Wyświetlanie tabeli
    st.dataframe(df)


# Funkcja do analizowania trendów językowych w czasie
def show_language_trends(videos):
    # Przygotowanie danych
    videos_with_dates = []
    for video in videos:
        if 'upload_date' in video and video['upload_date']:
            # Konwersja string do daty
            try:
                video['upload_date'] = pd.to_datetime(video['upload_date'])
                videos_with_dates.append(video)
            except:
                pass

    if len(videos_with_dates) < 10:
        st.warning("Za mało filmów z prawidłowymi datami. Potrzeba co najmniej 10 filmów do analizy trendów.")
        return

    # Tworzenie DataFrame
    df = pd.DataFrame(videos_with_dates)

    # Dodawanie kolumny roku i miesiąca
    df['year'] = df['upload_date'].dt.year
    df['month'] = df['upload_date'].dt.month
    df['yearmonth'] = df['upload_date'].dt.to_period('M')

    # Najczęstsze języki
    top_languages = df['language'].value_counts().head(5).index.tolist()

    # Grupowanie według roku i języka
    yearly_language_counts = df.groupby(['year', 'language']).size().reset_index(name='count')
    yearly_top_languages = yearly_language_counts[yearly_language_counts['language'].isin(top_languages)]
    yearly_top_languages = yearly_top_languages.sort_values('year')

    # Tworzenie wykresu trendów językowych
    st.subheader("Trendy Popularności Języków w Czasie")

    # Wybór typu trendu
    trend_type = st.radio(
        "Wybierz typ analizy trendu:",
        ["Roczny", "Miesięczny (ostatnie 24 miesiące)"]
    )

    if trend_type == "Roczny":
        # Wykres dla wszystkich lat
        fig = px.line(
            yearly_top_languages,
            x='year',
            y='count',
            color='language',
            title="Trendy Językowe Według Lat",
            labels={'year': 'Rok', 'count': 'Liczba Filmów', 'language': 'Język'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # Miesięczny (ostatnie 24 miesiące)
        # Konwersja yearmonth do string dla wykresu
        df['yearmonth_str'] = df['yearmonth'].astype(str)

        # Filtrowanie ostatnich 24 miesięcy
        last_24_months = sorted(df['yearmonth'].unique())
        if len(last_24_months) > 24:
            last_24_months = last_24_months[-24:]

        monthly_df = df[df['yearmonth'].isin(last_24_months)]

        # Grupowanie według miesiąca i języka
        monthly_language_counts = monthly_df.groupby(['yearmonth_str', 'language']).size().reset_index(name='count')
        monthly_language_counts = monthly_language_counts[monthly_language_counts['language'].isin(top_languages)]

        fig = px.line(
            monthly_language_counts,
            x='yearmonth_str',
            y='count',
            color='language',
            title="Trendy Językowe w Ostatnich Miesiącach",
            labels={'yearmonth_str': 'Miesiąc', 'count': 'Liczba Filmów', 'language': 'Język'},
            markers=True
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # Analiza wzrostu/spadku popularności
    st.subheader("Analiza Wzrostu/Spadku Popularności Języków")

    # Dla każdego z najczęstszych języków, sprawdź wzrost/spadek
    growth_data = []

    for lang in top_languages:
        lang_data = yearly_top_languages[yearly_top_languages['language'] == lang]

        if len(lang_data) >= 2:
            first_year = lang_data['year'].min()
            last_year = lang_data['year'].max()

            first_count = lang_data[lang_data['year'] == first_year]['count'].values[0]
            last_count = lang_data[lang_data['year'] == last_year]['count'].values[0]

            years_diff = last_year - first_year
            if years_diff > 0:
                growth_pct = ((last_count / first_count) - 1) * 100

                growth_data.append({
                    'language': lang,
                    'first_year': first_year,
                    'last_year': last_year,
                    'growth_percent': growth_pct,
                    'first_count': first_count,
                    'last_count': last_count
                })

    if growth_data:
        growth_df = pd.DataFrame(growth_data)
        growth_df = growth_df.sort_values('growth_percent', ascending=False)

        # Formatowanie danych
        growth_df['growth_percent'] = growth_df['growth_percent'].apply(lambda x: f"{x:.1f}%")

        # Zmiana nazw kolumn
        growth_df.columns = ['Język', 'Pierwszy Rok', 'Ostatni Rok', 'Wzrost %', 'Początkowa Liczba', 'Końcowa Liczba']

        st.dataframe(growth_df)

        # Wyświetlanie wniosków
        if len(growth_df) > 0:
            fastest_growing = growth_df.iloc[0]['Język']
            st.success(f"Język z najszybszym wzrostem: **{fastest_growing}**")

            if len(growth_df) > 1:
                slowest_growing = growth_df.iloc[-1]['Język']
                st.info(f"Język z najwolniejszym wzrostem/spadkiem: **{slowest_growing}**")


# Główna funkcja do wyświetlania strony zależności językowych
def show_language_page():
    st.header("🌐 Zależności Językowe")

    # Pobieranie danych
    videos, creators_multilang = get_language_data()

    if videos is None or creators_multilang is None:
        st.error("Nie udało się pobrać danych z bazy.")
        return

    # Dodaj zakładki dla różnych sekcji analizy
    tab1, tab2, tab3 = st.tabs(["Statystyki Językowe", "Graf Zależności", "Trendy Czasowe"])

    with tab1:
        # Wyświetlanie statystyk
        languages = {}
        for video in videos:
            lang = video['language']
            if lang not in languages:
                languages[lang] = {'count': 0, 'views': 0}
            languages[lang]['count'] += 1

            # Bezpieczna konwersja views do liczby
            try:
                # Jeśli views to string, konwertujemy na float lub int
                if 'views' in video:
                    views_value = video['views']
                    if isinstance(views_value, str):
                        # Usuwamy przecinki, które mogą być w liczbach
                        views_value = views_value.replace(',', '')
                        views_value = float(views_value)
                    languages[lang]['views'] += float(views_value) if views_value else 0
                else:
                    languages[lang]['views'] += 0
            except (ValueError, TypeError):
                # W przypadku błędu konwersji, nie dodajemy nic
                pass

        # Konwersja do DataFrame dla łatwiejszej manipulacji
        lang_df = pd.DataFrame([
            {'language': lang, 'videos': data['count'], 'total_views': data['views']}
            for lang, data in languages.items()
        ])

        # Obliczanie średniej liczby wyświetleń
        lang_df['avg_views'] = lang_df['total_views'] / lang_df['videos']
        lang_df = lang_df.sort_values('videos', ascending=False)

        # Wyświetlanie najpopularniejszych języków
        st.subheader("Najpopularniejsze Języki według Liczby Filmów")

        # Tworzenie dwóch kolumn dla filtrów
        col1, col2 = st.columns(2)

        with col1:
            # Opcje filtrowania
            min_videos = st.slider("Minimalna liczba filmów dla języka", 1, 50, 5)

        with col2:
            # Opcje sortowania
            sort_by = st.selectbox(
                "Sortuj języki według",
                ["Liczby Filmów", "Całkowitej Liczby Wyświetleń", "Średniej Liczby Wyświetleń"]
            )

        # Filtrowanie i sortowanie danych
        filtered_df = lang_df[lang_df['videos'] >= min_videos]

        if sort_by == "Liczby Filmów":
            filtered_df = filtered_df.sort_values('videos', ascending=False)
        elif sort_by == "Całkowitej Liczby Wyświetleń":
            filtered_df = filtered_df.sort_values('total_views', ascending=False)
        else:  # Średniej Liczby Wyświetleń
            filtered_df = filtered_df.sort_values('avg_views', ascending=False)

        # Ograniczenie do 15 najlepszych dla przejrzystości
        display_df = filtered_df.head(15).copy()

        # Formatowanie do wyświetlenia
        display_df['total_views'] = display_df['total_views'].apply(lambda x: f"{int(x):,}")
        display_df['avg_views'] = display_df['avg_views'].apply(lambda x: f"{int(x):,}")

        # Zmiana nazw kolumn do wyświetlenia
        display_df.columns = ['Język', 'Liczba Filmów', 'Całkowita Liczba Wyświetleń', 'Średnia Liczba Wyświetleń']

        # Wyświetlanie tabeli
        st.dataframe(display_df)

        # Wykres top 10 języków
        top10_df = filtered_df.head(10)
        fig = px.bar(
            top10_df,
            x='language',
            y='videos',
            title="Top 10 Najpopularniejszych Języków",
            labels={'language': 'Język', 'videos': 'Liczba Filmów'},
            color='videos'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Generowanie grafu
        min_videos_for_graph = st.slider("Minimalna liczba filmów dla języka w grafie", 1, 50, 5)

        if st.button("Generuj Graf Zależności Językowych"):
            with st.spinner("Generowanie grafu..."):
                G = generate_language_graph(videos, creators_multilang, min_videos_for_graph)
                num_languages = len(G.nodes)
                num_connections = len(G.edges)
                st.success(
                    f"Graf został wygenerowany! Pokazuje {num_languages} języków z {num_connections} połączeniami między nimi.")

        # Sprawdzanie, czy plik grafu istnieje i wyświetlanie go
        if os.path.exists(LANGUAGE_GRAPH_PATH):
            st.subheader("Graf Zależności Językowych")
            st.info("""
            Ten graf pokazuje zależności między językami na podstawie twórców, którzy tworzą treści w wielu językach.
            - Każdy węzeł reprezentuje język
            - Rozmiar węzła odzwierciedla liczbę filmów w danym języku
            - Krawędzie między językami wskazują na twórców, którzy tworzą filmy w obu językach
            - Grubość krawędzi pokazuje, ilu twórców łączy te dwa języki
            """)

            try:
                with open(LANGUAGE_GRAPH_PATH, "r", encoding="utf-8") as file:
                    graph_html = file.read()
                    components.html(graph_html, height=600)
            except Exception as e:
                st.error(f"Błąd podczas wyświetlania grafu: {str(e)}")

            # Umożliwienie wyboru języka, aby zobaczyć twórców
            languages_list = list(languages.keys())
            languages_list.sort()

            selected_language = st.selectbox(
                "Wybierz język, aby zobaczyć twórców",
                options=[""] + languages_list,
                index=0,
                format_func=lambda x: x if x else "Wybierz język..."
            )

            if selected_language:
                show_creators_by_language(selected_language, videos)

        # Dodawanie pomocnych wskazówek
        st.info("""
        **Wskazówki:**
        - Przeciągaj węzły, aby lepiej zobaczyć relacje
        - Najedź kursorem na węzły lub krawędzie, aby uzyskać więcej informacji
        - Użyj przycisków fizyki w prawym dolnym rogu, aby kontrolować fizykę grafu
        - Kliknij dwukrotnie na węzeł, aby wycentrować widok
        """)

    with tab3:
        # Analiza trendów językowych w czasie
        show_language_trends(videos)