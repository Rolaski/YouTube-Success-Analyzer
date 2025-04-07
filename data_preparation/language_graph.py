import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv
from arango import ArangoClient
import plotly.express as px
import plotly.graph_objects as go

# Wczytaj plik .env z folderu /database
env_path = os.path.join(os.path.dirname(__file__), '../database', '.env')

# Wczytaj dane uwierzytelniające
load_dotenv(env_path)
ARANGO_USERNAME = os.getenv("ARANGO_USERNAME")
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD")
ARANGO_DATABASE = os.getenv("ARANGO_DATABASE")

# Ścieżka do zapisywanego pliku z grafem
LANGUAGE_GRAPH_PATH = "../../language_graph.html"

def get_language_data():
    """
    Pobierz dane o językach z bazy ArangoDB.

    Zwraca:
        tuple: DataFrame z danymi o językach w filmach oraz twórcach
    """
    try:
        client = ArangoClient(hosts='http://localhost:8529')
        db = client.db(ARANGO_DATABASE, username=ARANGO_USERNAME, password=ARANGO_PASSWORD)

        query = """
        FOR v IN videos
            FILTER v.language != null AND v.language != ""
            LET creator = (
                FOR c, e IN 1..1 INBOUND v video_by_creator
                RETURN c
            )[0]
            FILTER creator != null
            RETURN {
                "video_key": v._key,
                "video_title": v.title,
                "video_views": TO_NUMBER(v.views),
                "language": v.language,
                "creator_key": creator._key,
                "creator_name": creator.name,
                "creator_subs": creator.total_subscribers
            }
        """
        cursor = db.aql.execute(query)
        video_language_data = pd.DataFrame([doc for doc in cursor])

        if 'video_views' in video_language_data.columns and not video_language_data.empty:
            video_language_data['video_views'] = pd.to_numeric(video_language_data['video_views'], errors='coerce')

        query_creators = "FOR c IN creators RETURN c"
        creators_data = pd.DataFrame([doc for doc in db.aql.execute(query_creators)])

        return video_language_data, creators_data

    except Exception as e:
        st.error(f"Błąd podczas pobierania danych językowych: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


def analyze_language_success(video_language_data):
    """
    Analizuj sukces języków (średnia oglądalność, liczba twórców itp.)
    """
    if video_language_data.empty:
        return pd.DataFrame()

    language_metrics = video_language_data.groupby('language').agg(
        video_count=('video_key', 'count'),
        avg_views=('video_views', 'mean'),
        total_views=('video_views', 'sum'),
        creator_count=('creator_key', lambda x: len(set(x)))
    ).reset_index()

    return language_metrics.sort_values('avg_views', ascending=False)


def generate_language_creator_graph(video_language_data, max_nodes=50, min_videos=1):
    """
    Generuj wizualizację grafu relacji twórca–język.
    """
    if video_language_data.empty:
        st.error("Brak danych do wygenerowania grafu.")
        return 0, 0

    G = nx.Graph()
    language_metrics = analyze_language_success(video_language_data)
    min_videos = 1

    valid_languages = language_metrics[language_metrics['video_count'] >= min_videos]['language'].tolist()

    if not valid_languages:
        st.warning(f"Brak języków z przynajmniej {min_videos} filmami.")
        return 0, 0

    filtered_data = video_language_data[video_language_data['language'].isin(valid_languages)]
    creator_language_counts = filtered_data.groupby(['creator_key', 'language']).size().reset_index(name='video_count')

    creator_views = filtered_data.groupby('creator_key').agg(
        total_views=('video_views', 'sum'),
        creator_name=('creator_name', 'first')
    ).reset_index()

    max_nodes = max(max_nodes, 50)

    top_creators = creator_views.sort_values('total_views', ascending=False).head(max_nodes)['creator_key'].tolist()
    filtered_edges = creator_language_counts[creator_language_counts['creator_key'].isin(top_creators)]

    languages_used_by_top_creators = set(filtered_edges['language'].unique())

    max_video_count = language_metrics['video_count'].max()
    for _, row in language_metrics.iterrows():
        size = 15 + (row['video_count'] / max_video_count) * 35
        color = "#8BC34A" if row['language'] not in languages_used_by_top_creators else "#4CAF50"
        G.add_node(
            f"lang_{row['language']}",
            label=row['language'],
            title=f"Język: {row['language']}<br>Filmy: {row['video_count']}<br>Śr. oglądalność: {row['avg_views']:,.0f}<br>Twórcy: {row['creator_count']}",
            color=color,
            size=size,
            shape="hexagon"
        )

    max_views = creator_views['total_views'].max()
    for _, row in creator_views[creator_views['creator_key'].isin(top_creators)].iterrows():
        size = 15 + (row['total_views'] / max_views) * 35
        G.add_node(
            f"creator_{row['creator_key']}",
            label=row['creator_name'],
            title=f"Twórca: {row['creator_name']}<br>Łączna liczba wyświetleń: {row['total_views']:,.0f}",
            color="#FF9800",
            size=size,
            shape="dot"
        )

    for _, row in filtered_edges.iterrows():
        if (f"creator_{row['creator_key']}" not in G.nodes or f"lang_{row['language']}" not in G.nodes):
            continue
        width = 1 + min(9, row['video_count'] / 2)
        G.add_edge(
            f"creator_{row['creator_key']}",
            f"lang_{row['language']}",
            width=width,
            title=f"{row['video_count']} filmów"
        )

    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    net.barnes_hut(gravity=-5000, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)

    net.options.nodes = {
        "font": {
            "size": 14,
            "strokeWidth": 3,
            "strokeColor": "#222222"
        }
    }

    net.options.interaction = {
        "hover": True,
        "tooltipDelay": 200
    }

    net.show_buttons(filter_=["physics"])
    net.save_graph(LANGUAGE_GRAPH_PATH)

    return len([n for n in G.nodes if n.startswith("creator_")]), len([n for n in G.nodes if n.startswith("lang_")])


def create_language_metrics_charts(language_metrics):
    """
    Tworzy wykresy statystyk językowych.
    """
    figures = []
    if language_metrics.empty:
        return figures

    top_languages = language_metrics.head(15)

    fig1 = px.bar(
        top_languages,
        x='language',
        y='avg_views',
        title="Średnia liczba wyświetleń wg języka",
        labels={'language': 'Język', 'avg_views': 'Śr. wyświetlenia'},
        color='avg_views',
        text_auto='.2s'
    )
    fig1.update_layout(xaxis_tickangle=-45)
    figures.append(fig1)

    fig2 = px.bar(
        top_languages,
        x='language',
        y='video_count',
        title="Liczba filmów wg języka",
        labels={'language': 'Język', 'video_count': 'Liczba filmów'},
        color='video_count',
        text_auto=True
    )
    fig2.update_layout(xaxis_tickangle=-45)
    figures.append(fig2)

    fig3 = px.bar(
        top_languages,
        x='language',
        y='creator_count',
        title="Liczba twórców wg języka",
        labels={'language': 'Język', 'creator_count': 'Liczba twórców'},
        color='creator_count',
        text_auto=True
    )
    fig3.update_layout(xaxis_tickangle=-45)
    figures.append(fig3)

    fig4 = px.scatter(
        language_metrics,
        x='video_count',
        y='avg_views',
        size='creator_count',
        color='total_views',
        hover_name='language',
        title="Macierz skuteczności języków",
        labels={
            'video_count': 'Liczba filmów',
            'avg_views': 'Śr. liczba wyświetleń',
            'creator_count': 'Liczba twórców',
            'total_views': 'Łączna liczba wyświetleń'
        },
        log_y=True
    )
    figures.append(fig4)

    return figures


def show_language_analysis_page():
    """
    Wyświetla stronę analizy języków w Streamlit.
    """
    st.header("🌍 Analiza zależności językowych")

    with st.spinner("Pobieranie danych językowych z ArangoDB..."):
        video_language_data, creators_data = get_language_data()

    if video_language_data.empty:
        st.error("Nie znaleziono danych językowych. Upewnij się, że baza zawiera filmy z informacją o języku.")
        return

    st.subheader("📊 Podsumowanie danych")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Filmy z informacją o języku", f"{len(video_language_data)}")
    with col2:
        st.metric("Unikalne języki", f"{video_language_data['language'].nunique()}")
    with col3:
        st.metric("Twórcy używający wielu języków",
                  f"{video_language_data.groupby('creator_key')['language'].nunique()[lambda x: x > 1].count()}")

    with st.spinner("Analiza metryk językowych..."):
        language_metrics = analyze_language_success(video_language_data)

    st.subheader("📈 Statystyki skuteczności języków")

    charts = create_language_metrics_charts(language_metrics)
    for fig in charts:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Graf relacji język–twórca")

    with st.form("language_graph_options"):
        col1, col2 = st.columns(2)

        with col1:
            max_nodes = st.number_input(
                "Maksymalna liczba twórców na grafie",
                min_value=5,
                max_value=100,
                value=25,
                help="Ogranicz liczbę twórców w grafie dla czytelniejszego widoku"
            )
        with col2:
            min_videos = st.number_input(
                "Minimalna liczba filmów dla języka",
                min_value=1,
                max_value=20,
                value=1,
                help="Uwzględnij tylko języki z przynajmniej tyloma filmami"
            )

        submitted = st.form_submit_button("Generuj graf")

    if submitted or not os.path.exists(LANGUAGE_GRAPH_PATH):
        with st.spinner("Generowanie grafu relacji język–twórca..."):
            creator_count, language_count = generate_language_creator_graph(
                video_language_data, max_nodes, min_videos
            )

            if creator_count > 0 and language_count > 0:
                st.success(f"Wygenerowano graf z {creator_count} twórcami i {language_count} językami.")

                with st.expander("Informacje debugowe"):
                    st.write(f"Języków w bazie: {video_language_data['language'].nunique()}")
                    st.write(f"Języki w bazie: {sorted(video_language_data['language'].unique())}")
                    st.write(f"Języki w grafie: {language_count}")
            else:
                st.warning("Nie udało się wygenerować grafu. Spróbuj zmienić parametry.")

    try:
        if os.path.exists(LANGUAGE_GRAPH_PATH):
            st.markdown("### Graf relacji twórca–język")
            st.markdown("""
            ### Legenda:
            - 🟠 **Pomarańczowe węzły**: Twórcy
            - 🟢 **Zielone węzły**: Języki
            - Grubość krawędzi oznacza liczbę filmów w danym języku
            """)
            with open(LANGUAGE_GRAPH_PATH, "r", encoding="utf-8") as file:
                graph_html = file.read()
                components.html(graph_html, height=600)

            st.info("""
            **Porady dotyczące eksploracji grafu:**
            - Przeciągaj węzły, aby zmienić układ
            - Najedź kursorem, aby zobaczyć szczegóły
            - Przybliżaj i oddalaj scrollując
            - Użyj przycisków w prawym dolnym rogu do sterowania fizyką
            - Kliknij dwukrotnie, aby wycentrować na węźle
            """)
    except Exception as e:
        st.error(f"Błąd podczas wyświetlania grafu: {str(e)}")

    if not video_language_data.empty:
        st.subheader("👑 Rozkład języków u najpopularniejszych twórców")

        top_creator_views = video_language_data.groupby('creator_key').agg(
            total_views=('video_views', 'sum'),
            creator_name=('creator_name', 'first')
        ).nlargest(10, 'total_views')

        top_creator_languages = video_language_data[
            video_language_data['creator_key'].isin(top_creator_views.index)
        ].groupby(['creator_name', 'language']).size().reset_index(name='video_count')

        pivot_data = top_creator_languages.pivot_table(
            index='creator_name',
            columns='language',
            values='video_count',
            fill_value=0
        )

        fig = px.imshow(
            pivot_data,
            labels=dict(x="Język", y="Twórca", color="Liczba filmów"),
            title="Użycie języków przez najpopularniejszych twórców",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔄 Różnorodność językowa twórców")

        creator_language_diversity = video_language_data.groupby('creator_key').agg(
            language_count=('language', lambda x: len(set(x))),
            video_count=('video_key', 'count'),
            avg_views=('video_views', 'mean'),
            creator_name=('creator_name', 'first')
        ).reset_index()

        fig = px.scatter(
            creator_language_diversity,
            x='language_count',
            y='avg_views',
            size='video_count',
            hover_name='creator_name',
            title="Różnorodność językowa vs. średnia liczba wyświetleń",
            labels={
                'language_count': 'Liczba języków',
                'avg_views': 'Śr. wyświetlenia na film',
                'video_count': 'Liczba filmów'
            },
            log_y=True
        )
        st.plotly_chart(fig, use_container_width=True)

        correlation = creator_language_diversity['language_count'].corr(creator_language_diversity['avg_views'])

        if correlation > 0.2:
            st.success(f"Silna dodatnia korelacja ({correlation:.2f}) między liczbą języków a oglądalnością!")
        elif correlation < -0.2:
            st.info(f"Ujemna korelacja ({correlation:.2f}) – specjalizacja może przynosić lepsze wyniki.")
        else:
            st.info(f"Brak istotnej korelacji ({correlation:.2f}) między liczbą języków a sukcesem filmów.")
