import os
import time
import streamlit as st
from dotenv import load_dotenv
from arango import ArangoClient
import re
import hashlib
import pandas as pd
from datetime import datetime

# Załaduj plik konfiguracyjny .env z katalogu bieżącego
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Pobierz dane logowania do bazy ArangoDB z zmiennych środowiskowych
ARANGO_USERNAME = os.getenv("ARANGO_USERNAME")
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD")
ARANGO_DATABASE = os.getenv("ARANGO_DATABASE")

# Inicjalizacja klienta ArangoDB i połączenie z bazą danych
client = ArangoClient(hosts='http://localhost:8529')
db = client.db(ARANGO_DATABASE, username=ARANGO_USERNAME, password=ARANGO_PASSWORD)


def get_creators():
    """
    Pobiera listę twórców z kolekcji 'creators' w bazie danych.
    Zwraca listę dokumentów lub pustą listę w przypadku błędu.
    """
    try:
        return list(db.collection('creators').all())
    except Exception as e:
        st.error(f"Błąd pobierania twórców: {str(e)}")
        return []


def get_videos():
    """
    Pobiera listę filmów z kolekcji 'videos' w bazie danych.
    Zwraca listę dokumentów lub pustą listę w przypadku błędu.
    """
    try:
        return list(db.collection('videos').all())
    except Exception as e:
        st.error(f"Błąd pobierania wideo: {str(e)}")
        return []


def delete_creator(creator_key):
    """
    Usuwa twórcę z bazy danych na podstawie podanego klucza.
    Zwraca True, jeśli operacja zakończyła się sukcesem, w przeciwnym razie False.
    """
    try:
        db.collection('creators').delete(creator_key)
        return True
    except Exception as e:
        st.error(f"Błąd usuwania twórcy: {str(e)}")
        return False


def delete_video(video_key):
    """
    Usuwa film z bazy danych na podstawie podanego klucza.
    Zwraca True, jeśli operacja zakończyła się sukcesem, w przeciwnym razie False.
    """
    try:
        db.collection('videos').delete(video_key)
        return True
    except Exception as e:
        st.error(f"Błąd usuwania wideo: {str(e)}")
        return False


def add_creator(name, url, subs, views, videos, gender, playlist_count, community_engagement):
    """
    Dodaje nowego twórcę do kolekcji 'creators'.

    Parametry:
        name: nazwa twórcy (używana też do generowania klucza)
        url: URL kanału
        subs: liczba subskrybentów
        views: liczba wyświetleń kanału
        videos: liczba filmów
        gender: płeć twórcy (Male lub Female)
        playlist_count: liczba playlist
        community_engagement: zaangażowanie społeczności

    Zwraca True, jeśli operacja się powiodła, w przeciwnym razie False.
    """
    try:
        creator_doc = {
            # Klucz generowany na podstawie nazwy, z zamianą niedozwolonych znaków na podkreślenia
            '_key': re.sub(r'[^a-zA-Z0-9_]', '_', name),
            'name': name,
            'channel_url': url,
            'total_subscribers': subs,
            'total_views': views,
            'video_count': videos,
            'gender': gender,
            'playlist_count': playlist_count,
            'community_engagement': community_engagement
        }
        db.collection('creators').insert(creator_doc, overwrite=False)
        st.success(f'✅ Twórca {name} został dodany!')
        return True
    except Exception as e:
        st.error(f'❌ Błąd dodawania twórcy: {str(e)}')
        return False


def update_creator(creator_key, updated_fields):
    """
    Aktualizuje rekord twórcy na podstawie podanego klucza.

    Parametry:
        creator_key: klucz dokumentu twórcy
        updated_fields: słownik zawierający pola do aktualizacji

    Zwraca True, jeśli operacja się powiodła, w przeciwnym razie False.
    """
    try:
        db.collection('creators').update({'_key': creator_key, **updated_fields})
        st.success(f"✅ Twórca {updated_fields.get('name', creator_key)} został zaktualizowany!")
        return True
    except Exception as e:
        st.error(f"❌ Błąd edycji twórcy: {str(e)}")
        return False


def add_video(title, url, views, duration, creator_key, upload_date, likes, language, subtitle, description, hashtags,
              comment_count, last_comment_date, max_quality, premiered, data_collector):
    """
    Dodaje nowe wideo do kolekcji 'videos' i tworzy relację między twórcą a filmem.

    Parametry:
        title: tytuł filmu
        url: URL filmu
        views: liczba wyświetleń
        duration: czas trwania filmu (w sekundach)
        creator_key: klucz twórcy, do którego film jest przypisany
        upload_date: data przesłania (obiekt datetime lub string)
        likes: liczba lajków
        language: język filmu
        subtitle: czy film ma napisy (boolean)
        description: czy film posiada opis (boolean)
        hashtags: hashtagi (string)
        comment_count: liczba komentarzy
        last_comment_date: data ostatniego komentarza (obiekt datetime lub string)
        max_quality: maksymalna jakość (liczba)
        premiered: czy film miał premierę (boolean)
        data_collector: informacja kto zebrał dane

    Zwraca True, jeśli operacja się powiodła, w przeciwnym razie False.
    """
    try:
        video_doc = {
            # Generowanie unikalnego klucza na podstawie tytułu (używając skrótu MD5)
            '_key': hashlib.md5(title.encode()).hexdigest()[:10],
            'title': title,
            'url': url,
            'views': views,
            'duration_seconds': duration,
            # Konwersja daty na string, jeśli jest obiektem datetime
            'upload_date': upload_date.strftime('%Y-%m-%d') if isinstance(upload_date, datetime) else upload_date,
            'likes': likes,
            'language': language,
            'subtitle': subtitle,
            'description': description,
            'hashtags': hashtags,
            'comment_count': comment_count,
            'last_comment_date': last_comment_date.strftime('%Y-%m-%d') if isinstance(last_comment_date,
                                                                                      datetime) else last_comment_date,
            'max_quality': max_quality,
            'premiered': premiered,
            'data_collector': data_collector
        }
        db.collection('videos').insert(video_doc)

        # Tworzenie relacji między twórcą a filmem
        edge_doc = {
            '_from': f'creators/{creator_key}',
            '_to': f'videos/{video_doc["_key"]}'
        }
        db.collection('video_by_creator').insert(edge_doc)

        return True
    except Exception as e:
        st.error(f"❌ Błąd dodawania wideo: {str(e)}")
        return False


def update_video(video_key, updated_fields):
    """
    Aktualizuje rekord wideo na podstawie podanego klucza.

    Parametry:
        video_key: klucz dokumentu wideo
        updated_fields: słownik zawierający pola do aktualizacji

    Zwraca True, jeśli operacja się powiodła, w przeciwnym razie False.
    """
    try:
        db.collection('videos').update({'_key': video_key, **updated_fields})
        st.success(f"✅ Wideo {updated_fields.get('title', video_key)} zostało zaktualizowane!")
        return True
    except Exception as e:
        st.error(f"❌ Błąd edycji wideo: {str(e)}")
        return False


def show_data_tables():
    """
    Wyświetla interfejs CRUD dla twórców i filmów:
    - Tabela z listą twórców
    - Formularz dodawania nowego twórcy
    - Formularz edycji wybranego twórcy
    - Lista i możliwość usunięcia twórcy
    - Tabela z listą filmów
    - Formularz dodawania nowego filmu
    - Formularz edycji wybranego filmu
    - Lista i możliwość usunięcia filmu
    """
    # Wyświetlenie tabeli z twórcami
    st.subheader("📜 Lista Twórców")
    creators = get_creators()
    if creators:
        df_creators = pd.DataFrame(creators).drop(columns=['_id'], errors='ignore')

        # Zamiana None na NaN i konwersja wszystkich wartości na stringi dla kompatybilności
        df_creators = df_creators.fillna(pd.NA)
        df_creators = df_creators.astype(str)

        st.dataframe(df_creators, hide_index=True, use_container_width=False)
    else:
        st.write("Brak danych o twórcach.")

    # Formularz do dodawania nowego twórcy
    st.subheader("➕ Dodaj nowego twórcę")
    with st.form("add_creator_form"):
        creator_name = st.text_input("Nazwa twórcy")
        creator_url = st.text_input("URL kanału")
        creator_subs = st.number_input("Liczba subskrybentów", min_value=0, step=1)
        creator_views = st.number_input("Liczba wyświetleń kanału", min_value=0, step=1)
        creator_videos = st.number_input("Liczba filmów", min_value=0, step=1)
        creator_gender = st.selectbox("Płeć twórcy", ["Male", "Female"])
        playlist_count = st.number_input("Liczba playlist", min_value=0, step=1)
        community_engagement = st.number_input("Zaangażowanie społeczności", min_value=0, step=1)
        submitted = st.form_submit_button("Dodaj twórcę")
        if submitted and creator_name:
            if add_creator(creator_name, creator_url, creator_subs, creator_views, creator_videos, creator_gender,
                           playlist_count, community_engagement):
                with st.status(f"✅ Twórca {creator_name} został dodany!", expanded=True) as status:
                    time.sleep(3)  # Wyświetla komunikat przez 3 sekundy
                    status.update(state="complete")
                st.rerun()

    # Sekcja edycji twórcy
    st.subheader("✏️ Edytuj twórcę")
    if creators:
        selected_edit_creator = st.selectbox(
            "Wybierz twórcę do edycji",
            options=[c['_key'] for c in creators],
            index=None,
            placeholder="Wybierz twórcę..."
        )
        # Pobierz rekord wybranego twórcy
        creator_to_edit = next((c for c in creators if c['_key'] == selected_edit_creator), None)
        if creator_to_edit:
            with st.form("edit_creator_form"):
                new_name = st.text_input("Nazwa twórcy", value=creator_to_edit.get('name', ''))
                new_url = st.text_input("URL kanału", value=creator_to_edit.get('channel_url', ''))
                # Konwersja wartości liczbowych, np. "40200.0" -> int(float("40200.0"))
                new_subs = st.number_input(
                    "Liczba subskrybentów",
                    min_value=0,
                    step=1,
                    value=int(float(creator_to_edit.get('total_subscribers', 0)))
                )
                new_views = st.number_input(
                    "Liczba wyświetleń kanału",
                    min_value=0,
                    step=1,
                    value=int(float(creator_to_edit.get('total_views', 0)))
                )
                new_videos = st.number_input(
                    "Liczba filmów",
                    min_value=0,
                    step=1,
                    value=int(float(creator_to_edit.get('video_count', 0)))
                )
                new_gender = st.selectbox(
                    "Płeć twórcy",
                    ["Male", "Female"],
                    index=0 if creator_to_edit.get('gender', 'Male') == "Male" else 1
                )
                new_playlist_count = st.number_input(
                    "Liczba playlist",
                    min_value=0,
                    step=1,
                    value=int(float(creator_to_edit.get('playlist_count', 0)))
                )
                new_engagement = st.number_input(
                    "Zaangażowanie społeczności",
                    min_value=0,
                    step=1,
                    value=int(float(creator_to_edit.get('community_engagement', 0)))
                )

                submitted_edit = st.form_submit_button("Zaktualizuj twórcę")
                if submitted_edit:
                    updated_fields = {
                        'name': new_name,
                        'channel_url': new_url,
                        'total_subscribers': new_subs,
                        'total_views': new_views,
                        'video_count': new_videos,
                        'gender': new_gender,
                        'playlist_count': new_playlist_count,
                        'community_engagement': new_engagement
                    }
                    if update_creator(selected_edit_creator, updated_fields):
                        time.sleep(3)
                        st.rerun()

    # Lista do usunięcia twórcy
    st.subheader("🗑 Usuń twórcę")
    if creators:
        creator_keys = [c['_key'] for c in creators]
        selected_creator = st.selectbox("Wybierz twórcę do usunięcia", creator_keys, index=None,
                                        placeholder="Wybierz twórce...")
        if st.button("Usuń twórcę"):
            if delete_creator(selected_creator):
                with st.status("✅ Twórca został usunięty!", expanded=True) as status:
                    time.sleep(3)
                    status.update(state="complete")
                st.rerun()

    # --- Tabela z wideo ---
    st.subheader("📜 Lista Wideo")
    videos = get_videos()
    if videos:
        df_videos = pd.DataFrame(videos).drop(columns=['_id'], errors='ignore')
        df_videos = df_videos.fillna(pd.NA)
        df_videos = df_videos.astype(str)
        st.dataframe(df_videos, hide_index=True, use_container_width=False)
    else:
        st.write("Brak danych o wideo.")

    # Formularz do dodawania wideo
    st.subheader("🎬 Dodaj nowe wideo")
    with st.form("add_video_form"):
        video_title = st.text_input("Tytuł filmu")
        video_url = st.text_input("URL filmu")
        video_views = st.number_input("Liczba wyświetleń", min_value=0, step=1)
        video_duration = st.number_input("Czas trwania (sekundy)", min_value=0, step=1)
        upload_date = st.date_input("Data przesłania")
        likes = st.number_input("Liczba lajków", min_value=0, step=1)
        language = st.text_input("Język")
        subtitle = st.checkbox("Czy ma napisy?")
        description = st.checkbox("Czy posiada opis?")
        hashtags = st.text_area("Hashtagi")
        comment_count = st.number_input("Liczba komentarzy", min_value=0, step=1)
        last_comment_date = st.date_input("Data ostatniego komentarza")
        max_quality_options = ["144p", "240p", "360p", "480p", "720p", "1080p", "1440p", "2160p"]
        max_quality = int(st.selectbox("Maksymalna jakość", max_quality_options).replace("p", ""))
        premiered = st.checkbox("Czy miało premierę?")
        data_collector = st.text_input("Kto zebrał dane?")

        if creators:
            creator_key = st.selectbox("Twórca filmu", creator_keys)
            submitted_video = st.form_submit_button("Dodaj wideo")
            if submitted_video and video_title:
                if add_video(video_title, video_url, video_views, video_duration, creator_key, upload_date, likes,
                             language, subtitle, description, hashtags, comment_count, last_comment_date, max_quality,
                             premiered, data_collector):
                    with st.status(f"✅ Wideo {video_title} zostało dodane!", expanded=True) as status:
                        time.sleep(3)
                        status.update(state="complete")
                    st.rerun()

    # Sekcja edycji filmu
    st.subheader("✏️ Edytuj wideo")
    videos = get_videos()
    if videos:
        # Tworzymy mapę: klucz -> tytuł
        video_titles = {v['_key']: v['title'] for v in videos}
        selected_video_title = st.selectbox(
            "Wybierz wideo do edycji",
            options=list(video_titles.values()),
            index=None,
            placeholder="Wybierz wideo..."
        )
        if selected_video_title:
            # Znajdź klucz na podstawie wybranego tytułu
            video_key = [k for k, v in video_titles.items() if v == selected_video_title][0]
            # Pobierz rekord wybranego filmu
            video_to_edit = next((v for v in videos if v['_key'] == video_key), None)
            if video_to_edit:
                with st.form("edit_video_form"):
                    new_title = st.text_input("Tytuł filmu", value=video_to_edit.get('title', ''))
                    new_url = st.text_input("URL filmu", value=video_to_edit.get('url', ''))
                    new_views = st.number_input("Liczba wyświetleń", min_value=0, step=1,
                                                value=int(float(video_to_edit.get('views', 0))))
                    new_duration = st.number_input("Czas trwania (sekundy)", min_value=0, step=1,
                                                   value=int(float(video_to_edit.get('duration_seconds', 0))))

                    try:
                        current_upload_date = datetime.strptime(video_to_edit.get('upload_date', '2020-01-01'),
                                                                '%Y-%m-%d').date()
                    except Exception:
                        current_upload_date = datetime.now().date()
                    new_upload_date = st.date_input("Data przesłania", value=current_upload_date)

                    new_likes = st.number_input("Liczba lajków", min_value=0, step=1,
                                                value=int(float(video_to_edit.get('likes', 0))))
                    new_language = st.text_input("Język", value=video_to_edit.get('language', ''))
                    new_subtitle = st.checkbox("Czy ma napisy?", value=bool(video_to_edit.get('subtitle', False)))
                    new_description = st.checkbox("Czy posiada opis?",
                                                  value=bool(video_to_edit.get('description', False)))
                    new_hashtags = st.text_area("Hashtagi", value=str(video_to_edit.get('hashtags', '')))
                    new_comment_count = st.number_input("Liczba komentarzy", min_value=0, step=1,
                                                        value=int(float(video_to_edit.get('comment_count', 0))))

                    try:
                        current_last_comment_date = datetime.strptime(
                            video_to_edit.get('last_comment_date', '2020-01-01'), '%Y-%m-%d').date()
                    except Exception:
                        current_last_comment_date = datetime.now().date()
                    new_last_comment_date = st.date_input("Data ostatniego komentarza", value=current_last_comment_date)

                    max_quality_options = ["144", "240", "360", "480", "720", "1080", "1440", "2160"]
                    default_quality = str(video_to_edit.get('max_quality', '720'))
                    new_max_quality = int(st.selectbox("Maksymalna jakość", max_quality_options,
                                                       index=max_quality_options.index(default_quality)))

                    new_premiered = st.checkbox("Czy miało premierę?",
                                                value=bool(video_to_edit.get('premiered', False)))
                    new_data_collector = st.text_input("Kto zebrał dane?",
                                                       value=video_to_edit.get('data_collector', ''))

                    submitted_video_edit = st.form_submit_button("Zaktualizuj wideo")
                    if submitted_video_edit:
                        updated_fields = {
                            'title': new_title,
                            'url': new_url,
                            'views': new_views,
                            'duration_seconds': new_duration,
                            'upload_date': new_upload_date.strftime('%Y-%m-%d'),
                            'likes': new_likes,
                            'language': new_language,
                            'subtitle': new_subtitle,
                            'description': new_description,
                            'hashtags': new_hashtags,
                            'comment_count': new_comment_count,
                            'last_comment_date': new_last_comment_date.strftime('%Y-%m-%d'),
                            'max_quality': new_max_quality,
                            'premiered': new_premiered,
                            'data_collector': new_data_collector
                        }
                        if update_video(video_key, updated_fields):
                            time.sleep(3)
                            st.rerun()
    else:
        st.write("Brak filmów do edycji.")

    # Lista do usunięcia wideo
    st.subheader("🗑 Usuń wideo")
    if videos:
        video_titles = {v['_key']: v['title'] for v in videos}  # Mapa: klucz -> tytuł
        selected_video_title = st.selectbox(
            "Wybierz wideo do usunięcia",
            options=list(video_titles.values()),
            index=None,
            placeholder="Select video..."
        )
        if st.button("Usuń wideo"):
            video_key = [k for k, v in video_titles.items() if v == selected_video_title][0]
            if delete_video(video_key):
                st.success(f"✅ Wideo '{selected_video_title}' zostało usunięte!")
                time.sleep(3)
                st.rerun()
    else:
        st.write("Brak wyników dla wyszukiwania")


def show_crud_page():
    """
    Główna funkcja interfejsu Streamlit do zarządzania bazą danych.
    Wyświetla wszystkie sekcje: listy, formularze dodawania, edycji i usuwania rekordów.
    """
    st.header("Zarządzanie bazą danych")
    show_data_tables()

# XXXXXXXxxxxxx xxxxxxxx
