import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
import joblib
import os
from arango import ArangoClient
from dotenv import load_dotenv
from data_preparation.graph import show_graph_page
from database.crud import show_crud_page

# Załaduj plik .env z folderu /database
env_path = os.path.join(os.path.dirname(__file__), 'database', '.env')

# Ładowanie zmiennych środowiskowych
load_dotenv(env_path)

# Pobranie danych uwierzytelniających ArangoDB z zmiennych środowiskowych
ARANGO_USERNAME = os.getenv('ARANGO_USERNAME')
ARANGO_PASSWORD = os.getenv('ARANGO_PASSWORD')
ARANGO_DATABASE = os.getenv('ARANGO_DATABASE')


# Funkcja do połączenia z ArangoDB i pobrania danych
def get_data_from_arango():
    try:
        # Connect to ArangoDB
        client = ArangoClient(hosts='http://localhost:8529')
        db = client.db(ARANGO_DATABASE, username=ARANGO_USERNAME, password=ARANGO_PASSWORD)

        query = """
        FOR v IN videos
            LET creator = (
                FOR c, e IN 1..1 INBOUND v video_by_creator
                    RETURN c
            )[0]
            RETURN MERGE(v, {"creator": creator})
        """
        cursor = db.aql.execute(query)
        data = [doc for doc in cursor]

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Ensure that key columns are numeric
        numeric_cols = ['views', 'likes', 'comment_count', 'hashtag_count', 'duration_seconds']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert invalid values to NaN

        # Convert dates
        date_columns = ['upload_date', 'last_comment_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')  # Convert invalid values to NaT

        return df
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return None


# Funkcja do przygotowania danych
def prepare_data(df):
    # Na początku funkcji prepare_data oraz analyze_success_patterns dodaj:
    for col in ['hashtag_count', 'duration_seconds', 'creator_community_engagement', 'creator_video_count']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Popraw konwersję dat, aby uniknąć błędów formatowania
    if 'upload_date' in df.columns:
        df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce')
    if 'last_comment_date' in df.columns:
        df['last_comment_date'] = pd.to_datetime(df['last_comment_date'], errors='coerce')

    # Obliczenie liczby dni aktywności tylko dla poprawnych dat
    if 'upload_date' in df.columns and 'last_comment_date' in df.columns:
        mask = ~(df['upload_date'].isna() | df['last_comment_date'].isna())
        df.loc[mask, 'days_active'] = (df.loc[mask, 'last_comment_date'] - df.loc[mask, 'upload_date']).dt.days

    # Bezpieczne obliczanie wskaźników
    if 'comment_count' in df.columns and 'views' in df.columns:
        mask = df['views'] > 0
        df.loc[mask, 'comment_ratio'] = df.loc[mask, 'comment_count'] / df.loc[mask, 'views']

    if 'likes' in df.columns and 'views' in df.columns:
        mask = df['views'] > 0
        df.loc[mask, 'like_ratio'] = df.loc[mask, 'likes'] / df.loc[mask, 'views']

    if 'description' in df.columns:
        df['has_description'] = ~df['description'].isna()
        df['description_length'] = df['description'].astype(str).apply(len)

    if 'title' in df.columns:
        df['title_length'] = df['title'].astype(str).apply(len)

    if 'hashtags' in df.columns:
        df['hashtag_count'] = df['hashtags'].astype(str).str.count('#')

    if 'hashtag_count' in df.columns and 'hashtag_category' not in df.columns:
        df['hashtag_count'] = pd.to_numeric(df['hashtag_count'], errors='coerce')
        df['hashtag_category'] = pd.cut(
            df['hashtag_count'],
            bins=[-1, 0, 3, 5, 10, float('inf')],  # Tutaj 10.01 jest używane jako górna granica
            labels=['0', '1-3', '4-5', '6-10', '10+']
        )

    # Dodaj bezpieczną ekstrakcję danych twórcy
    if 'creator' in df.columns:
        # Sprawdź czy pierwszy niepusty element jest słownikiem
        valid_creators = df['creator'].dropna()
        if len(valid_creators) > 0 and isinstance(valid_creators.iloc[0], dict):
            # Rozpakowanie danych o twórcy
            creator_df = pd.json_normalize(df['creator'])
            creator_columns = ['name', 'gender', 'total_subscribers', 'total_views',
                               'video_count', 'playlist_count', 'community_engagement']

            for col in creator_columns:
                if col in creator_df.columns:
                    df[f'creator_{col}'] = creator_df[col]

    # Określenie docelowej zmiennej (sukces = liczba wyświetleń)
    if 'views' in df.columns:
        df['log_views'] = np.log1p(df['views'])  # Logarytmiczna transformacja dla lepszego rozkładu

    # Usunięcie kolumn z zbyt wieloma wartościami null
    threshold = 0.5
    df = df.dropna(axis=1, thresh=int(threshold * len(df)))

    # Usunięcie wierszy z brakującymi wartościami w kluczowych kolumnach
    if 'views' in df.columns:
        df = df.dropna(subset=['views'])

    # Konwersja kolumn na odpowiednie typy
    type_conversions = {
        'views': 'float',
        'likes': 'float',
        'comment_count': 'float',
        'hashtag_count': 'float',
        'duration_seconds': 'float',
        'creator_total_subscribers': 'float',
        'creator_video_count': 'float',
        'creator_community_engagement': 'float'
    }

    for col, dtype in type_conversions.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# Funkcja do trenowania modelu
def train_model(df):
    # Określenie zmiennych niezależnych i zależnej
    if 'log_views' in df.columns:
        y = df['log_views']
    else:
        y = np.log1p(df['views'])

    # Usunięcie kolumn, które nie powinny być używane jako cechy
    exclude_cols = ['_id', '_key', '_rev', 'url', 'creator', 'views', 'log_views',
                    'title', 'description', 'hashtags']

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()

    # Identyfikacja typów kolumn
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Konwersja kolumn boolowskich na stringi
    for col in categorical_cols:
        X[col] = X[col].astype(str)

    # Preprocessing: imputacja brakujących wartości i kodowanie zmiennych kategorycznych
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Pipeline z preprocessingiem i modelem
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', SelectFromModel(GradientBoostingRegressor())),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trenowanie modelu
    model.fit(X_train, y_train)

    # Ocena modelu
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Zapisanie modelu
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'youtube_success_model.pkl'))

    # Obliczenie ważności cech
    regressor = model.named_steps['regressor']
    feature_names = []

    # Pobierz nazwy cech po transformacji
    if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    else:
        feature_names = numeric_cols + categorical_cols

    # Jeśli w pipeline znajduje się selektor cech, przefiltruj nazwy cech
    if 'selector' in model.named_steps:
        selector = model.named_steps['selector']
        mask = selector.get_support()
        feature_names = [f for f, m in zip(feature_names, mask) if m]

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': regressor.feature_importances_
    }).sort_values('importance', ascending=False)

    return model, feature_importance, {'mse': mse, 'r2': r2}


# Funkcja do analizy ogólnych wzorców sukcesu
def analyze_success_patterns(df):
    # Na początku funkcji prepare_data oraz analyze_success_patterns dodaj:
    for col in ['hashtag_count', 'duration_seconds', 'creator_community_engagement', 'creator_video_count']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Wyniki analizy, które zostaną zwrócone
    insights = {}

    # 1. Czas trwania a wyświetlenia
    if 'duration_seconds' in df.columns and 'views' in df.columns:
        # Usuwanie skrajnych wartości dla lepszej wizualizacji
        q_low = df['duration_seconds'].quantile(0.01)
        q_high = df['duration_seconds'].quantile(0.99)
        df_filtered = df[(df['duration_seconds'] >= q_low) & (df['duration_seconds'] <= q_high)].copy()

        # Przypisanie z użyciem .loc aby uniknąć SettingWithCopyWarning
        df_filtered.loc[:, 'duration_minutes'] = df_filtered['duration_seconds'] / 60

        df_filtered.loc[:, 'duration_category'] = pd.cut(
            df_filtered['duration_minutes'],
            bins=[0, 3, 5, 10, 15, 30, 60, float('inf')],
            labels=['0-3 min', '3-5 min', '5-10 min', '10-15 min', '15-30 min', '30-60 min', '60+ min'],
            # duplicates='drop'  # Usunięcie duplikatów w progach
        )

        # Obliczanie średniej liczby wyświetleń dla każdego przedziału
        duration_analysis = df_filtered.groupby('duration_category', observed=True)['views'].agg(
            ['mean', 'count', 'median']).reset_index()
        duration_analysis = duration_analysis[
            duration_analysis['count'] >= 3]  # Minimalna liczba filmów dla wiarygodności

        # Dodanie do wyników
        insights['best_duration'] = duration_analysis.iloc[duration_analysis['mean'].argmax()]['duration_category']
        insights['duration_analysis'] = duration_analysis

    # 2. Język a wyświetlenia
    if 'language' in df.columns and 'views' in df.columns:
        # Obliczanie średniej liczby wyświetleń dla każdego języka
        language_analysis = df.groupby('language', observed=True)['views'].agg(
            ['mean', 'count', 'median', 'sum']).reset_index()
        language_analysis = language_analysis[language_analysis['count'] >= 3]  # Minimum 3 filmy dla wiarygodności
        language_analysis = language_analysis.sort_values('mean', ascending=False)

        # Dodanie do wyników
        top_languages = language_analysis.sort_values('mean', ascending=False).head(5)
        insights['top_languages'] = top_languages

    # 3. Zaangażowanie społeczności a wyświetlenia
    if 'creator_community_engagement' in df.columns and 'views' in df.columns:
        # Tworzenie kategorii zaangażowania społeczności
        df['engagement_category'] = pd.cut(
            df['creator_community_engagement'],
            bins=[-1, 0, 1, 3, 5, 10, float('inf')],
            labels=['0', '1', '1-3', '3-5', '5-10', '10+']
        )

        # Obliczanie średniej liczby wyświetleń dla każdej kategorii
        engagement_analysis = df.groupby('engagement_category', observed=True)['views'].agg(
            ['mean', 'count', 'median']).reset_index()
        engagement_analysis = engagement_analysis[engagement_analysis['count'] >= 2]  # Minimum 2 filmy w kategorii

        # Dodanie do wyników
        insights['engagement_analysis'] = engagement_analysis

    # 4. Wpływ liczby filmów na kanale na sukces
    if 'creator_video_count' in df.columns and 'views' in df.columns:
        # Tworzenie kategorii liczby filmów
        df['video_count_category'] = pd.cut(
            df['creator_video_count'],
            bins=[0, 10, 50, 100, 500, float('inf')],
            labels=['1-10', '11-50', '51-100', '101-500', '500+']
        )

        # Obliczanie średniej liczby wyświetleń dla każdej kategorii
        video_count_analysis = df.groupby('video_count_category', observed=True)['views'].agg(
            ['mean', 'count', 'median']).reset_index()
        video_count_analysis = video_count_analysis[
            video_count_analysis['count'] >= 2]  # Minimum 2 filmy dla wiarygodności

        # Dodanie do wyników
        insights['video_count_analysis'] = video_count_analysis

    # 5. Wpływ hashtags na wyświetlenia
    if 'hashtag_count' in df.columns and 'views' in df.columns:
        # Tworzenie kategorii liczby hashtagów
        # Użycie .loc przy przypisaniu oraz duplicates='drop'
        df.loc[:, 'hashtag_category'] = pd.cut(
            df['hashtag_count'],
            bins=[-1, 0, 3, 5, 10, float('inf')],
            labels=['0', '1-3', '4-5', '6-10', '10+'],
        )

        # Obliczanie średniej liczby wyświetleń dla każdej kategorii
        hashtag_analysis = df.groupby('hashtag_category', observed=True)['views'].agg(
            ['mean', 'count', 'median']).reset_index()
        hashtag_analysis = hashtag_analysis[hashtag_analysis['count'] >= 2]  # Minimum 2 filmy dla wiarygodności

        # Dodanie do wyników
        insights['hashtag_analysis'] = hashtag_analysis

    if 'views' in df.columns:
        success_threshold = df['views'].quantile(0.75)
        df['is_successful'] = df['views'] >= success_threshold
        insights['success_threshold'] = success_threshold

        # Cechy charakterystyczne dla filmów odnoszących sukces
        success_df = df[df['is_successful']]
        regular_df = df[~df['is_successful']]

        # Porównanie cech dla filmów odnoszących sukces i pozostałych
        comparison = {}

        numeric_cols = ['duration_seconds', 'likes', 'comment_count',
                        'creator_total_subscribers', 'creator_video_count',
                        'creator_community_engagement', 'hashtag_count']

        for col in numeric_cols:
            if col in df.columns:
                # Sprawdzamy, czy w obu grupach są dane
                if not success_df[col].isna().all() and not regular_df[col].isna().all():
                    success_mean = success_df[col].mean()
                    regular_mean = regular_df[col].mean()

                    # Unikamy dzielenia przez zero lub wartości bardzo bliskie zeru
                    if abs(regular_mean) > 1e-10:  # Używamy małej wartości zamiast dokładnego zera
                        diff_pct = ((success_mean / regular_mean) - 1) * 100
                    else:
                        diff_pct = 0 if abs(success_mean) < 1e-10 else 100  # 100% więcej jeśli success_mean > 0

                    # Unikamy skrajnie dużych wartości, które mogą zaburzyć wykres
                    if abs(diff_pct) > 1000:
                        diff_pct = 1000 if diff_pct > 0 else -1000

                    comparison[col] = {
                        'successful_mean': success_mean,
                        'regular_mean': regular_mean,
                        'difference_pct': diff_pct
                    }

        insights['success_vs_regular'] = comparison

    return insights


# Funkcja do analizy indywidualnego filmu
def analyze_video(df, video_id):
    # Konwersja identyfikatora do string i wyszukiwanie filmu
    df['_key'] = df['_key'].astype(str)
    video_id = str(video_id)
    video_matches = df[df['_key'] == video_id]

    # Jeśli nie znaleziono, spróbuj wyszukać po tytule (fallback)
    if len(video_matches) == 0:
        video_matches = df[df['title'].str.contains(video_id, case=False, na=False)]
        if len(video_matches) == 0:
            return {"error": "Film nie został znaleziony. Upewnij się, że identyfikator jest prawidłowy."}

    video = video_matches.iloc[0]
    analysis = {}

    # 1. Podstawowe informacje o filmie
    analysis['basic_info'] = {
        'title': video.get('title', 'Brak tytułu'),
        'views': video.get('views', 0),
        'likes': video.get('likes', 0),
        'comments': video.get('comment_count', 0),
        'duration_seconds': video.get('duration_seconds', 0),
        'upload_date': video.get('upload_date', 'Unknown'),
        'language': video.get('language', 'Unknown'),
        'hashtag_count': video.get('hashtag_count', 0),
        'description_length': video.get('description_length', 0),
        'title_length': video.get('title_length', 0)
    }

    # 2. Ocena sukcesu na podstawie rankingu
    if 'views' in df.columns:
        views = video.get('views', 0)
        N = len(df)
        rank = (df['views'] > views).sum() + 1  # film o najwyższych wyświetleniach otrzymuje rank = 1
        top_percent = (rank / N) * 100  # im niższa wartość, tym lepiej
        thresholds = {f'top_{int(100 - q * 100)}%': df['views'].quantile(q) for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]}
        analysis['success_rating'] = {
            'top_percent': top_percent,
            'thresholds': thresholds,
            'is_viral': views >= thresholds.get('top_10%', float('inf')),
            'is_successful': views >= thresholds.get('top_25%', float('inf'))
        }

    # 3. Porównanie z filmami w tym samym języku
    similar_lang = df[df['language'] == video.get('language', 'Unknown')]
    language_avg_views = similar_lang['views'].mean() if not similar_lang.empty else None
    if language_avg_views:
        language_diff = (video.get('views', 0) / language_avg_views - 1) * 100
        analysis['language_comparison'] = {
            'avg_views': language_avg_views,
            'percent_difference': language_diff,
            'better_than_average': language_diff > 0
        }

    # 4. Analiza zaangażowania
    engagement = {}
    if video.get('views', 0) > 0:
        engagement['like_ratio'] = (video.get('likes', 0) / video.get('views', 1)) * 100
        engagement['comment_ratio'] = (video.get('comment_count', 0) / video.get('views', 1)) * 100

        # Obliczanie średnich wskaźników dla całego zbioru
        avg_like_ratio = (df['likes'] / df['views'] * 100).mean() if (df['views'] > 0).all() else None
        avg_comment_ratio = (df['comment_count'] / df['views'] * 100).mean() if (df['views'] > 0).all() else None

        engagement['like_ratio_vs_avg'] = engagement['like_ratio'] / (avg_like_ratio or 1)
        engagement['comment_ratio_vs_avg'] = engagement['comment_ratio'] / (avg_comment_ratio or 1)
    analysis['engagement_metrics'] = engagement

    # 5. Szczegółowa analiza cech i rekomendacje
    recommendations = []  # wskazówki do poprawy lub działania
    strengths = []  # mocne strony filmu
    factors = []  # czynniki wpływające na wyświetlenia

    # a) Język
    if language_avg_views:
        if language_diff > 50:
            factors.append(
                f"Film ma o {language_diff:.1f}% więcej wyświetleń niż średnia dla filmów w języku {video.get('language', 'Unknown')}, co świadczy o silnym potencjale w tej grupie."
            )
            strengths.append("Świetny wynik w ramach danego języka.")
        else:
            factors.append(
                f"Film osiąga wyniki o {abs(language_diff):.1f}% {'wyższe' if language_diff > 0 else 'niższe'} niż średnia dla filmów w języku {video.get('language', 'Unknown')}."
            )
            if language_diff < 0:
                recommendations.append(
                    "Rozważ działania marketingowe lub lepsze targetowanie, aby poprawić wyniki w tej grupie."
                )

    # b) Długość filmu
    duration_min = video.get('duration_seconds', 0) / 60
    avg_duration = df['duration_seconds'].mean() / 60 if 'duration_seconds' in df.columns else None
    if avg_duration:
        if duration_min < avg_duration * 0.7:
            factors.append(f"Film jest krótszy ({duration_min:.1f} min) niż średnia ({avg_duration:.1f} min).")
            recommendations.append(
                "Rozważ wydłużenie filmu, aby dostarczyć więcej treści, co może przyczynić się do lepszego zaangażowania widzów."
            )
        elif duration_min > avg_duration * 1.3:
            factors.append(f"Film jest dłuższy ({duration_min:.1f} min) niż średnia ({avg_duration:.1f} min).")
            recommendations.append(
                "Skrócenie filmu może pomóc w utrzymaniu uwagi widzów."
            )
        else:
            strengths.append("Długość filmu jest zbliżona do średniej, co wskazuje na odpowiedni balans treści.")

    # c) Tytuł
    current_title = video.get('title_length', 0)
    avg_title_length = df['title_length'].mean() if 'title_length' in df.columns else None
    if avg_title_length:
        if current_title < avg_title_length * 0.8:
            factors.append(
                f"Tytuł filmu ma {current_title} znaków, podczas gdy średnia wynosi {avg_title_length:.0f} znaków.")
            recommendations.append(
                "Rozważ uatrakcyjnienie tytułu poprzez dodanie większej ilości informacji lub emocji, co może zwiększyć CTR."
            )
        else:
            strengths.append("Tytuł filmu jest na poziomie lub powyżej średniej.")

    # d) Opis filmu
    current_desc = video.get('description_length', 0)
    avg_desc_length = df['description_length'].mean() if 'description_length' in df.columns else None
    if avg_desc_length:
        if current_desc < avg_desc_length * 0.5:
            factors.append(
                f"Opis filmu ma tylko {current_desc} znaków, co jest znacznie poniżej średniej ({avg_desc_length:.0f} znaków).")
            recommendations.append(
                "Rozważ rozbudowanie opisu filmu z uwzględnieniem kluczowych słów i szczegółów, co może poprawić SEO i zaangażowanie widzów."
            )
        else:
            strengths.append("Opis filmu jest wystarczająco rozbudowany.")

    # e) Hashtagi
    current_hashtag = video.get('hashtag_count', 0)
    avg_hashtag = df['hashtag_count'].mean() if 'hashtag_count' in df.columns else None
    if avg_hashtag:
        if current_hashtag < avg_hashtag * 0.7:
            factors.append(
                f"Film wykorzystuje {current_hashtag} hashtagów, co jest poniżej średniej ({avg_hashtag:.1f}).")
            recommendations.append(
                "Rozważ dodanie trafnych hashtagów, które pomogą w zwiększeniu zasięgu filmu."
            )
        elif current_hashtag > avg_hashtag * 1.5:
            factors.append(f"Film używa {current_hashtag} hashtagów, co przekracza średnią ({avg_hashtag:.1f}).")
            recommendations.append(
                "Zbyt duża liczba hashtagów może rozpraszać – warto ograniczyć się do kilku najtrafniejszych."
            )
        else:
            strengths.append("Liczba hashtagów jest optymalna.")

    # f) Zaangażowanie widzów
    like_ratio = (video.get('likes', 0) / video.get('views', 1)) * 100
    comment_ratio = (video.get('comments', 0) / video.get('views', 1)) * 100
    avg_like_ratio = (df['likes'] / df['views'] * 100).mean() if (df['views'] > 0).all() else None
    avg_comment_ratio = (df['comment_count'] / df['views'] * 100).mean() if (df['views'] > 0).all() else None
    if avg_like_ratio:
        if like_ratio < avg_like_ratio * 0.8:
            factors.append(
                f"Współczynnik polubień wynosi {like_ratio:.2f}%, podczas gdy średnia to {avg_like_ratio:.2f}%.")
            recommendations.append(
                "Rozważ dodanie wyraźnego wezwania do działania, aby zwiększyć liczbę polubień."
            )
        else:
            strengths.append("Współczynnik polubień jest na dobrym poziomie.")
    if avg_comment_ratio:
        if comment_ratio < avg_comment_ratio * 0.8:
            factors.append(
                f"Współczynnik komentarzy wynosi {comment_ratio:.2f}%, podczas gdy średnia to {avg_comment_ratio:.2f}%.")
            # Dla bardzo udanych filmów łagodniej
            if analysis.get('success_rating', {}).get('top_percent', 100) <= 10:
                recommendations.append(
                    "Mimo niskiego współczynnika komentarzy film osiąga ogromną liczbę wyświetleń. Można jednak rozważyć zachęcenie widzów do komentowania, aby zwiększyć interakcje."
                )
            else:
                recommendations.append(
                    "Niski współczynnik komentarzy sugeruje, że widzowie mogą być mniej zaangażowani. Zachęć do komentowania, np. poprzez pytania lub ankiety."
                )
        else:
            strengths.append("Współczynnik komentarzy jest zadowalający.")

    # g) Podsumowanie końcowe – komunikat zależny od pozycji w rankingu
    if 'success_rating' in analysis:
        top_percent = analysis['success_rating']['top_percent']
        if top_percent <= 10:
            final_msg = "Gratulacje! Film jest jednym z najlepszych na platformie."
        elif top_percent <= 25:
            final_msg = "Film osiąga bardzo dobre wyniki, ale warto rozważyć pewne usprawnienia dla jeszcze lepszego efektu."
        else:
            final_msg = "Film ma potencjał, jednak istnieją obszary, które warto zoptymalizować, aby zwiększyć zasięg i zaangażowanie."
    else:
        final_msg = ""

    # Łączenie wyników
    analysis['factors'] = factors  # Czynniki wpływające na wyświetlenia (zarówno atuty, jak i obszary do poprawy)
    analysis['optimization_suggestions'] = recommendations  # Konkretne wskazówki co można poprawić lub utrzymać
    analysis['strengths'] = strengths  # Mocne strony filmu
    analysis['final_summary'] = final_msg

    return analysis


# Streamlit UI
def main():
    st.set_page_config(page_title="Analiza Sukcesu na YouTube", page_icon="📊", layout="wide")

    # Tytuł aplikacji
    st.title("🎥 Analiza Sukcesu na YouTube")

    # Sidebar z nawigacją
    st.sidebar.title("Nawigacja")
    page = st.sidebar.radio("Wybierz stronę",
                            ["Ogólna Analiza Sukcesu", "Analiza Pojedynczego Filmu",
                             "Graf", "Zależności Językowe", "Zarządzanie bazą", "Finalne Podsumowanie"])

    # Wczytanie danych
    try:
        with st.spinner('Łączenie z bazą danych i pobieranie danych...'):
            df = get_data_from_arango()

        with st.spinner('Przygotowywanie danych do analizy...'):
            df = prepare_data(df)

        if page == "Ogólna Analiza Sukcesu":
            show_general_success_page(df)
        elif page == "Analiza Pojedynczego Filmu":
            show_single_video_analysis_page(df)
        elif page == "Finalne Podsumowanie":
            show_final_summary_page(df)
        elif page == "Graf":
            show_graph_page()
        elif page == "Zależności Językowe":
            from data_preparation.language_graph import show_language_page
            show_language_page()
        elif page == "Zarządzanie bazą":
            show_crud_page()

    except Exception as e:
        st.error(f"Wystąpił błąd podczas pobierania danych: {str(e)}")

        # Alternatywne rozwiązanie - wczytanie przykładowych danych, jeśli połączenie z bazą danych nie zadziała
        st.warning("Używanie przykładowych danych testowych...")

        # Tworzenie przykładowych danych dla demonstracji
        sample_data = create_sample_data()
        df = sample_data

        if page == "Ogólna Analiza Sukcesu":
            show_general_success_page(df)
        elif page == "Analiza Pojedynczego Filmu":
            show_single_video_analysis_page(df)
        elif page == "Finalne Podsumowanie":
            show_final_summary_page(df)
        elif page == "Graf":
            st.warning("Funkcja grafu nie jest dostępna w trybie przykładowych danych.")
        elif page == "Zależności Językowe":
            st.warning("Funkcja zależności językowych nie jest dostępna w trybie przykładowych danych.")
        elif page == "Zarządzanie bazą":
            st.warning("Funkcja zarządzania bazą nie jest dostępna w trybie przykładowych danych.")


# Funkcja do tworzenia przykładowych danych (w przypadku problemów z bazą danych)
def create_sample_data():
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    # Wygenerowanie danych dla 100 filmów
    np.random.seed(42)
    n_videos = 100

    # Podstawowe dane
    data = {
        '_key': [f'video_{i}' for i in range(n_videos)],
        'title': [f'Przykładowy film {i}' for i in range(n_videos)],
        'views': np.random.power(0.3, n_videos) * 1000000,
        'likes': np.random.power(0.4, n_videos) * 100000,
        'comment_count': np.random.power(0.5, n_videos) * 10000,
        'duration_seconds': np.random.choice([60, 180, 300, 600, 900, 1200, 1800, 3600], n_videos),
        'language': np.random.choice(['Polski', 'Angielski', 'Niemiecki', 'Francuski', 'Hiszpański'], n_videos,
                                     p=[0.4, 0.3, 0.1, 0.1, 0.1]),
        'hashtag_count': np.random.randint(0, 15, n_videos),
        'description_length': np.random.randint(0, 2000, n_videos),
        'title_length': np.random.randint(10, 100, n_videos),
        'upload_date': [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(n_videos)],
        'creator_name': np.random.choice(['Kanal A', 'Kanal B', 'Kanal C', 'Kanal D', 'Kanal E'], n_videos),
        'creator_total_subscribers': np.random.power(0.3, n_videos) * 1000000,
        'creator_video_count': np.random.randint(1, 500, n_videos),
        'creator_community_engagement': np.random.randint(0, 10, n_videos)
    }

    df = pd.DataFrame(data)

    # Dodatkowe przetwarzanie
    df['duration_minutes'] = df['duration_seconds'] / 60
    df['duration_category'] = pd.cut(
        df['duration_minutes'],
        bins=[0, 3, 5, 10, 15, 30, 60, 60.01],
        labels=['0-3 min', '3-5 min', '5-10 min', '10-15 min', '15-30 min', '30-60 min', '60+ min'],
        duplicates='drop'
    )

    df['hashtag_category'] = pd.cut(
        df['hashtag_count'],
        bins=[-1, 0, 3, 5, 10, 10.01],
        labels=['0', '1-3', '4-5', '6-10', '10+']
    )

    df['like_ratio'] = df['likes'] / df['views']
    df['comment_ratio'] = df['comment_count'] / df['views']
    df['log_views'] = np.log1p(df['views'])

    return df


# Strona ogólnej analizy sukcesu
def show_general_success_page(df):
    st.header("Ogólna Analiza Sukcesu na YouTube")

    # Definiujemy porównywalne nazwy cech na początku funkcji, aby były dostępne w całym jej zakresie
    feature_names = {
        'duration_seconds': 'Czas trwania (s)',
        'likes': 'Polubienia',
        'comment_count': 'Liczba komentarzy',
        'hashtag_count': 'Liczba hashtagów',
        'creator_total_subscribers': 'Subskrybenci kanału',
        'creator_video_count': 'Liczba filmów na kanale',
        'creator_community_engagement': 'Zaangażowanie społeczności'
    }

    with st.spinner('Analizowanie wzorców sukcesu...'):
        insights = analyze_success_patterns(df)

    # Filtry globalne
    st.sidebar.subheader("Filtry Analizy")

    # Filtry języka, jeśli są dostępne
    if 'language' in df.columns:
        languages = df['language'].dropna().unique()
        languages = sorted([lang for lang in languages if lang])

        if languages:
            selected_languages = st.sidebar.multiselect(
                "Filtruj według języka",
                options=["Wszystkie"] + languages,
                default=["Wszystkie"]
            )

            if selected_languages and "Wszystkie" not in selected_languages:
                df_filtered = df[df['language'].isin(selected_languages)]
                st.sidebar.info(f"Filtrowanie dla języków: {', '.join(selected_languages)}")
            else:
                df_filtered = df
        else:
            df_filtered = df
    else:
        df_filtered = df

    # Dodanie zakładek dla lepszej organizacji
    tabs = st.tabs(["Podsumowanie", "Czas Trwania", "Języki", "Zaangażowanie", "Hashtagi", "Analiza Sukcesu"])

    with tabs[0]:  # Podsumowanie
        # Podsumowanie danych
        st.subheader("📊 Podsumowanie Danych")

        # Statystyki w karcie
        with st.container():
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Liczba Filmów", f"{len(df_filtered):,}")

            with col2:
                st.metric("Średnia Wyświetleń", f"{int(df_filtered['views'].mean()):,}")

            with col3:
                st.metric("Mediana Wyświetleń", f"{int(df_filtered['views'].median()):,}")

            with col4:
                if 'is_successful' in df_filtered.columns:
                    success_rate = df_filtered['is_successful'].mean() * 100
                    st.metric("Procent Filmów z Sukcesem", f"{success_rate:.1f}%")

        # Próg sukcesu
        if 'success_threshold' in insights:
            st.info(
                f"📈 **Próg sukcesu**: Film uznajemy za sukces, gdy ma co najmniej **{int(insights['success_threshold']):,}** wyświetleń (górne 25% filmów).")

        # Korelacja między czynnikami (nowa sekcja)
        if 'correlation' in insights and 'views_correlation' in insights['correlation']:
            st.subheader("📊 Korelacje z Liczbą Wyświetleń")

            views_corr = insights['correlation']['views_correlation']

            if not views_corr.empty:
                # Tworzymy DataFrame do wyświetlenia
                corr_df = pd.DataFrame({
                    'Czynnik': views_corr.index,
                    'Korelacja': views_corr.values
                })

                # Bardziej przyjazne nazwy czynników
                factor_names = {
                    'duration_seconds': 'Czas trwania (s)',
                    'likes': 'Polubienia',
                    'comment_count': 'Liczba komentarzy',
                    'hashtag_count': 'Liczba hashtagów',
                    'creator_total_subscribers': 'Subskrybenci kanału',
                    'creator_video_count': 'Liczba filmów na kanale',
                    'creator_community_engagement': 'Zaangażowanie społeczności'
                }

                corr_df['Czynnik'] = corr_df['Czynnik'].map(lambda x: feature_names.get(x, x))

                # Sortowanie według wartości bezwzględnej korelacji (najsilniejsze na górze)
                corr_df['Abs_Corr'] = abs(corr_df['Korelacja'])
                corr_df = corr_df.sort_values('Abs_Corr', ascending=False).drop(columns=['Abs_Corr'])

                # Wykres słupkowy z korelacjami
                fig = px.bar(
                    corr_df,
                    x='Korelacja',
                    y='Czynnik',
                    title="Korelacja czynników z liczbą wyświetleń",
                    color='Korelacja',
                    color_continuous_scale=['red', 'white', 'green'],  # Czerwony dla negatywnej, zielony dla pozytywnej
                    range_color=[-1, 1],  # Zakres korelacji od -1 do 1
                    orientation='h'  # Poziomy układ
                )

                # Dostosowanie wykresu
                fig.update_layout(
                    xaxis_title="Współczynnik korelacji",
                    yaxis_title="",
                    xaxis=dict(tickvals=[-1, -0.5, 0, 0.5, 1])
                )

                st.plotly_chart(fig, use_container_width=True)

                # Komentarze na temat korelacji
                if 'correlation_insights' in insights and insights['correlation_insights']:
                    with st.expander("📝 Interpretacja korelacji", expanded=True):
                        for insight in insights['correlation_insights']:
                            st.write(insight)

                        st.info(
                            "**Uwaga**: Korelacja nie oznacza przyczynowości. Silna korelacja wskazuje jedynie na związek między zmiennymi, nie na to, że jedna zmienna powoduje zmiany w drugiej.")

        # Podsumowanie głównych czynników sukcesu
        st.subheader("🌟 Kluczowe Czynniki Sukcesu")

        # Zbieramy wszystkie dostępne insighty
        success_factors = []

        # Dodaj informację o progu sukcesu (zawsze dostępna)
        if 'success_threshold' in insights:
            success_factors.append(("📈 Próg sukcesu",
                                    f"Film uznajemy za sukces, gdy ma co najmniej **{int(insights['success_threshold']):,}** wyświetleń (górne 25% filmów)."))

        if 'duration_insights' in insights and 'comment' in insights['duration_insights'] and \
                insights['duration_insights']['comment']:
            success_factors.append(("⏱️ Czas trwania", insights['duration_insights']['comment']))
        elif 'best_duration' in insights:
            success_factors.append(
                ("⏱️ Czas trwania", f"Najlepiej sprawdzają się filmy o długości **{insights['best_duration']}**."))

        if 'language_insights' in insights and 'comment' in insights['language_insights'] and \
                insights['language_insights']['comment']:
            success_factors.append(("🌐 Język", insights['language_insights']['comment']))
        elif 'top_languages' in insights and len(insights['top_languages']) > 0:
            top_language = insights['top_languages'].iloc[0]['language']
            success_factors.append(("🌐 Język", f"Najlepiej sprawdza się język **{top_language}**."))

        if 'engagement_insights' in insights and 'comment' in insights['engagement_insights'] and \
                insights['engagement_insights']['comment']:
            success_factors.append(("👥 Zaangażowanie społeczności", insights['engagement_insights']['comment']))
        elif 'engagement_analysis' in insights and len(insights['engagement_analysis']) > 0:
            best_engagement = \
                insights['engagement_analysis'].iloc[insights['engagement_analysis']['mean'].argmax()][
                    'engagement_category']
            success_factors.append(("👥 Zaangażowanie społeczności",
                                    f"Najlepiej sprawdza się zaangażowanie społeczności na poziomie **{best_engagement}** postów tygodniowo."))

        if 'video_count_insights' in insights and 'comment' in insights['video_count_insights'] and \
                insights['video_count_insights']['comment']:
            success_factors.append(("📼 Liczba filmów na kanale", insights['video_count_insights']['comment']))
        elif 'video_count_analysis' in insights and len(insights['video_count_analysis']) > 0:
            best_video_count = \
                insights['video_count_analysis'].iloc[insights['video_count_analysis']['mean'].argmax()][
                    'video_count_category']
            success_factors.append(
                ("📼 Liczba filmów na kanale", f"Najlepiej sprawdzają się kanały z **{best_video_count}** filmami."))

        if 'hashtag_insights' in insights and 'comment' in insights['hashtag_insights'] and \
                insights['hashtag_insights']['comment']:
            success_factors.append(("🔖 Hashtagi", insights['hashtag_insights']['comment']))
        elif 'hashtag_analysis' in insights and len(insights['hashtag_analysis']) > 0:
            best_hashtag_count = insights['hashtag_analysis'].iloc[insights['hashtag_analysis']['mean'].argmax()][
                'hashtag_category']
            success_factors.append(
                ("🔖 Hashtagi", f"Najlepiej sprawdzają się filmy z **{best_hashtag_count}** hashtagami."))

            # Dodajemy czynnik na podstawie porównania cech (zawsze jeśli mamy dane)
        if 'success_vs_regular' in insights and len(insights['success_vs_regular']) > 0:
            # Znajdowanie cechy z największą różnicą
            features_with_diffs = []
            for feature, values in insights['success_vs_regular'].items():
                if 'difference_pct' in values and pd.notna(values['difference_pct']):
                    features_with_diffs.append((feature, values['difference_pct']))

            if features_with_diffs:
                top_feature, diff_pct = max(features_with_diffs, key=lambda x: abs(x[1]))
                feature_name = feature_names.get(top_feature, top_feature)

                if diff_pct > 0:
                    success_factors.append(
                        ("🔄 Największa różnica",
                         f"Filmy z sukcesem mają o **{diff_pct:.1f}%** wyższą wartość cechy **{feature_name}** niż pozostałe filmy.")
                    )
                else:
                    success_factors.append(
                        ("🔄 Największa różnica",
                         f"Filmy z sukcesem mają o **{abs(diff_pct):.1f}%** niższą wartość cechy **{feature_name}** niż pozostałe filmy.")
                    )

            # Dodanie rekomendacji na podstawie korelacji
            if 'correlation' in insights and 'top_positive' in insights['correlation'] and not insights['correlation'][
                'top_positive'].empty:
                top_corr_feature = insights['correlation']['top_positive'].index[0]
                top_corr_value = insights['correlation']['top_positive'].values[0]

                if top_corr_value > 0.1:  # Nawet słaba korelacja może być interesująca
                    feature_name = feature_names.get(top_corr_feature, top_corr_feature)

                if top_corr_value > 0.5:
                    strength = "silna"
                elif top_corr_value > 0.3:
                    strength = "średnia"
                else:
                    strength = "słaba"

                success_factors.append(
                    ("📊 Korelacja",
                     f"Istnieje **{strength}** pozytywna korelacja ({top_corr_value:.2f}) między liczbą wyświetleń a cechą **{feature_name}**.")
                )

        if success_factors:
            # Wyświetlamy karty z czynnikami sukcesu
            for title, comment in success_factors:
                st.write(f"**{title}**: {comment}")
        else:
            st.info("Brak wystarczających danych do określenia kluczowych czynników sukcesu.")

            # Dodaj sugestie, co można zrobić, aby uzyskać lepsze insighty
            st.write("Aby uzyskać więcej insightów, spróbuj:")
            st.write("1. Dodać więcej danych do analizy")
            st.write(
                "2. Upewnić się, że dane zawierają zróżnicowane wartości w kolumnach takich jak liczba hashtagów, czas trwania, itp.")
            st.write("3. Sprawdzić, czy dane zawierają filmy z różnych kategorii i języków")

    with tabs[1]:  # Czas Trwania
        # 1. Czas trwania a sukces
        st.subheader("⏱️ Optymalny Czas Trwania Filmu")

        if 'duration_analysis' in insights:
            # Dodaj filtr dla wykresu czasu trwania
            duration_df = insights['duration_analysis']

            # Wykres
            fig = px.bar(
                duration_df,
                x='duration_category',
                y='mean',
                title="Średnia liczba wyświetleń według czasu trwania filmu",
                labels={'duration_category': 'Czas trwania', 'mean': 'Średnia liczba wyświetleń'},
                color='mean',
                text_auto='.2s',
                custom_data=['count', 'median']  # Dodatkowe dane dla tooltipa
            )

            # Dostosowanie tooltipa
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>Średnia wyświetleń: %{y:,.0f}<br>Mediana wyświetleń: %{customdata[1]:,.0f}<br>Liczba filmów: %{customdata[0]}"
            )

            st.plotly_chart(fig, use_container_width=True)

            if 'best_duration' in insights:
                best_duration = insights['best_duration']

                # Dodanie komentarza analitycznego
                if 'duration_insights' in insights and 'comment' in insights['duration_insights']:
                    st.success(f"🏆 {insights['duration_insights']['comment']}")
                else:
                    st.success(f"🏆 Najlepiej sprawdzają się filmy o długości **{best_duration}**.")

                # Dodatkowe statystyki w ekspanderze
                with st.expander("📊 Szczegółowe statystyki"):
                    st.dataframe(duration_df)
        else:
            st.info("Niewystarczające dane do analizy wpływu czasu trwania na wyświetlenia.")

    with tabs[2]:  # Języki
        # 2. Język a sukces
        st.subheader("🌐 Najpopularniejsze Języki")

        if 'top_languages' in insights:
            # Wykres
            language_df = insights['top_languages']

            # Dodajemy kolumnę całkowitych wyświetleń dla każdego języka
            if 'sum' not in language_df.columns:
                language_df['sum'] = language_df['mean'] * language_df['count']

            # Opcje sortowania
            sort_options = ["Średnia liczba wyświetleń", "Całkowita liczba wyświetleń", "Liczba filmów"]
            sort_by = st.radio("Sortuj według:", sort_options, horizontal=True)

            if sort_by == "Średnia liczba wyświetleń":
                language_df = language_df.sort_values('mean', ascending=False)
                y_column = 'mean'
                title = "Średnia liczba wyświetleń według języka"
                y_label = "Średnia liczba wyświetleń"
            elif sort_by == "Całkowita liczba wyświetleń":
                language_df = language_df.sort_values('sum', ascending=False)
                y_column = 'sum'
                title = "Całkowita liczba wyświetleń według języka"
                y_label = "Całkowita liczba wyświetleń"
            else:  # Liczba filmów
                language_df = language_df.sort_values('count', ascending=False)
                y_column = 'count'
                title = "Liczba filmów według języka"
                y_label = "Liczba filmów"

            # Ograniczamy do top 10 języków dla przejrzystości
            language_df = language_df.head(10)

            fig = px.bar(
                language_df,
                x='language',
                y=y_column,
                title=title,
                labels={'language': 'Język', y_column: y_label},
                color=y_column,
                text_auto='.2s',
            )

            # Bezpieczne dodanie tooltipów
            tooltip_data = []
            tooltip_template = "<b>%{x}</b><br>" + f"{y_label}: %{{y:,.0f}}<br>Liczba filmów: %{{customdata[0]}}"

            if 'count' in language_df.columns:
                tooltip_data.append('count')

                if 'median' in language_df.columns and y_column != 'median':
                    tooltip_data.append('median')
                    tooltip_template += "<br>Mediana wyświetleń: %{customdata[1]:,.0f}"

            if tooltip_data:
                fig.update_traces(
                    hovertemplate=tooltip_template,
                    customdata=language_df[tooltip_data].values
                )

            st.plotly_chart(fig, use_container_width=True)

            # Dodanie komentarza analitycznego
            if 'language_insights' in insights and 'comment' in insights['language_insights']:
                st.success(f"🏆 {insights['language_insights']['comment']}")
            else:
                top_language = language_df.iloc[0]['language']
                st.success(f"🏆 Najlepiej sprawdza się język **{top_language}**.")

            # Dodatkowe statystyki w ekspanderze
            with st.expander("📊 Szczegółowe statystyki językowe"):
                st.dataframe(language_df)
        else:
            st.info("Niewystarczające dane do analizy wpływu języka na wyświetlenia.")

    with tabs[3]:  # Zaangażowanie
        # 3. Zaangażowanie społeczności a sukces
        st.subheader("👥 Wpływ Zaangażowania Społeczności")

        if 'engagement_analysis' in insights:
            engagement_df = insights['engagement_analysis']

            # Wykres
            fig = px.bar(
                engagement_df,
                x='engagement_category',
                y='mean',
                title="Średnia liczba wyświetleń według zaangażowania społeczności (postów na tydzień)",
                labels={'engagement_category': 'Posty na tydzień', 'mean': 'Średnia liczba wyświetleń'},
                color='mean',
                text_auto='.2s',
            )

            # Bezpieczne dodanie tooltipów
            tooltip_data = []
            tooltip_template = "<b>%{x}</b><br>Średnia wyświetleń: %{y:,.0f}<br>Liczba filmów: %{customdata[0]}"

            if 'count' in engagement_df.columns:
                tooltip_data.append('count')

                if 'median' in engagement_df.columns:
                    tooltip_data.append('median')
                    tooltip_template += "<br>Mediana wyświetleń: %{customdata[1]:,.0f}"

            if tooltip_data:
                fig.update_traces(
                    hovertemplate=tooltip_template,
                    customdata=engagement_df[tooltip_data].values
                )

            st.plotly_chart(fig, use_container_width=True)

            # Dodanie komentarza analitycznego
            if 'engagement_insights' in insights and 'comment' in insights['engagement_insights']:
                st.success(f"🏆 {insights['engagement_insights']['comment']}")
            else:
                best_engagement = engagement_df.iloc[engagement_df['mean'].argmax()]['engagement_category']
                st.success(
                    f"🏆 Najlepiej sprawdza się zaangażowanie społeczności na poziomie **{best_engagement}** postów na tydzień.")

            # Szczegółowe dane
            with st.expander("📊 Szczegółowe statystyki zaangażowania"):
                st.dataframe(engagement_df)

            # Dodatkowe wyjaśnienie jeśli są tylko ograniczone dane
            if len(engagement_df) <= 2:
                st.info(
                    "⚠️ W danych występuje niewiele różnych poziomów zaangażowania społeczności. Dla bardziej szczegółowej analizy potrzebne są bardziej zróżnicowane dane.")
        else:
            st.info("Niewystarczające dane do analizy wpływu zaangażowania społeczności na wyświetlenia.")

        # 4. Wpływ liczby filmów na kanale na sukces
        st.subheader("📼 Wpływ Liczby Filmów na Kanale")

        if 'video_count_analysis' in insights:
            video_count_df = insights['video_count_analysis']

            # Jeśli mamy mapowanie kwantyli na zakresy, dodajemy bardziej zrozumiałe etykiety
            if 'video_count_quantile_ranges' in insights:
                # Tworzymy mapowanie kategorii na opisy
                category_mapping = insights['video_count_quantile_ranges']

                # Tworzymy nową kolumnę z opisowymi etykietami
                video_count_df['display_category'] = video_count_df['video_count_category'].map(
                    lambda x: f"{x} ({category_mapping.get(x, '')})" if x in category_mapping else x
                )
            else:
                video_count_df['display_category'] = video_count_df['video_count_category']

                # Wykres
                fig = px.bar(
                    video_count_df,
                    x='display_category',
                    y='mean',
                    title="Średnia liczba wyświetleń według liczby filmów na kanale",
                    labels={'display_category': 'Liczba filmów', 'mean': 'Średnia liczba wyświetleń'},
                    color='mean',
                    text_auto='.2s',
                )

                # Bezpieczne dodanie tooltipów
                tooltip_data = []
                tooltip_template = "<b>%{x}</b><br>Średnia wyświetleń: %{y:,.0f}<br>Liczba filmów: %{customdata[0]}"

                if 'count' in video_count_df.columns:
                    tooltip_data.append('count')

                    if 'median' in video_count_df.columns:
                        tooltip_data.append('median')
                        tooltip_template += "<br>Mediana wyświetleń: %{customdata[1]:,.0f}"

                if tooltip_data:
                    fig.update_traces(
                        hovertemplate=tooltip_template,
                        customdata=video_count_df[tooltip_data].values
                    )

            st.plotly_chart(fig, use_container_width=True)

            # Dodanie komentarza analitycznego
            if 'video_count_insights' in insights and 'comment' in insights['video_count_insights']:
                st.success(f"🏆 {insights['video_count_insights']['comment']}")
            else:
                best_video_count = video_count_df.iloc[video_count_df['mean'].argmax()]['video_count_category']
                st.success(f"🏆 Najlepiej sprawdzają się kanały z **{best_video_count}** filmami.")

            # Szczegółowe dane
            with st.expander("📊 Szczegółowe statystyki liczby filmów"):
                st.dataframe(video_count_df)
        else:
            st.info("Niewystarczające dane do analizy wpływu liczby filmów na wyświetlenia.")

    with tabs[4]:  # Hashtagi
        # 5. Wpływ hashtagów na sukces
        st.subheader("🔖 Wpływ Hashtagów")

        if 'hashtag_analysis' in insights:
            hashtag_df = insights['hashtag_analysis']

            # Wykres
            fig = px.bar(
                hashtag_df,
                x='hashtag_category',
                y='mean',
                title="Średnia liczba wyświetleń według liczby hashtagów",
                labels={'hashtag_category': 'Liczba hashtagów', 'mean': 'Średnia liczba wyświetleń'},
                color='mean',
                text_auto='.2s',
            )

            # Bezpieczne dodanie tooltipów
            tooltip_data = []
            tooltip_template = "<b>%{x}</b><br>Średnia wyświetleń: %{y:,.0f}<br>Liczba filmów: %{customdata[0]}"

            if 'count' in hashtag_df.columns:
                tooltip_data.append('count')

                if 'median' in hashtag_df.columns:
                    tooltip_data.append('median')
                    tooltip_template += "<br>Mediana wyświetleń: %{customdata[1]:,.0f}"

            if tooltip_data:
                fig.update_traces(
                    hovertemplate=tooltip_template,
                    customdata=hashtag_df[tooltip_data].values
                )

            st.plotly_chart(fig, use_container_width=True)

            # Dodanie komentarza analitycznego
            if 'hashtag_insights' in insights and 'comment' in insights['hashtag_insights']:
                st.success(f"🏆 {insights['hashtag_insights']['comment']}")
            else:
                best_hashtag_count = hashtag_df.iloc[hashtag_df['mean'].argmax()]['hashtag_category']
                st.success(f"🏆 Najlepiej sprawdzają się filmy z **{best_hashtag_count}** hashtagami.")

            # Szczegółowe dane
            with st.expander("📊 Szczegółowe statystyki hashtagów"):
                st.dataframe(hashtag_df)

            # Dodatkowe wyjaśnienie jeśli są tylko ograniczone dane
            if len(hashtag_df) <= 1:
                st.warning(
                    "⚠️ W danych występuje niewiele różnych wartości liczby hashtagów. Dla bardziej szczegółowej analizy potrzebne są bardziej zróżnicowane dane.")

                # Histogram liczby hashtagów
                if 'hashtag_count' in df_filtered.columns:
                    hashtag_counts = df_filtered['hashtag_count'].dropna()
                    if not hashtag_counts.empty:
                        fig = px.histogram(
                            hashtag_counts,
                            title="Rozkład liczby hashtagów w filmach",
                            labels={'value': 'Liczba hashtagów', 'count': 'Liczba filmów'},
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.info(f"Średnia liczba hashtagów: {hashtag_counts.mean():.2f}")
                        st.info(f"Mediana liczby hashtagów: {hashtag_counts.median()}")
                        st.info(
                            f"Najczęstsza liczba hashtagów: {hashtag_counts.mode().iloc[0] if not hashtag_counts.mode().empty else 'Brak danych'}")
        else:
            st.info("Niewystarczające dane do analizy wpływu hashtagów na wyświetlenia.")

    with tabs[5]:  # Analiza Sukcesu
        # 6. Porównanie cech filmów odnoszących sukces i pozostałych
        st.subheader("⚔️ Co Wyróżnia Filmy Odnoszące Sukces?")

        if 'success_vs_regular' in insights:
            comparison_data = []
            for feature, values in insights['success_vs_regular'].items():
                # Upewnij się, że wartości nie są NaN
                if pd.notna(values['successful_mean']) and pd.notna(values['regular_mean']) and pd.notna(
                        values['difference_pct']):
                    comparison_data.append({
                        'Cecha': feature,
                        'Filmy z Sukcesem': values['successful_mean'],
                        'Pozostałe Filmy': values['regular_mean'],
                        'Różnica %': values['difference_pct']
                    })

            # Jeśli nie mamy żadnych danych do porównania, dodajmy jakieś informacje
            if not comparison_data:
                st.warning(
                    "Nie znaleziono wystarczających danych do porównania cech między filmami z sukcesem a pozostałymi.")
                if 'success_threshold' in insights:
                    st.info(
                        f"Filmy z sukcesem to te, które mają powyżej {int(insights['success_threshold']):,} wyświetleń.")
            else:
                comparison_df = pd.DataFrame(comparison_data)

                comparison_df['Cecha'] = comparison_df['Cecha'].map(lambda x: feature_names.get(x, x))
                comparison_df = comparison_df.sort_values('Różnica %', ascending=False)

                if len(comparison_df) > 0:
                    # Wykres
                    fig = px.bar(
                        comparison_df,
                        x='Cecha',
                        y='Różnica %',
                        title="Procentowa różnica między filmami odnoszącymi sukces a pozostałymi",
                        color='Różnica %',
                        color_continuous_scale=['red', 'white', 'green'],
                        # Czerwony dla negatywnych, zielony dla pozytywnych
                        text_auto='.1f',
                    )

                    # Dodaj ograniczenia osi Y dla lepszej czytelności
                    y_values = comparison_df['Różnica %'].values
                    if len(y_values) > 0:
                        # Ustaw granice osi Y na podstawie danych, ale z rozsądnymi limitami
                        y_min = max(-200, min(y_values) * 1.1)
                        y_max = min(500, max(y_values) * 1.1)

                        # Upewnij się, że przedział nie jest zbyt mały
                        if abs(y_max - y_min) < 50:
                            if y_min < 0:
                                y_min = min(-50, y_min * 1.5)
                            if y_max > 0:
                                y_max = max(50, y_max * 1.5)

                        fig.update_layout(yaxis_range=[y_min, y_max])

                    # Bezpieczne dodanie tooltipów
                    if 'Filmy z Sukcesem' in comparison_df.columns and 'Pozostałe Filmy' in comparison_df.columns:
                        fig.update_traces(
                            hovertemplate="<b>%{x}</b><br>Różnica: %{y:.1f}%<br>Filmy z sukcesem: %{customdata[0]:.2f}<br>Pozostałe filmy: %{customdata[1]:.2f}",
                            customdata=comparison_df[['Filmy z Sukcesem', 'Pozostałe Filmy']].values
                        )

                    st.plotly_chart(fig, use_container_width=True)

                    # Bardziej szczegółowa tabela
                    st.subheader("Szczegółowe Porównanie Cech")

                    # Formatowanie liczb w tabeli dla lepszej czytelności
                    formatted_comparison = comparison_df.copy()
                    formatted_comparison['Filmy z Sukcesem'] = formatted_comparison['Filmy z Sukcesem'].apply(
                        lambda x: f"{x:.2f}")
                    formatted_comparison['Pozostałe Filmy'] = formatted_comparison['Pozostałe Filmy'].apply(
                        lambda x: f"{x:.2f}")
                    formatted_comparison['Różnica %'] = formatted_comparison['Różnica %'].apply(lambda x: f"{x:.1f}%")

                    st.dataframe(formatted_comparison, hide_index=True)

                    # Analiza cech z największymi różnicami
                    if len(comparison_df) > 0:
                        top_feature = comparison_df.iloc[0]['Cecha']
                        top_diff = comparison_df.iloc[0]['Różnica %']

                        if top_diff > 0:
                            st.success(
                                f"🔍 **Najważniejsza różnica**: Filmy odnoszące sukces mają o **{top_diff:.1f}%** wyższą wartość cechy **{top_feature}** niż pozostałe filmy.")
                        else:
                            st.info(
                                f"🔍 **Najważniejsza różnica**: Filmy odnoszące sukces mają o **{abs(top_diff):.1f}%** niższą wartość cechy **{top_feature}** niż pozostałe filmy.")
        else:
            st.info("Niewystarczające dane do porównania cech filmów odnoszących sukces.")

        # Podsumowanie
        st.subheader("📝 Podsumowanie i Rekomendacje")

        recommendations = []

        if 'duration_insights' in insights and 'best_category' in insights['duration_insights']:
            recommendations.append(f"✅ Twórz filmy o długości **{insights['duration_insights']['best_category']}**.")
        elif 'best_duration' in insights:
            recommendations.append(f"✅ Twórz filmy o długości **{insights['best_duration']}**.")

        if 'language_insights' in insights and 'top_language' in insights['language_insights']:
            recommendations.append(
                f"✅ Jeśli to możliwe, twórz treści w języku **{insights['language_insights']['top_language']}**.")
        elif 'top_languages' in insights and len(insights['top_languages']) > 0:
            top_language = insights['top_languages'].iloc[0]['language']
            recommendations.append(f"✅ Jeśli to możliwe, twórz treści w języku **{top_language}**.")

        if 'hashtag_insights' in insights and 'best_category' in insights['hashtag_insights']:
            recommendations.append(
                f"✅ Używaj **{insights['hashtag_insights']['best_category']}** hashtagów w swoich filmach.")
        elif 'hashtag_analysis' in insights:
            best_hashtag_count = insights['hashtag_analysis'].iloc[insights['hashtag_analysis']['mean'].argmax()][
                'hashtag_category']
            recommendations.append(f"✅ Używaj **{best_hashtag_count}** hashtagów w swoich filmach.")

        if 'engagement_insights' in insights and 'best_category' in insights['engagement_insights']:
            recommendations.append(
                f"✅ Utrzymuj aktywność na poziomie **{insights['engagement_insights']['best_category']}** postów społecznościowych tygodniowo.")
        elif 'engagement_analysis' in insights:
            best_engagement = insights['engagement_analysis'].iloc[insights['engagement_analysis']['mean'].argmax()][
                'engagement_category']
            recommendations.append(
                f"✅ Utrzymuj aktywność na poziomie **{best_engagement}** postów społecznościowych tygodniowo.")

        if 'video_count_insights' in insights and 'best_category' in insights['video_count_insights']:
            recommendations.append(
                f"✅ Dąż do posiadania **{insights['video_count_insights']['best_category']}** filmów na swoim kanale.")
        elif 'video_count_analysis' in insights:
            best_video_count = insights['video_count_analysis'].iloc[insights['video_count_analysis']['mean'].argmax()][
                'video_count_category']
            recommendations.append(f"✅ Dąż do posiadania **{best_video_count}** filmów na swoim kanale.")

        if 'success_vs_regular' in insights and len(insights['success_vs_regular']) > 0:
            # Znajdowanie cechy z największą różnicą
            top_feature = max(insights['success_vs_regular'].items(), key=lambda x: x[1]['difference_pct'])
            feature_name = feature_names.get(top_feature[0], top_feature[0])
            diff_pct = top_feature[1]['difference_pct']

            if diff_pct > 50:  # Tylko jeśli różnica jest znacząca
                recommendations.append(
                    f"✅ Skup się na zwiększaniu **{feature_name}** - filmy z sukcesem mają o **{diff_pct:.1f}%** wyższą wartość tej cechy.")

        # Dodanie rekomendacji na podstawie korelacji
        if 'correlation' in insights and 'top_positive' in insights['correlation'] and not insights['correlation'][
            'top_positive'].empty:
            top_corr_feature = insights['correlation']['top_positive'].index[0]
            top_corr_value = insights['correlation']['top_positive'].values[0]

            if top_corr_value > 0.1:  # Nawet słaba korelacja może być interesująca
                feature_name = feature_names.get(top_corr_feature, top_corr_feature)
                recommendations.append(
                    f"✅ Zwróć uwagę na **{feature_name}** - ma najsilniejszą pozytywną korelację ({top_corr_value:.2f}) z liczbą wyświetleń.")

        # Wyświetlenie rekomendacji
        if recommendations:
            for rec in recommendations:
                st.markdown(rec)
        else:
            st.info("Niewystarczające dane do wygenerowania szczegółowych rekomendacji.")

        # Przycisk do trenowania modelu
        if st.button("Trenuj Model ML do Przewidywania Sukcesu"):
            with st.spinner('Trenowanie modelu uczenia maszynowego...'):
                try:
                    from sklearn.impute import SimpleImputer
                    model, feature_importance, metrics = train_model(df_filtered)

                    st.success(f"Model został wytrenowany! R² = {metrics['r2']:.3f}")

                    # Wyświetlenie ważności cech
                    st.subheader("Ważność Cech w Modelu")

                    # Ograniczenie do 15 najważniejszych cech
                    feature_importance = feature_importance.head(15)

                    fig = px.bar(
                        feature_importance,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="15 najważniejszych cech wpływających na liczbę wyświetleń",
                        labels={'importance': 'Ważność', 'feature': 'Cecha'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Wystąpił błąd podczas trenowania modelu: {str(e)}")


def show_single_video_analysis_page(df):
    st.header("Analiza Pojedynczego Filmu")

    # Lista filmów do wyboru
    video_options = df[['_key', 'title', 'views']].sort_values('views', ascending=False)

    # Tworzymy etykietę wyświetlaną bez klucza
    video_options['display_option'] = video_options.apply(
        lambda x: f"{x['title']} - {int(x['views']):,} wyświetleń", axis=1
    )

    # Tworzymy słownik mapujący etykietę na _key
    video_dict = dict(zip(video_options['display_option'], video_options['_key']))

    # Wybór filmu z listy etykiet
    selected_option = st.selectbox(
        "Wybierz film do analizy:",
        options=list(video_dict.keys())
    )

    if selected_option:
        # Wyodrębnienie _key z wybranej opcji
        video_id = video_dict[selected_option]

        with st.spinner('Analizowanie filmu...'):
            analysis = analyze_video(df, video_id)

        if 'error' in analysis:
            st.error(analysis['error'])
        else:
            # Panel z podstawowymi informacjami
            st.subheader("📺 Podstawowe Informacje")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Wyświetlenia", f"{int(analysis['basic_info']['views']):,}")

                # Ocena sukcesu
                if 'success_rating' in analysis:
                    top_percent = analysis['success_rating']['top_percent']
                    # Komunikaty dostosowane do pozycji filmu w rankingu
                    if top_percent <= 1:
                        st.success("🌟 Ten film jest najpopularniejszy na platformie!")
                    elif top_percent <= 10:
                        st.success(f"🌟 Ten film znajduje się w TOP {int(top_percent)}% filmów.")
                    elif top_percent <= 25:
                        st.success(f"👍 Ten film jest bardzo udany i plasuje się w TOP {int(top_percent)}% filmów.")
                    elif top_percent <= 50:
                        st.info(f"😊 Ten film osiąga wyniki lepsze niż {int(100 - top_percent)}% filmów.")
                    else:
                        st.warning(
                            f"🤔 Ten film wymaga poprawy – osiąga wyniki lepsze niż tylko {int(100 - top_percent)}% filmów.")

            with col2:
                st.metric("Polubienia", f"{int(analysis['basic_info']['likes']):,}")
                st.metric("Komentarze", f"{int(analysis['basic_info']['comments']):,}")

            with col3:
                duration_min = analysis['basic_info']['duration_seconds'] / 60
                st.metric("Czas trwania", f"{duration_min:.1f} min")
                st.metric("Język", analysis['basic_info']['language'])

            # Porównanie z podobnymi filmami
            st.subheader("🔍 Porównanie z Podobnymi Filmami")

            if 'language_comparison' in analysis:
                lang_comp = analysis['language_comparison']
                col1, col2 = st.columns(2)

                with col1:
                    comparison_text = "lepiej" if lang_comp['better_than_average'] else "gorzej"
                    comparison_icon = "📈" if lang_comp['better_than_average'] else "📉"
                    st.info(
                        f"{comparison_icon} Film ma {abs(lang_comp['percent_difference']):.1f}% {comparison_text} wyświetleń niż średnia dla filmów w języku {analysis['basic_info']['language']}.")

                with col2:
                    st.metric(
                        "Średnia Wyświetleń dla Języka",
                        f"{int(lang_comp['avg_views']):,}",
                        f"{lang_comp['percent_difference']:.1f}%"
                    )

            # Metryki zaangażowania
            if 'engagement_metrics' in analysis:
                st.subheader("👥 Metryki Zaangażowania")
                eng = analysis['engagement_metrics']

                col1, col2 = st.columns(2)

                with col1:
                    if 'like_ratio' in eng:
                        st.metric(
                            "Współczynnik polubień",
                            f"{eng['like_ratio']:.2f}%",
                            f"{(eng['like_ratio_vs_avg'] - 1) * 100:.1f}% vs średnia"
                        )

                with col2:
                    if 'comment_ratio' in eng:
                        st.metric(
                            "Współczynnik komentarzy",
                            f"{eng['comment_ratio']:.2f}%",
                            f"{(eng['comment_ratio_vs_avg'] - 1) * 100:.1f}% vs średnia"
                        )

            # Sugestie optymalizacji
            if 'optimization_suggestions' in analysis and len(analysis['optimization_suggestions']) > 0:
                st.subheader("🚀 Co Można Poprawić?")

                for suggestion in analysis['optimization_suggestions']:
                    st.info(f"💡 {suggestion}")
            else:
                st.subheader("🚀 Co Można Poprawić?")
                st.success("👍 Ten film jest dobrze zoptymalizowany! Nie znaleziono istotnych sugestii do poprawy.")

            # Podsumowanie
            st.subheader("📊 Podsumowanie")

            if 'success_rating' in analysis:
                top_percent = analysis['success_rating']['top_percent']
                if top_percent <= 25:
                    st.success("Film osiągnął lub przekroczył oczekiwany potencjał. Gratulacje!")
                elif top_percent <= 50:
                    st.info("Film radzi sobie dobrze, ale warto rozważyć dalsze usprawnienia.")
                else:
                    st.warning(
                        "Film nie osiągnął pełnego potencjału. Rozważ zmianę tytułu, miniatury, optymalizację SEO oraz analizę konkurencyjnych treści.")


def show_final_summary_page(df):
    st.header("🚀 Finalne Podsumowanie - Jak Odnieść Sukces na YouTube")

    # Wprowadzenie
    st.markdown("""
    To podsumowanie zawiera kompleksowe rekomendacje i wskazówki, jak prowadzić kanał na YouTube, 
    aby osiągnąć maksymalny sukces. Zalecenia są oparte na analizie danych z bazy, 
    modelowaniu maszynowym oraz zależnościach wykrytych między różnymi czynnikami.
    """)

    # Analiza danych
    with st.spinner('Analizowanie wszystkich danych i generowanie rekomendacji...'):
        # Pobieranie insightów z różnych metod analizy
        insights = analyze_success_patterns(df)

        # Próba załadowania wytrenowanego modelu, jeśli jest dostępny
        model_path = 'models/youtube_success_model.pkl'
        model_available = os.path.exists(model_path)

        if model_available:
            try:
                model = joblib.load(model_path)
                if hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
                    regressor = model.named_steps['regressor']
                    if hasattr(regressor, 'feature_importances_'):
                        # Pobranie nazw cech po transformacji
                        feature_names = []
                        if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
                            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                        else:
                            feature_names = [f"feature_{i}" for i in range(len(regressor.feature_importances_))]

                        # Jeśli jest selektor cech, to korzystamy z niego
                        if 'selector' in model.named_steps:
                            selector = model.named_steps['selector']
                            mask = selector.get_support()
                            feature_names = [f for m, f in zip(mask, feature_names) if m]

                        # Ważność cech
                        feature_importance = pd.DataFrame({
                            'feature': feature_names[:len(regressor.feature_importances_)],
                            'importance': regressor.feature_importances_
                        }).sort_values('importance', ascending=False)
                    else:
                        feature_importance = None
                else:
                    feature_importance = None
            except Exception as e:
                st.error(f"Błąd podczas ładowania modelu: {str(e)}")
                model_available = False
                feature_importance = None
        else:
            feature_importance = None

    # Przewodnik krok po kroku - używamy tabów zamiast ekspanderów
    st.subheader("📋 Krok po Kroku do Sukcesu na YouTube")

    # Używamy tabów zamiast ekspanderów
    tabs = st.tabs([
        "🎯 Krok 1: Wybór tematyki i języka",
        "⏱️ Krok 2: Długość i format",
        "🏷️ Krok 3: Hashtagi i opis",
        "👥 Krok 4: Zaangażowanie społeczności",
        "🔄 Krok 5: Regularne publikowanie",
        "🧪 Krok 6: Testowanie i optymalizacja"
    ])

    with tabs[0]:  # Krok 1
        st.markdown("### Wybór języka i tematyki")

        if 'top_languages' in insights and len(insights['top_languages']) > 0:
            top_language = insights['top_languages'].iloc[0]['language']
            st.success(f"🌍 **Rekomendowany język**: {top_language}")

            # Wykres pokazujący wydajność według języka
            st.markdown("#### Porównanie wydajności według języka:")

            # Pobierz top 5 języków
            top5_langs = insights['top_languages'].head(5)
            fig = px.bar(
                top5_langs,
                x='language',
                y='mean',
                title="Średnia liczba wyświetleń według języka",
                color='mean',
                labels={'language': 'Język', 'mean': 'Średnia liczba wyświetleń'}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Strategia tematyczna:**
        1. Zidentyfikuj niszę z wysokim potencjałem i niższą konkurencją
        2. Sprawdź trendy w wybranym języku używając narzędzi jak Google Trends
        3. Wybierz obszar tematyczny, który możesz konsekwentnie rozwijać przez co najmniej rok
        4. Analizuj konkurencję, aby znaleźć luki tematyczne, które możesz wypełnić
        5. Dostosuj tematykę do języka - niektóre tematy mogą być bardziej popularne w określonych regionach językowych
        """)

    with tabs[1]:  # Krok 2
        st.markdown("### Optymalizacja długości i formatu")

        if 'duration_analysis' in insights and len(insights['duration_analysis']) > 0:
            best_duration = insights['duration_analysis'].iloc[insights['duration_analysis']['mean'].argmax()][
                'duration_category']
            st.success(f"⏱️ **Optymalna długość filmu**: {best_duration}")

            # Wykres pokazujący wydajność według długości
            st.markdown("#### Wpływ długości filmu na wyświetlenia:")
            fig = px.bar(
                insights['duration_analysis'],
                x='duration_category',
                y='mean',
                title="Średnie wyświetlenia według długości filmu",
                color='mean',
                labels={'duration_category': 'Długość filmu', 'mean': 'Średnie wyświetlenia'}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Rekomendacje dotyczące formatu:**
        1. Przygotuj pierwszy hook w pierwszych 15 sekundach, aby przyciągnąć uwagę widzów
        2. Utrzymuj dynamiczne tempo, zmieniając ujęcia co 5-10 sekund
        3. Używaj segmentacji treści, aby umożliwić łatwą nawigację po filmie
        4. Testuj różne formaty (poradniki, reakcje, wywiady) i analizuj, które najlepiej działają dla Twojej grupy odbiorców
        5. Zadbaj o jakość dźwięku - często ważniejszą niż obraz
        6. Stwórz rozpoznawalną strukturę filmów (intro, powitanie, treść, outro)
        """)

    with tabs[2]:  # Krok 3
        st.markdown("### Optymalizacja metadanych")

        if 'hashtag_analysis' in insights and len(insights['hashtag_analysis']) > 0:
            best_hashtags = insights['hashtag_analysis'].iloc[insights['hashtag_analysis']['mean'].argmax()][
                'hashtag_category']
            st.success(f"🔖 **Optymalna liczba hashtagów**: {best_hashtags}")

            # Wykres pokazujący wydajność według liczby hashtagów
            st.markdown("#### Wpływ liczby hashtagów na wyświetlenia:")
            fig = px.bar(
                insights['hashtag_analysis'],
                x='hashtag_category',
                y='mean',
                title="Średnie wyświetlenia według liczby hashtagów",
                color='mean',
                labels={'hashtag_category': 'Liczba hashtagów', 'mean': 'Średnie wyświetlenia'}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Optymalizacja metadanych:**
        1. Używaj najważniejszych słów kluczowych w tytule filmu
        2. Tworząc opis:
           - Umieść najważniejsze informacje w pierwszych 2-3 liniach
           - Dodaj timestampy do dłuższych filmów
           - Umieść linki do powiązanych treści i do Twoich mediów społecznościowych
        3. Wybieraj hashtagi, które są:
           - Popularne, ale nie za bardzo (aby nie zginąć w natłoku treści)
           - Precyzyjne i związane z tematyką filmu
           - Mix popularnych i niszowych hashtagów
        4. Projektuj miniaturki, które:
           - Wyróżniają się kolorystycznie
           - Zawierają wyraźny tekst (maksymalnie 3-4 słowa)
           - Wzbudzają ciekawość, bez clickbaitu
        """)

    with tabs[3]:  # Krok 4
        st.markdown("### Strategie zaangażowania społeczności")

        if 'engagement_analysis' in insights and len(insights['engagement_analysis']) > 0:
            best_engagement = insights['engagement_analysis'].iloc[insights['engagement_analysis']['mean'].argmax()][
                'engagement_category']
            st.success(f"👨‍👩‍👧‍👦 **Optymalna częstotliwość postów społecznościowych**: {best_engagement} tygodniowo")

            # Wykres pokazujący wydajność według zaangażowania społeczności
            st.markdown("#### Wpływ aktywności społecznościowej na wyświetlenia:")
            fig = px.bar(
                insights['engagement_analysis'],
                x='engagement_category',
                y='mean',
                title="Średnie wyświetlenia według zaangażowania społeczności",
                color='mean',
                labels={'engagement_category': 'Posty tygodniowo', 'mean': 'Średnie wyświetlenia'}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Strategie angażowania widzów:**
        1. Zadawaj pytania w filmach, które zachęcają do komentowania
        2. Odpowiadaj na komentarze, szczególnie w pierwszych 24 godzinach po publikacji
        3. Organizuj regularne formaty angażujące społeczność (Q&A, przegląd komentarzy)
        4. Buduj społeczność poza YouTube (Discord, Instagram, itp.)
        5. Konsekwentnie publikuj treści według określonego harmonogramu
        6. Organizuj konkursy i wyzwania dla społeczności
        7. Twórz treści we współpracy z innymi twórcami (collab)
        8. Twórz ankiety i korzystaj z funkcji społecznościowych YouTube
        """)

    with tabs[4]:  # Krok 5
        st.markdown("### Znaczenie konsekwencji w publikowaniu")

        if 'video_count_analysis' in insights and len(insights['video_count_analysis']) > 0:
            best_video_count = insights['video_count_analysis'].iloc[insights['video_count_analysis']['mean'].argmax()][
                'video_count_category']
            st.success(f"📈 **Optymalny rozmiar biblioteki treści**: {best_video_count} filmów")

            # Wykres pokazujący wydajność według liczby filmów
            st.markdown("#### Wpływ liczby filmów na kanale na wyświetlenia:")
            fig = px.bar(
                insights['video_count_analysis'],
                x='video_count_category',
                y='mean',
                title="Średnie wyświetlenia według liczby filmów na kanale",
                color='mean',
                labels={'video_count_category': 'Liczba filmów', 'mean': 'Średnie wyświetlenia'}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Zasady konsekwentnego tworzenia treści:**
        1. Ustal realistyczny harmonogram publikacji (1-3 filmy tygodniowo jest optymalnym tempem dla większości twórców)
        2. Twórz seryjne treści, które budują lojalność widzów
        3. Wykorzystuj narzędzia planowania treści, aby zapewnić regularność
        4. Analizuj metryki, aby ustalić optymalny dzień i godzinę publikacji
        5. Buduj "backlog" filmów, aby zachować regularność nawet w trudnych okresach
        6. Twórz kalendarz treści z wyprzedzeniem miesięcznym lub kwartalnym
        7. Ustal system pracy, który pozwoli Ci efektywnie tworzyć treści
        8. Monitoruj i dostosowuj się do sezonowych trendów
        """)

    with tabs[5]:  # Krok 6
        st.markdown("### Strategie testowania i optymalizacji")

        # Model insights if available
        if model_available and feature_importance is not None:
            st.markdown("#### Najważniejsze czynniki wpływające na sukces według modelu ML:")
            top_features = feature_importance.head(10)
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 czynników sukcesu według modelu uczenia maszynowego",
                labels={'importance': 'Ważność', 'feature': 'Czynnik'}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Strategie optymalizacji opartej na danych:**
        1. Stale monitoruj statystyki w YouTube Studio
        2. Testuj różne:
           - Miniatury (A/B testing)
           - Formaty tytułów
           - Call-to-action w filmach
        3. Analizuj retencję widzów, aby identyfikować momenty, w których widzowie przestają oglądać
        4. Korzystaj z narzędzi zewnętrznych do analizy trendów i konkurencji
        5. Cyklicznie przeglądaj najlepiej działające treści i wyciągaj z nich wnioski
        6. Dostosowuj strategię SEO w oparciu o zmieniające się algorytmy YouTube
        7. Korzystaj z narzędzi analitycznych, aby identyfikować nowe słowa kluczowe
        8. Zbieraj bezpośredni feedback od widzów poprzez ankiety i komentarze
        """)

    # Final summary - POZA TABAMI
    st.markdown("---")
    st.subheader("💎 Podsumowanie Kluczowych Czynników Sukcesu")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Co najbardziej wpływa na liczbę wyświetleń")

        success_factors = []

        if 'best_duration' in insights:
            success_factors.append(f"✅ Optymalna długość filmu: **{insights['best_duration']}**")

        if 'top_languages' in insights and len(insights['top_languages']) > 0:
            top_language = insights['top_languages'].iloc[0]['language']
            success_factors.append(f"✅ Najlepiej performujący język: **{top_language}**")

        if 'hashtag_analysis' in insights and len(insights['hashtag_analysis']) > 0:
            best_hashtags = insights['hashtag_analysis'].iloc[insights['hashtag_analysis']['mean'].argmax()][
                'hashtag_category']
            success_factors.append(f"✅ Optymalna liczba hashtagów: **{best_hashtags}**")

        if 'engagement_analysis' in insights and len(insights['engagement_analysis']) > 0:
            best_engagement = insights['engagement_analysis'].iloc[insights['engagement_analysis']['mean'].argmax()][
                'engagement_category']
            success_factors.append(
                f"✅ Najlepsza częstotliwość postów społecznościowych: **{best_engagement}** tygodniowo")

        if 'video_count_analysis' in insights and len(insights['video_count_analysis']) > 0:
            best_video_count = insights['video_count_analysis'].iloc[insights['video_count_analysis']['mean'].argmax()][
                'video_count_category']
            success_factors.append(f"✅ Optymalny rozmiar biblioteki treści: **{best_video_count}** filmów")

        # Display success factors
        for factor in success_factors:
            st.markdown(factor)

    with col2:
        st.markdown("#### Najważniejsze rekomendacje")

        st.markdown("""
        1. **Konsekwencja** - regularnie publikuj treści według ustalonego harmonogramu
        2. **Jakość** - stawiaj na wartościowe treści, które rozwiązują problemy widzów
        3. **Optymalizacja** - testuj różne podejścia i analizuj dane aby doskonalić strategię
        4. **Zaangażowanie** - buduj społeczność poprzez interakcje z widzami
        5. **Cierpliwość** - sukces na YouTube to maraton, nie sprint - bądź gotów inwestować czas długoterminowo
        """)

    # Additional links and resources
    st.markdown("---")
    st.markdown("### 📚 Dodatkowe Zasoby")
    st.markdown("""
    - [YouTube Creator Academy](https://creatoracademy.youtube.com/)
    - [vidIQ - Narzędzie do analizy YouTube](https://vidiq.com/)
    - [TubeBuddy - Optymalizacja kanału](https://www.tubebuddy.com/)
    - [Social Blade - Statystyki i dane](https://socialblade.com/)
    """)

if __name__ == "__main__":
    main()
