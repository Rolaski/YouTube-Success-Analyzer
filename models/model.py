import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet

def convert_bool_to_str(X):
    return X.astype(str)

class YouTubeSuccessModel:
    def __init__(self, data=None):
        """
        Inicjalizacja modelu predykcji sukcesu na YouTube

        Parameters:
        ----------
        data : pandas.DataFrame, optional
            Dane treningowe
        """
        self.data = data
        self.model = None
        self.preprocessor = None
        self.feature_importance = None
        self.metrics = None

    def prepare_features(self):
        """
        Przygotowanie cech do treningu modelu

        Returns:
        -------
        X : pandas.DataFrame
            Cechy
        y : pandas.Series
            Zmienna celu
        """
        if self.data is None:
            raise ValueError("Dane nie zostały załadowane")

        # Logarytmowanie wyświetleń (zmienna celu) dla lepszego rozkładu
        if 'log_views' in self.data.columns:
            y = self.data['log_views']
        else:
            y = np.log1p(self.data['views'])

        # Usunięcie kolumn, które nie powinny być używane jako cechy
        exclude_cols = ['_id', '_key', '_rev', 'url', 'creator', 'views', 'log_views',
                        'title', 'description', 'hashtags', 'is_successful']

        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        X = self.data[feature_cols].copy()

        return X, y

    def create_preprocessor(self, X):
        """
        Tworzenie preprocessora dla danych

        Parameters:
        ----------
        X : pandas.DataFrame
            Cechy do przetworzenia

        Returns:
        -------
        preprocessor : ColumnTransformer
            Preprocessor dla danych
        """
        # Identyfikacja typów kolumn
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Preprocessor dla cech numerycznych
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Preprocessor dla cech kategorycznych (w tym bool → str)
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('bool_to_str', FunctionTransformer(convert_bool_to_str, feature_names_out="one-to-one")),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Złożenie preprocessorów
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        return preprocessor

    def train(self, algorithm='ensemble', cv=5, n_jobs=-1):
        """
        Trenowanie modelu

        Parameters:
        ----------
        algorithm : str, default='ensemble'
            Algorytm do trenowania: 'rf' (Random Forest), 'gb' (Gradient Boosting),
            'xgb' (XGBoost), 'elastic' (ElasticNet) lub 'ensemble' (Ensemble)
        cv : int, default=5
            Liczba foldów w kroswalidacji
        n_jobs : int, default=-1
            Liczba zadań do wykonania równolegle (-1 = wszystkie dostępne procesory)

        Returns:
        -------
        self : YouTubeSuccessModel
            Wytrenowany model
        """
        # Przygotowanie cech
        X, y = self.prepare_features()

        # Preprocessor
        self.preprocessor = self.create_preprocessor(X)

        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Wybór algorytmu
        if algorithm == 'rf':
            # Random Forest
            param_grid = {
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [None, 10, 20],
                'regressor__min_samples_split': [2, 5, 10]
            }
            model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('selector', SelectFromModel(GradientBoostingRegressor(n_estimators=100))),
                ('regressor', RandomForestRegressor(random_state=42))
            ])

        elif algorithm == 'gb':
            # Gradient Boosting
            param_grid = {
                'regressor__n_estimators': [100, 200],
                'regressor__learning_rate': [0.01, 0.1],
                'regressor__max_depth': [3, 5, 7]
            }
            model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', GradientBoostingRegressor(random_state=42))
            ])

        elif algorithm == 'xgb':
            # XGBoost
            param_grid = {
                'regressor__n_estimators': [100, 200],
                'regressor__learning_rate': [0.01, 0.1],
                'regressor__max_depth': [3, 5, 7]
            }
            model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', XGBRegressor(random_state=42))
            ])

        elif algorithm == 'elastic':
            # ElasticNet
            param_grid = {
                'regressor__alpha': [0.1, 1.0, 10.0],
                'regressor__l1_ratio': [0.1, 0.5, 0.9]
            }
            model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', ElasticNet(random_state=42))
            ])

        elif algorithm == 'ensemble':
            # Prostszy model ensemblowy (połączenie RF i GB)
            model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('selector', SelectFromModel(GradientBoostingRegressor(n_estimators=100))),
                ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
            ])
            param_grid = {}  # Puste grid search dla szybszego treningu

        else:
            raise ValueError(f"Nieznany algorytm: {algorithm}")

        # Trenowanie modelu z grid search, jeśli są parametry
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='r2', n_jobs=n_jobs)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
        else:
            model.fit(X_train, y_train)
            self.model = model

        # Ocena modelu
        y_pred = self.model.predict(X_test)
        self.metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

        # Obliczenie ważności cech
        self.calculate_feature_importance()

        return self

    def calculate_feature_importance(self):
        """
        Obliczenie ważności cech

        Returns:
        -------
        feature_importance : pandas.DataFrame
            DataFrame z ważnością cech
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        # Sprawdzenie, czy model zawiera Random Forest lub Gradient Boosting
        if hasattr(self.model, 'named_steps') and 'regressor' in self.model.named_steps:
            regressor = self.model.named_steps['regressor']

            if hasattr(regressor, 'feature_importances_'):
                # Pobierz nazwy cech po transformacji
                if hasattr(self.model.named_steps['preprocessor'], 'get_feature_names_out'):
                    feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
                else:
                    # Alternatywne podejście, jeśli poprzednie nie zadziała
                    X, _ = self.prepare_features()
                    feature_names = X.columns.tolist()

                # Jeśli jest selektor cech, to korzystamy z niego
                if 'selector' in self.model.named_steps:
                    selector = self.model.named_steps['selector']
                    mask = selector.get_support()
                    feature_names = [f for m, f in zip(mask, feature_names) if m]

                # Ważność cech
                self.feature_importance = pd.DataFrame({
                    'feature': feature_names[:len(regressor.feature_importances_)],
                    'importance': regressor.feature_importances_
                }).sort_values('importance', ascending=False)

                return self.feature_importance

        # Jeśli nie ma metody feature_importances_
        self.feature_importance = pd.DataFrame({'feature': ['unknown'], 'importance': [0]})
        return self.feature_importance

    def predict(self, X):
        """
        Przewidywanie liczby wyświetleń

        Parameters:
        ----------
        X : pandas.DataFrame
            Cechy dla których ma być wykonana predykcja

        Returns:
        -------
        predictions : numpy.ndarray
            Przewidywana liczba wyświetleń (bez logarytmowania)
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        # Predykcja logarytmicznych wyświetleń
        log_predictions = self.model.predict(X)

        # Konwersja z powrotem do oryginalnej skali
        predictions = np.expm1(log_predictions)

        return predictions

    def save(self, path='models/youtube_success_model.pkl'):
        """
        Zapisanie modelu do pliku

        Parameters:
        ----------
        path : str, default='models/youtube_success_model.pkl'
            Ścieżka do zapisu modelu
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        # Stworzenie katalogu, jeśli nie istnieje
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Zapisanie modelu
        joblib.dump(self.model, path)

    def load(self, path='models/youtube_success_model.pkl'):
        """
        Wczytanie modelu z pliku

        Parameters:
        ----------
        path : str, default='models/youtube_success_model.pkl'
            Ścieżka do modelu

        Returns:
        -------
        self : YouTubeSuccessModel
            Wczytany model
        """
        self.model = joblib.load(path)
        return self

    def plot_feature_importance(self, top_n=20):
        """
        Wizualizacja ważności cech

        Parameters:
        ----------
        top_n : int, default=20
            Liczba najważniejszych cech do wyświetlenia

        Returns:
        -------
        fig : matplotlib.figure.Figure
            Wykres
        """
        if self.feature_importance is None:
            raise ValueError("Ważność cech nie została obliczona")

        # Wybierz top_n najważniejszych cech
        top_features = self.feature_importance.head(top_n)

        # Tworzenie wykresu
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_features, ax=ax)
        ax.set_title(f'Top {top_n} najważniejszych cech')
        ax.set_xlabel('Ważność')
        ax.set_ylabel('Cecha')

        plt.tight_layout()
        return fig

    def plot_actual_vs_predicted(self, X, y_true):
        """
        Wykres wartości rzeczywistych vs. przewidywanych

        Parameters:
        ----------
        X : pandas.DataFrame
            Cechy
        y_true : pandas.Series
            Rzeczywiste wartości

        Returns:
        -------
        fig : matplotlib.figure.Figure
            Wykres
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        # Predykcja
        y_pred = self.predict(X)

        # Konwersja y_true z logarytmicznej skali, jeśli potrzeba
        if max(y_true) < 30:  # prawdopodobnie zlogarytmowane wartości
            y_true = np.expm1(y_true)

        # Tworzenie wykresu
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(y_true, y_pred, alpha=0.3)

        # Dodaj linię idealnej predykcji
        max_val = max(max(y_true), max(y_pred))
        ax.plot([0, max_val], [0, max_val], 'r--')

        ax.set_title('Rzeczywiste vs. Przewidywane Wyświetlenia')
        ax.set_xlabel('Rzeczywiste Wyświetlenia')
        ax.set_ylabel('Przewidywane Wyświetlenia')
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.tight_layout()
        return fig


# Przykład użycia (jak moduł)
if __name__ == "__main__":
    from arango import ArangoClient
    from dotenv import load_dotenv
    import os

    # Załaduj plik .env z folderu /database
    env_path = os.path.join(os.path.dirname(__file__), 'database', '.env')
    # Ładowanie zmiennych środowiskowych
    load_dotenv(env_path)

    # Pobranie danych uwierzytelniających ArangoDB z zmiennych środowiskowych
    ARANGO_USERNAME = os.getenv('ARANGO_USERNAME')
    ARANGO_PASSWORD = os.getenv('ARANGO_PASSWORD')
    ARANGO_DATABASE = os.getenv('ARANGO_DATABASE')


    # Funkcja do pobierania danych z ArangoDB
    def get_data_from_arango():
        # Połączenie z ArangoDB
        client = ArangoClient(hosts='http://localhost:8529')
        db = client.db(ARANGO_DATABASE, username=ARANGO_USERNAME, password=ARANGO_PASSWORD)

        # Zapytanie AQL pobierające dane o filmach i ich twórcach
        aql = """
            FOR v IN videos
                LET creator = DOCUMENT(CONCAT('creators/', v.creator_id))
                RETURN MERGE(v, { 
                    creator_name: creator.name,
                    creator_subs: creator.subscribers,
                    creator_videos: creator.video_count,
                    creator_category: creator.category,
                    creator_language: creator.language
                })
            """
        cursor = db.aql.execute(aql)
        data = [doc for doc in cursor]

        # Konwersja do DataFrame
        df = pd.DataFrame(data)

        # Konwersja kolumny 'views' na typ numeryczny (w przypadku błędnych wartości zostaną zamienione na NaN)
        df['views'] = pd.to_numeric(df['views'], errors='coerce')

        # Obliczenie progu sukcesu na podstawie 75 percentyla
        views_threshold = df['views'].quantile(0.75)
        df['is_successful'] = df['views'] >= views_threshold

        # Dodanie logarytmowanej kolumny wyświetleń
        df['log_views'] = np.log1p(df['views'])

        return df


    # Wczytaj dane i wytrenuj model
    df = get_data_from_arango()
    model = YouTubeSuccessModel(df)
    model.train(algorithm='ensemble')

    # Wyświetl metryki i zapisz model
    print(f"Model metrics: {model.metrics}")
    model.save()

    # Wizualizacja najważniejszych cech
    fig = model.plot_feature_importance(top_n=15)
    plt.savefig('results/feature_importance.png')

    # Wizualizacja rzeczywistych vs przewidywanych wartości
    X, y = model.prepare_features()
    fig2 = model.plot_actual_vs_predicted(X, y)
    plt.savefig('results/actual_vs_predicted.png')

    print("Model trained and saved successfully.")