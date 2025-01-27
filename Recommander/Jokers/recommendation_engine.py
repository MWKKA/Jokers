import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import NearestNeighbors
import re

class MovieRecommender:
    def __init__(self, df, tfidf_matrix, genre_encoded):
        self.df = df
        self.tfidf_matrix = tfidf_matrix
        self.genre_encoded = genre_encoded

    @classmethod
    def from_csv(cls, csv_path):
        df = pd.read_csv(csv_path, low_memory=False)

        #TfidfVectorizer sur la colonne 'tokens'
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['tokens'])

        #encodage des genres
        mlb = MultiLabelBinarizer()
        df['genres'] = df['genres'].apply(lambda x: x.split(','))  # Diviser les genres si nécessaire
        genre_encoded = mlb.fit_transform(df['genres'])

        #ajout du résultat dans dataframe
        df['genre_encoded'] = list(genre_encoded)

        return cls(df, tfidf_matrix, genre_encoded)

    def recommend(self, movie_title, start_year=None, n_recommendations=5):
        #lower pour la casse
        movie_title = movie_title.lower()

        #recherche partielle du titre
        movie_row = self.df[self.df['primaryTitle'].str.lower().str.contains(movie_title)]
        
        if movie_row.empty:
            raise ValueError(f"Aucun film trouvé pour le titre contenant '{movie_title}'.")

        #vérification de l'année
        if start_year is not None:
            try:
                start_year = int(start_year)
            except ValueError:
                raise ValueError(f"L'année de sortie '{start_year}' est invalide.")
            
            movie_row = movie_row[movie_row['startYear'] == start_year]

        if movie_row.empty:
            raise ValueError(f"Aucun film trouvé pour l'année {start_year} avec le titre contenant '{movie_title}'.")

        #film a été trouvé = recommandations
        movie_index = movie_row.index[0]

        #calcul similarité avec autres films
        movie_tfidf = self.tfidf_matrix[movie_index]
        similarities = cosine_similarity(movie_tfidf, self.tfidf_matrix).flatten()

        self.df['similarities'] = similarities

        #normalisation des caractéristiques
        numerical_features = self.df[['similarities', 'averageRating', 'numVotes']].fillna(0)
        scaler = StandardScaler()
        numerical_features_scaled = scaler.fit_transform(numerical_features)

        #pondérations
        scaling_factor_avg_rating = 1.2
        scaling_factor_num_votes = 1.1
        genre_weight = 100

        numerical_features_scaled[:, 1] *= scaling_factor_avg_rating
        numerical_features_scaled[:, 2] *= scaling_factor_num_votes

        #genres avec pondération
        genre_encoded = np.array(self.df['genre_encoded'].tolist())
        genre_encoded = genre_encoded.astype(float) * genre_weight

        #combiner les caractéristiques
        X_combined = np.hstack([genre_encoded, numerical_features_scaled])

        nn = NearestNeighbors(n_neighbors=n_recommendations + 1, metric='euclidean')
        nn.fit(X_combined)

        distances, indices = nn.kneighbors([X_combined[movie_index]])
        similar_indices = indices[0][1:]  # Exclure le film lui-même

        #films similaires
        similar_movies = self.df.iloc[similar_indices]
        return similar_movies[['primaryTitle', 'startYear', 'averageRating', 'poster_url']].to_dict(orient='records')