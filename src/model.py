import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from datetime import datetime, timedelta

def create_user_item_matrix(ratings_data):
    """
    Creates a user-item matrix from ratings data.
    Handles NaN values by filling them with 0.
    """
    try:
        user_item_matrix = ratings_data.pivot_table(index='user_id', columns='movie_id', values='rating', aggfunc='mean')
        user_item_matrix.fillna(0, inplace=True)
        return user_item_matrix
    except Exception as e:
        print(f"Error in creating user-item matrix: {e}")
        return None

def create_user_similarity_matrix(user_item_matrix):
    """
    Creates a user similarity matrix based on cosine similarity of ratings.
    """
    try:
        user_similarity = 1 - pairwise_distances(user_item_matrix, metric='cosine')
        user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
        return user_similarity_df
    except Exception as e:
        print(f"Error in creating user similarity matrix: {e}")
        return None

def dynamic_recommendation(user_id, user_similarity_df, user_item_matrix, ratings_data, movies_data, use_svd, use_time_based, user_data, n_recommendations, min_ratings=5):
    """
    Generates dynamic recommendations for a given user by selecting the appropriate recommendation method.
    """
    # Get content-based recommendations
    content_recs = content_based_recommendation(user_id, user_data, movies_data, ratings_data, n_recommendations)

    # If content-based recommendations are empty, use an alternative method
    if not content_recs:
        print(f"No content-based recommendations available. Using alternative method.")
        return svd_based_recommendation(user_id, ratings_data, movies_data, n_recommendations)

    # Check how many ratings the user has given
    user_ratings_count = ratings_data[ratings_data['user_id'] == user_id].shape[0]
    
    # If the user has rated fewer than min_ratings movies, use popularity-based recommendations
    if user_ratings_count < min_ratings:
        return popularity_based_recommendation(user_id, ratings_data, n_recommendations)
    
    # Re-run content-based recommendations in case the first attempt had too few results
    content_recs = content_based_recommendation(user_id, user_data, movies_data, ratings_data, n_recommendations)
    
    # If content-based recommendations are still insufficient, use SVD-based recommendations
    if len(content_recs) < n_recommendations:
        return svd_based_recommendation(user_id, ratings_data, movies_data, n_recommendations)
    
    return content_recs

def calculate_genre_popularity(weighted_ratings_df):
    """
    Calculates the genre popularity based on the weighted ratings.
    """
    return weighted_ratings_df.mean(axis=0)

def calculate_weighted_ratings(user_genre_interactions):
    """
    Calculates the weighted average ratings of each genre for each user.
    """
    weighted_ratings = {}
    
    for genre in user_genre_interactions.columns[::2]:  # For each genre (ignoring 'rating_count' columns)
        total_rating_col = f"{genre}_total_rating"
        rating_count_col = f"{genre}_rating_count"
        weighted_ratings[genre] = user_genre_interactions[total_rating_col] / user_genre_interactions[rating_count_col]
    
    return pd.DataFrame(weighted_ratings)

def content_based_recommendation(user_id, user_data, movies_data, ratings_data, n_recommendations=10):
    """
    Generates content-based recommendations for a given user based on movie genres and user preferences.
    """
    if user_data is None:
        print("Error: user_data is None.")
        return []  # Return an empty list

    # Retrieve user information
    user = user_data[user_data['user_id'] == user_id]
    if user.empty:
        print(f"User ID {user_id} not found.")
        return []  # Return an empty list if user not found

    # Check if the 'age' column exists
    if 'age' not in user_data.columns:
        print("Error: 'age' column not found in user_data.")
        return []  # Return an empty list if missing

    # Determine user's age group
    age = user['age'].values[0]
    if age <= 18:
        age_group = '0-18'
    elif age <= 25:
        age_group = '19-25'
    elif age <= 35:
        age_group = '26-35'
    elif age <= 50:
        age_group = '36-50'
    else:
        age_group = '50+'

    print(f"User's Age Group: {age_group}")

    # Define genre preferences based on age group
    age_group_preferences = {
        '0-18': ['Comedy', 'Action', 'Family'],
        '19-25': ['Action', 'Comedy', 'Sci-Fi', 'Thriller'],
        '26-35': ['Drama', 'Action', 'Comedy'],
        '36-50': ['Drama', 'Action', 'Comedy'],
        '50+': ['Drama', 'Comedy', 'Action', 'Documentary']
    }

    # Adjust preferences based on user's gender
    user_gender = user['gender'].values[0]
    preferred_genres = age_group_preferences.get(age_group, [])
    if user_gender == 'F':
        preferred_genres += ['Drama', 'Romance']
    elif user_gender == 'M':
        preferred_genres += ['Action', 'Adventure']

    # Merge ratings data with movies data
    merged_data = pd.merge(ratings_data, movies_data, left_on='movie_id', right_on='movie_id', how='left')

    # Sort movies by rating
    recommended_movies = merged_data.sort_values(by='rating', ascending=False)
    return recommended_movies['movie_id'].head(n_recommendations).tolist()

def user_based_recommendations(user_id, user_similarity_df, user_item_matrix, top_n=5):
    """
    Generates user-based collaborative filtering recommendations with predicted ratings.
    """
    similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False)
    user_watched = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    
    recommendations = pd.Series(dtype=float)
    for sim_user, sim_score in similar_users.iloc[1:].items():
        if sim_user in user_item_matrix.index:  # Check if sim_user exists in user_item_matrix
            weighted_ratings = user_item_matrix.loc[sim_user] * sim_score
            recommendations = recommendations.add(weighted_ratings, fill_value=0)
    
    recommendations = recommendations.drop(index=user_watched, errors='ignore').sort_values(ascending=False)
    recommendations = recommendations.head(top_n)
    
    # Predicted ratings and actual ratings
    predicted_ratings = recommendations
    actual_ratings = user_item_matrix.loc[user_id, recommendations.index]
    
    result = pd.DataFrame({'movie_id': recommendations.index, 'predicted_rating': predicted_ratings, 'actual_rating': actual_ratings})
    return result

def genre_based_recommendations(user_id, user_item_matrix, movie_features, movies_data, user_genre_interactions, top_n=5):
    """
    Generates genre-based recommendations by considering user preferences for genres, 
    and also considering age group and gender-based preferences.
    """
    # Get user's rated movies
    user_ratings = user_item_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0].index

    # Get user's genre data
    user_genre_data = user_genre_interactions.loc[user_id]

    # Get top genres based on user's preferences
    top_genres = user_genre_data.sort_values(ascending=False).head(top_n).index

    # Filter movies that match these genres
    genre_based_movies = movies_data[movies_data['genres'].apply(lambda genres: any(genre in genres for genre in top_genres))]

    # Get the ratings for the genre-based movies
    genre_based_movies = genre_based_movies[genre_based_movies['movie_id'].isin(rated_movies)]

    return genre_based_movies

def main_recommendation(user_id, ratings_data, user_data, movies_data, threshold=3, top_n=5, use_svd=False, use_time_based=False):
    """
    Main function to select a recommendation strategy and provide movie recommendations.
    """
    user_item_matrix = create_user_item_matrix(ratings_data)
    if user_item_matrix is None:
        return None

    user_similarity_df = create_user_similarity_matrix(user_item_matrix)
    if user_similarity_df is None:
        return None

    return dynamic_recommendation(user_id, user_similarity_df, user_item_matrix, ratings_data, movies_data, use_svd=use_svd, use_time_based=use_time_based, user_data=user_data, n_recommendations=top_n)
def svd_based_recommendation(user_id, ratings_data, movies_data, n_recommendations=10):
    """
    Generates SVD-based recommendations for a given user.
    """
    user_movie_matrix = ratings_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    svd = TruncatedSVD(n_components=50)
    matrix_svd = svd.fit_transform(user_movie_matrix)
    
    user_index = user_movie_matrix.index.get_loc(user_id)
    user_vector = matrix_svd[user_index]
    
    similarity = cosine_similarity(user_vector.reshape(1, -1), matrix_svd)
    similar_users = similarity.argsort()[0][-n_recommendations-1:-1]
    
    recommendations = user_movie_matrix.iloc[similar_users].mean(axis=0).sort_values(ascending=False).index.tolist()
    return recommendations[:n_recommendations]

def popularity_based_recommendation(user_id, ratings_data, n_recommendations=10):
    """
    Recommends the most popular movies based on the popularity of movies in the dataset.
    """
    movie_popularity = ratings_data.groupby('movie_id').size().sort_values(ascending=False)
    return movie_popularity.head(n_recommendations).index.tolist()

def main_recommendation(user_id, ratings_data, user_data, movies_data, threshold=3, top_n=5, use_svd=False, use_time_based=False):
    """
    Main function to select a recommendation strategy and provide movie recommendations.
    """
    user_item_matrix = create_user_item_matrix(ratings_data)
    if user_item_matrix is None:
        return None

    user_similarity_df = create_user_similarity_matrix(user_item_matrix)
    if user_similarity_df is None:
        return None

    return dynamic_recommendation(user_id, user_similarity_df, user_item_matrix, ratings_data, movies_data, use_svd=use_svd, use_time_based=use_time_based, user_data=user_data, n_recommendations=top_n)


class RecommendationSystem:
    def __init__(self, ratings_data, movies_data, user_data=None, use_svd=False, use_time_based=False):
        """ Initializes the RecommendationSystem class. """
        self.ratings_data = ratings_data
        self.movies_data = movies_data
        self.user_data = user_data  
        self.use_svd = use_svd
        self.use_time_based = use_time_based
        self.user_item_matrix = create_user_item_matrix(ratings_data)
        self.user_similarity_df = create_user_similarity_matrix(self.user_item_matrix)
        self.genre_popularity_df = None
    
    def generate_recommendations(self, user_id, threshold=3, top_n=5, use_svd=False, use_time_based=False):
        """ Recommends top_n movies to the user. """
        if user_id not in self.ratings_data['user_id'].unique():
            print(f"User ID {user_id} is invalid.")
            return None
        
        recommendations = main_recommendation(
            user_id, 
            self.ratings_data, 
            self.user_data,
            self.movies_data, 
            threshold=threshold, 
            top_n=top_n, 
            use_svd=self.use_svd, 
            use_time_based=self.use_time_based
        )
        
        if recommendations is not None:
            return recommendations
        else:
            print("An error occurred or no data was found for recommendations.")
            return None
