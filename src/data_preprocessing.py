import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# General data loading and saving functions
def load_data(file_path, columns=None, sep="\t", encoding=None):
    """Loads data from a file."""
    return pd.read_csv(file_path, sep=sep, names=columns, encoding=encoding)

def save_dataframe_to_csv(dataframe, file_name, directory_path):
    """Saves a DataFrame to a CSV file in the specified directory."""
    directory_path = os.path.abspath(directory_path)
    os.makedirs(directory_path, exist_ok=True)
    file_path = os.path.join(directory_path, file_name)
    try:
        dataframe.to_csv(file_path, index=False)
        print(f"File saved to {file_path}")
    except Exception as e:
        raise OSError(f"Failed to save file: {file_path}. Error: {e}")

# General display function
def display_dataset_info(data, dataset_name, head=True):
    """Displays dataset info, statistics, and optionally the first 5 rows."""
    print(f"{dataset_name} Dataset Info:")
    print(data.info())
    print(data.describe(include='all'), "\n")
    if head:
        print(f"{dataset_name} First 5 rows:")
        print(data.head(), "\n")

# User statistics
def calculate_user_stats(ratings_data):
    """Calculates average and standard deviation of ratings for each user."""
    user_stats = ratings_data.groupby("user_id")["rating"].agg(["mean", "std"]).reset_index()
    user_stats.columns = ["user_id", "avg_rating", "std_rating"]
    return user_stats

def identify_inconsistent_users(user_stats, threshold=1.5):
    """Identifies inconsistent users based on rating standard deviation."""
    inconsistent_users = user_stats[user_stats["std_rating"] > threshold]
    consistent_users = user_stats[user_stats["std_rating"] <= threshold]
    return inconsistent_users, consistent_users

def calculate_user_weights(ratings_data):
    """Calculates weights for users based on rating standard deviation."""
    user_std = ratings_data.groupby('user_id')['rating'].std()
    user_weights = 1 / (1 + user_std)
    return user_weights

def merge_user_weights(ratings_data, user_weights):
    """Merges calculated user weights into the ratings data."""
    user_weights = user_weights.reset_index()
    user_weights.columns = ['user_id', 'weight']
    if 'weight' in ratings_data.columns:
        ratings_data = ratings_data.drop(columns=['weight'])
    ratings_data = ratings_data.merge(user_weights, on='user_id', how='left')
    return ratings_data

def calculate_user_weighted_ratings(ratings_data, users_data, movies_data, genre_columns):
    """Calculates weighted ratings based on user demographics."""
    # Merge ratings and users data
    merged_data = ratings_data.merge(users_data, on="user_id", how="left")
    
    # Step 1: Calculate gender weight
    avg_rating_by_gender = merged_data.groupby('gender')['rating'].mean()
    gender_weight = {
        'M': 1 / (1 + abs(avg_rating_by_gender.get('M', 0) - avg_rating_by_gender.get('F', 0))),
        'F': 1 / (1 + abs(avg_rating_by_gender.get('F', 0) - avg_rating_by_gender.get('M', 0)))
    }
    # Apply gender weight to users data
    users_data['gender_weight'] = users_data['gender'].map(gender_weight)
    
    # Step 2: Calculate age weight
    avg_rating_by_age = merged_data.groupby('age_group')['rating'].mean()
    age_weights = 1 / (1 + abs(avg_rating_by_age - avg_rating_by_age.mean()))
    users_data = users_data.merge(age_weights.rename('age_weight'), left_on='age_group', right_index=True, how='left')
    
    # Step 3: Merge age_weight and gender_weight with merged_data
    merged_data = merged_data.merge(users_data[['user_id', 'age_weight', 'gender_weight']], on="user_id", how="left")
    
    # Step 4: Calculate weighted rating
    merged_data['weighted_rating'] = (
        merged_data['rating'] * merged_data['age_weight'] * merged_data['gender_weight']
    )
    
    return merged_data

# Genre-specific statistics
def get_genre_columns(movies_data):
    """Returns the list of genre columns in the movies_data DataFrame."""
    genre_columns = [col for col in movies_data.columns if movies_data[col].nunique() == 2 and set(movies_data[col].unique()) == {0, 1}]
    return genre_columns

def calculate_genre_avg_ratings(movies_data, weighted_ratings, genre_columns):
    """Calculates the average rating per genre."""
    # Merge ratings with movies to ensure genre columns are available
    merged_data = weighted_ratings.merge(movies_data[['movie_id'] + genre_columns], left_on='item_id', right_on='movie_id', how='left')
    
    genre_ratings = {}
    for genre in genre_columns:
        genre_ratings[genre] = merged_data[merged_data[genre] == 1]['rating'].mean()
    
    genre_avg_ratings = pd.DataFrame.from_dict(genre_ratings, orient='index', columns=['average_rating'])
    genre_avg_ratings = genre_avg_ratings.sort_values(by='average_rating', ascending=False)
    return genre_avg_ratings


def calculate_user_genre_interactions(merged_data, movies):
    """Calculates user-genre interactions based on ratings and genres."""
    genre_columns = [col for col in movies.columns if movies[col].nunique() == 2 and set(movies[col].unique()) == {0, 1}]
    movies_with_ratings = movies.merge(
        merged_data[['item_id', 'user_id', 'rating']],
        left_on='movie_id', 
        right_on='item_id', 
        how='left'
    )
    user_genre_interactions = pd.DataFrame(index=merged_data['user_id'].unique())
    for genre in genre_columns:
        genre_movies = movies_with_ratings[movies_with_ratings[genre] == 1]
        total_ratings = genre_movies.groupby('user_id')['rating'].sum()
        rating_counts = genre_movies.groupby('user_id')['rating'].count()
        user_genre_interactions[genre + '_total_rating'] = total_ratings
        user_genre_interactions[genre + '_rating_count'] = rating_counts
    user_genre_interactions.fillna(0, inplace=True)
    return user_genre_interactions
