import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_visualization_data():
    """Load the datasets for visualization."""
    users_data = pd.read_csv("../data//extended_users.csv")
    movies_data = pd.read_csv("../data/processed/extended_movies.csv")
    ratings_data = pd.read_csv("../data/processed/extended_ratings.csv")
    return users_data, movies_data, ratings_data

def plot_age_distribution(users_data):
    """Plot the age distribution of users."""
    plt.figure(figsize=(10, 6))
    sns.histplot(users_data['age'], bins=20, kde=True, color='blue')
    plt.title("User Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of Users")
    save_plot(plt, "user_age_distribution.png")

def plot_gender_genre_popularity(users_data, movies_data, genre_columns):
    """Plot the gender-based genre popularity."""
    genre_cols = [col for col in movies_data.columns if col in genre_columns]
    gender_genre_popularity = users_data.merge(movies_data, how='left', left_on='user_id', right_on='movie_id')
    gender_genre_popularity = gender_genre_popularity.groupby('gender')[genre_cols].sum()

    gender_genre_popularity.T.plot(kind='bar', figsize=(14, 8), stacked=True, color=['blue', 'pink'])
    plt.title("Gender-Based Genre Popularity")
    plt.ylabel("Total Genre Scores")
    plt.xlabel("Genres")
    save_plot(plt, "gender_genre_popularity.png")

def plot_genre_popularity(movies_data, genre_columns):
    """Plot the popularity of movie genres."""
    genre_popularity = movies_data[genre_columns].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=genre_popularity.index, y=genre_popularity.values, palette="viridis")
    plt.title("Popularity of Movie Genres")
    plt.xticks(rotation=45)
    plt.ylabel("Number of Movies")
    save_plot(plt, "movie_genre_popularity.png")

def plot_movie_avg_ratings(ratings_data):
    """Plot the distribution of average movie ratings."""
    if 'average_rating' in ratings_data.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(ratings_data['movie_avg_rating'], bins=30, kde=True, color='green')
        plt.title("Distribution of Average Movie Ratings")
        plt.xlabel("Average Rating")
        plt.ylabel("Number of Movies")
        save_plot(plt, "average_movie_ratings.png")
    else:
        print("Error: 'average_rating' column not found!")

def plot_user_rating_behavior(ratings_data):
    """Plot the rating behavior of users."""
    user_rating_counts = ratings_data.groupby('user_id')['rating'].count()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_rating_counts, bins=30, color='orange', kde=True)
    plt.title("User Rating Behavior")
    plt.xlabel("Number of Rated Movies")
    plt.ylabel("Number of Users")
    save_plot(plt, "user_rating_behavior.png")

def plot_most_popular_movies(ratings_data):
    """Plot the most popular movies based on rating count."""
    movie_popularity = ratings_data.groupby('item_id')['rating'].count().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=movie_popularity.values, y=movie_popularity.index, palette="coolwarm")
    plt.title("Most Popular Movies")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Movie ID")
    save_plot(plt, "most_popular_movies.png")

def plot_rating_density_by_year(ratings_data):
    """Plot the rating density over years."""
    plt.figure(figsize=(12, 6))
    sns.countplot(data=ratings_data, x='rating_year', color='purple')
    plt.title("Rating Density by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Ratings")
    save_plot(plt, "rating_density_by_year.png")

def plot_yearly_genre_popularity(ratings_data, movies_data, genre_columns):
    """Plot the popularity of genres over the years."""
    ratings_with_genres = ratings_data.merge(movies_data, how='left', left_on='item_id', right_on='movie_id')
    year_genre_popularity = ratings_with_genres.groupby(['rating_year'])[genre_columns].sum()

    plt.figure(figsize=(14, 8))
    sns.heatmap(year_genre_popularity.T, cmap="YlGnBu", annot=False, cbar=True)
    plt.title("Genre Popularity by Year")
    plt.xlabel("Year")
    plt.ylabel("Genres")
    plt.xticks(rotation=45)
    save_plot(plt, "yearly_genre_popularity.png")

def plot_correlation_heatmap(ratings_data):
    """Plot the correlation heatmap for numerical features."""
    numeric_columns = ratings_data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True)
    plt.title("Correlation Matrix Between Features")
    save_plot(plt, "correlation_matrix.png")

def plot_age_group_genre_popularity(users_data, movies_data, genre_columns):
    """Plot genre popularity by age group."""
    users_data['age_group'] = pd.cut(users_data['age'], bins=[0, 18, 25, 35, 50, 100], labels=['0-18', '19-25', '26-35', '36-50', '50+'])
    age_genre_popularity = users_data.merge(movies_data, how='left', left_on='user_id', right_on='movie_id')
    age_genre_popularity = age_genre_popularity.groupby('age_group')[genre_columns].sum()

    plt.figure(figsize=(14, 8))
    sns.heatmap(age_genre_popularity.T, cmap="YlGnBu", annot=True, fmt=".0f")
    plt.title("Genre Popularity by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Genres")
    save_plot(plt, "age_group_genre_popularity.png")

def plot_gender_age_rating(ratings_data, users_data):
    """Plot average ratings by gender and age group."""
    rating_behavior = ratings_data.merge(users_data, on='user_id')
    gender_age_ratings = rating_behavior.groupby(['gender', 'age_group'])['rating'].mean().unstack()

    gender_age_ratings.plot(kind='bar', figsize=(12, 6), colormap='coolwarm')
    plt.title("Average Rating by Gender and Age Group")
    plt.ylabel("Average Rating")
    plt.xlabel("Gender")
    plt.legend(title="Age Group")
    save_plot(plt, "gender_age_rating.png")

def plot_genre_rating_correlation(movies_data, genre_avg_ratings, genre_columns):
    """Plot the correlation between movie genres and ratings."""
    genre_ratings = genre_avg_ratings[genre_columns].join(movies_data[['item_id']], how='left', on='item_id')
    correlation = genre_ratings.corr()['rating'].drop('rating').sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=correlation.index, y=correlation.values, palette='viridis')
    plt.title("Correlation Between Movie Genres and Ratings")
    plt.ylabel("Correlation")
    plt.xlabel("Genres")
    plt.xticks(rotation=45)
    save_plot(plt, "genre_rating_correlation.png")

def plot_user_rating_consistency(ratings_data):
    """Plot user rating consistency."""
    user_variance = ratings_data.groupby('user_id')['rating'].var()
    threshold = user_variance.mean()
    ratings_data['consistency'] = ratings_data['user_id'].map(lambda x: 'Consistent' if user_variance[x] < threshold else 'Inconsistent')

    sns.countplot(data=ratings_data, x='consistency', palette="pastel")
    plt.title("User Rating Consistency")
    plt.ylabel("Number of Users")
    plt.xlabel("Consistency Status")
    save_plot(plt, "user_rating_consistency.png")

# Plot saving function
def save_plot(plot, filename, output_dir="../reports/eda_reports"):
    """Save the generated plot to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, filename)
    plot.savefig(plot_path)
    plt.close()  # Close the plot to avoid memory issues