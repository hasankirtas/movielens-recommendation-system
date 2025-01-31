# MovieLens Recommendation System Project Overview

This project is my first hands-on experience with recommendation systems. Using the MovieLens dataset, I explored different recommendation techniques, ranging from content-based filtering to collaborative filtering (SVD and user-based).

The goal was to gain practical experience in building recommendation models, handling user-item interactions, and developing dynamic recommendation strategies. Throughout this project, I focused on:

- **Data preprocessing and feature engineering**
- **Understanding and implementing collaborative filtering techniques**
- **Experimenting with content-based recommendations**
- **Structuring a modular, reusable codebase for recommendation models**

Rather than focusing purely on model performance, my main objective was to understand the fundamentals of recommendation systems and their real-world applications. Since the project timeline was limited and my technical expertise in this area was still developing, I couldn't fully focus on optimizing the model's performance. However, optimizing and evaluating model performance is something I plan to explore further in the future, once I have a more solid understanding of recommendation techniques.

---

### Objectives
- Develop an end-to-end recommendation pipeline.
- Explore content-based and collaborative filtering approaches.
- Analyze user behavior and rating patterns.
- Build a dynamic recommendation function based on user interactions.
- Ensure the project is reproducible and well-documented.

### Steps Followed

#### 1. Data Preprocessing and Feature Engineering

During data preprocessing, we focused on cleaning the data and creating new, meaningful features to enhance the recommendation model. Visualizations were created to better understand user and movie interactions, as well as demographic patterns. These visualizations helped in uncovering key insights that guided the model-building process.

**Key Visualizations and Insights:**
- **Rating Distribution:** Analyzed how ratings are distributed across different age groups, genders, and movie genres. This helped to identify any biases or imbalances in the data.
- **Genre Preferences:** Visualized genre preferences based on age groups and genders. This gave insights into which genres are more popular among specific user groups.
- **User-Item Interactions:** Plotted interactions between users and movies, which helped identify sparsity in user ratings and areas for improvement.

#### 2. Feature Engineering and Derived Features

Based on the insights gathered from visualizations, several new features were engineered to improve the recommendation system's performance. These features include:

- **age_group_avg_ratings:** Average ratings given by users in different age groups. This allowed for personalized recommendations based on the user's age group.
- **genre_popularity:** Popularity of each genre based on the number of ratings received. This metric was used to incorporate genre bias into the recommendations.
- **user_genre_interactions:** Interactions between users and genres, capturing preferences based on the number of movies rated in each genre. This feature was crucial for improving content-based recommendations.

These new features were incorporated into the models to refine and personalize the recommendation results.

#### 3. Model Development

The recommendation system was built using a combination of content-based, collaborative filtering, and popularity-based methods. Below are the functions and approaches used to develop the model:

**Key Functions:**
- **create_user_item_matrix:** Constructs a user-item interaction matrix, which is central to the collaborative filtering methods. This matrix contains ratings for each user-item pair.
- **create_user_similarity_matrix:** Generates a user similarity matrix, used in user-based collaborative filtering to find similar users and recommend items they liked.
- **dynamic_recommendation:** This function selects the appropriate recommendation method dynamically, based on the user’s activity and interactions. If a user has rated many movies, SVD-based recommendations are prioritized; if the user has rated few movies, content-based or popularity-based methods are used.
- **calculate_genre_popularity:** Calculates the popularity of movie genres based on the average ratings and frequency of interactions, enhancing content-based recommendations by incorporating genre bias.
- **calculate_weighted_ratings:** Weighs ratings based on user interactions and genre popularity to generate more meaningful recommendations.
- **content_based_recommendation:** A content-based recommendation method that suggests movies to users based on the movie’s genre, popularity, and the user's past preferences.
- **svd_based_recommendation:** A collaborative filtering method using Singular Value Decomposition (SVD), which factorizes the user-item interaction matrix and predicts user ratings for unrated items.
- **popularity_based_recommendation:** Recommends movies based on their popularity across all users, which is especially useful for new users with little interaction data.
- **user_based_recommendations:** A collaborative filtering method based on finding users similar to the target user and recommending items liked by those similar users.
- **genre_based_recommendations:** Recommends movies based on the user’s genre preferences, ensuring the suggestions align with the user’s known likes.
- **main_recommendation:** The final recommendation function that uses the dynamic_recommendation function to select the best approach based on the user’s profile and activity.
- **RecommendationSystem:** A class that consolidates all recommendation functions into a single framework, making it easy to handle different types of recommendations for various user profiles.

#### 4. Hybrid Model and Decision Making

While a hybrid model combining different recommendation techniques seemed appealing at first, I decided against it. The MovieLens dataset had some inconsistencies between its sub-datasets, which made combining them difficult and prone to issues. Additionally, creating a hybrid model would have unnecessarily complicated the system without significantly improving results. 

Instead, I opted for a dynamic recommendation model, which intelligently switches between recommendation techniques based on the user’s activity. This provided a more flexible and reliable approach, avoiding the complexity of a hybrid model while delivering effective recommendations.

#### 5. Challenges and Solutions

- **Handling Sparse Data:** The MovieLens dataset is sparse with many missing values. I addressed this by filling missing ratings with default values and applying matrix factorization techniques.
- **Feature Engineering:** One of the challenges was selecting relevant features that could enhance the recommendation model. I solved this by experimenting with user interaction data, genre preferences, and age group-based filtering.
- **Model Evaluation:** Ensuring the models didn't overfit was challenging. I applied cross-validation techniques and regularization methods to address this.

### Tools and Libraries Used

- **Programming Language:** Python
- **Libraries:** pandas, numpy (Data manipulation), scikit-learn (Collaborative filtering, SVD), matplotlib, seaborn (Visualization), Surprise (Collaborative filtering models), SciPy (Matrix factorization)

### File Structure
```plaintext
Project/
├── data/
│   ├── raw/
│   ├── processed/
├── models/
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── model_training.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── utils.py
│   ├── model.py
├── reports/
│   ├── eda_reports/
├── README.md
├── environment.yml
├── requirements.txt
```

---

### Installation
To set up the project environment:

Using Conda:
```bash
conda env create -f environment.yml
conda activate your_project_name
```

Using Pip:
```bash
pip install -r requirements.txt
```

---

### Conclusion
This project helped me gain practical experience in building recommendation systems, including both content-based and collaborative filtering techniques. It allowed me to learn about data preprocessing, feature engineering, and model evaluation.

However, towards the end of the project, I realized that recommendation systems might not align with my long-term career interests. While I value the knowledge and skills gained, I have found that I am more interested in other areas of data science. This will guide my future projects and career development.

Although I couldn't focus fully on model optimization due to time constraints, I plan to explore this area further in the future.

---

### Acknowledgements
Thanks to the MovieLens dataset and Kaggle for providing such a valuable resource for learning and experimentation.

---



---
