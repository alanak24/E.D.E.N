import pandas as pd
from sklearn.neighbors import NearestNeighbors

#mock data
data = [
    [1,101,5], [1,102,4], [1,103,2], [1,104,3],
    [2,101,5], [2,102,5], [2,104,4], [2,105,3],
    [3,102,3], [3,103,5], [3,104,4], [3,106,4],
    [4,101,2], [4,103,4], [4,105,5], [4,106,3],
    [5,102,5], [5,103,4], [5,104,5], [5,107,4],
    [6,101,4], [6,105,3], [6,106,5], [6,107,4],
    [7,102,2], [7,103,3], [7,104,2], [7,105,4],
    [8,101,5], [8,102,4], [8,106,4], [8,107,5],
    [9,103,5], [9,104,4], [9,105,5], [9,107,3],
    [10,101,3], [10,102,4], [10,103,2], [10,107,5]
]

df = pd.DataFrame(data, columns=['user_id', 'movie_id', 'rating'])

movies_data = [
    [101, "Toy Story"],
    [102, "Avengers"],
    [103, "Titanic"],
    [104, "Inception"],
    [105, "The Dark Knight"],
    [106, "Interstellar"],
    [107, "Joker"]
]

movies = pd.DataFrame(movies_data, columns=['movie_id', 'title'])

# matrix --------------------------------------------------------
user_movie_matrix = df.pivot_table(
    index='user_id',
    columns='movie_id',
    values='rating'
).fillna(0)

print(user_movie_matrix)
#-----------------------------------------------------------------------------------
# KNN model

model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(user_movie_matrix)

# Find user method
def find_similar_users(user_id, k=3):
    if user_id not in user_movie_matrix.index:
        print("User not found")
        return []
    
    user_vector = user_movie_matrix.loc[user_id].values.reshape(1, -1)
    
    distances, indices = model.kneighbors(user_vector, n_neighbors=k+3)
    
    neighbors = []
    
    for i in range(1, len(indices[0])):  # skip self
        neighbor_id = user_movie_matrix.index[indices[0][i]]
        similarity = 1 - distances[0][i]
        neighbors.append((neighbor_id, similarity))
        
    return neighbors

# ----------------------------------------------------------------------------------------

# -----------------------------
def recommend_movies(user_id, k=5, n_recommendations=5):
    neighbors = find_similar_users(user_id, k)
    
    if len(neighbors) == 0:
        return []
    
    watched_movies = set(
        user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
    )
    
    movie_scores = {}
    
    for neighbor_id, similarity in neighbors:
        neighbor_ratings = user_movie_matrix.loc[neighbor_id]
        
        for movie_id, rating in neighbor_ratings.items():
            if movie_id not in watched_movies and rating > 0:
                if movie_id not in movie_scores:
                    movie_scores[movie_id] = 0
                movie_scores[movie_id] += rating * similarity
    
    ranked_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_movies[:n_recommendations]

# -----------------------------
# 6. TEST
# -----------------------------
target_user = 1

print("\nSimilar users:")
print(find_similar_users(target_user, k=3))


print("\nRecommended movies:")
recommendations = recommend_movies(target_user, k=3, n_recommendations=3)

for movie_id, score in recommendations:
    title_row = movies[movies['movie_id'] == movie_id]
    
    if not title_row.empty:
        title = title_row['title'].values[0]
    else:
        title = f"Movie {movie_id}"
    
    print(f"{title} (movie_id={movie_id}) -> score: {score:.2f}")