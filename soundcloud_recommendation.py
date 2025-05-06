import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

# Set page configuration
st.set_page_config(
    page_title="SoundCloud Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff5500;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .track-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ff5500;
    }
    .track-title {
        font-weight: bold;
        font-size: 1.2rem;
    }
    .track-artist {
        color: #666;
        font-size: 1rem;
    }
    .track-genre {
        color: #ff5500;
        font-size: 0.9rem;
    }
    .track-stats {
        color: #999;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Create synthetic data for demo
@st.cache_data
def generate_sample_data():
    # Create sample users
    users = list(range(1, 101))
    
    # Create sample tracks with metadata
    genres = ["Hip Hop", "Rock", "Electronic", "Pop", "R&B", "Jazz", "Classical", "Indie", "Trap", "House"]
    moods = ["Energetic", "Calm", "Melancholic", "Upbeat", "Relaxed", "Intense", "Dreamy", "Dark", "Happy"]
    
    track_data = []
    for i in range(1, 201):
        track = {
            'track_id': i,
            'title': f"Track {i}",
            'artist': f"Artist {random.randint(1, 50)}",
            'genre': random.choice(genres),
            'subgenre': random.choice(genres),
            'mood': random.choice(moods),
            'bpm': random.randint(60, 180),
            'release_year': random.randint(2010, 2023),
            'duration': random.randint(120, 480),
            'plays': random.randint(1000, 1000000),
            'likes': random.randint(100, 50000),
            'reposts': random.randint(10, 5000),
            'comments': random.randint(5, 1000),
            'lyrics_available': random.choice([True, False]),
            'tags': ", ".join(random.sample(genres + moods, k=random.randint(2, 5)))
        }
        track_data.append(track)
    
    tracks_df = pd.DataFrame(track_data)
    
    # Create user listening history
    user_track_interactions = []
    for user_id in users:
        # Each user listens to 10-30 tracks
        num_tracks = random.randint(10, 30)
        listened_tracks = random.sample(list(tracks_df['track_id']), num_tracks)
        
        for track_id in listened_tracks:
            interaction = {
                'user_id': user_id,
                'track_id': track_id,
                'listen_count': random.randint(1, 50),
                'liked': random.random() > 0.7,
                'reposted': random.random() > 0.9,
                'commented': random.random() > 0.95,
                'in_playlist': random.random() > 0.8,
                'skipped': random.random() > 0.8,
            }
            user_track_interactions.append(interaction)
    
    user_track_df = pd.DataFrame(user_track_interactions)
    
    # Create user-track matrix for collaborative filtering
    user_track_matrix = user_track_df.pivot_table(
        index='user_id', 
        columns='track_id', 
        values='listen_count',
        fill_value=0
    )
    
    return tracks_df, user_track_df, user_track_matrix

def create_user_profile(user_id, user_track_df, tracks_df):
    """Create a user profile based on their listening history"""
    user_data = user_track_df[user_track_df['user_id'] == user_id]
    
    if user_data.empty:
        return None
    
    # Merge with track information
    user_profile = user_data.merge(tracks_df, on='track_id')
    
    # Aggregate genre and mood preferences
    genre_counts = {}
    mood_counts = {}
    
    for _, row in user_profile.iterrows():
        weight = row['listen_count'] * (3 if row['liked'] else 1)
        
        genre = row['genre']
        genre_counts[genre] = genre_counts.get(genre, 0) + weight
        
        mood = row['mood']
        mood_counts[mood] = mood_counts.get(mood, 0) + weight
    
    # Normalize counts
    total_genre = sum(genre_counts.values())
    total_mood = sum(mood_counts.values())
    
    genre_prefs = {k: v/total_genre for k, v in genre_counts.items()}
    mood_prefs = {k: v/total_mood for k, v in mood_counts.items()}
    
    return {
        'user_id': user_id,
        'listened_tracks': set(user_profile['track_id']),
        'genre_preferences': genre_prefs,
        'mood_preferences': mood_prefs,
        'favorite_artists': user_profile['artist'].value_counts().to_dict(),
        'bpm_range': (user_profile['bpm'].min(), user_profile['bpm'].max()),
        'avg_duration': user_profile['duration'].mean()
    }

def get_user_item_matrix(user_track_df):
    """Create a utility matrix for collaborative filtering"""
    # We'll use listen_count as the primary signal, but adjust based on explicit actions
    
    # Create a copy to avoid modifying the original
    df = user_track_df.copy()
    
    # Adjust listen_count based on explicit actions
    df['adjusted_score'] = df['listen_count']
    df.loc[df['liked'], 'adjusted_score'] *= 1.5
    df.loc[df['reposted'], 'adjusted_score'] *= 2.0
    df.loc[df['commented'], 'adjusted_score'] *= 1.3
    df.loc[df['in_playlist'], 'adjusted_score'] *= 1.8
    df.loc[df['skipped'], 'adjusted_score'] *= 0.5
    
    # Create matrix with adjusted scores
    user_item_matrix = df.pivot_table(
        index='user_id', 
        columns='track_id', 
        values='adjusted_score',
        fill_value=0
    )
    
    return user_item_matrix

def collaborative_filtering(user_id, user_item_matrix, n_recommendations=10):
    """User-based collaborative filtering"""
    # Calculate user similarity
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, 
                              index=user_item_matrix.index, 
                              columns=user_item_matrix.index)
    
    # Get similar users
    user_idx = user_item_matrix.index.get_loc(user_id)
    similar_users = user_sim_df.iloc[user_idx].sort_values(ascending=False)[1:11]  # Top 10 similar users
    
    # Get tracks listened by similar users but not by the target user
    user_tracks = set(user_item_matrix.columns[user_item_matrix.loc[user_id] > 0])
    
    recommendations = {}
    for sim_user_id, sim_score in similar_users.items():
        sim_user_tracks = set(user_item_matrix.columns[user_item_matrix.loc[sim_user_id] > 0])
        new_tracks = sim_user_tracks - user_tracks
        
        for track in new_tracks:
            if track not in recommendations:
                recommendations[track] = 0
            recommendations[track] += sim_score * user_item_matrix.loc[sim_user_id, track]
    
    # Sort recommendations by score
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_recommendations = [track for track, score in sorted_recs[:n_recommendations]]
    
    return top_recommendations

def content_based_filtering(user_profile, tracks_df, n_recommendations=10):
    """Content-based filtering based on track metadata"""
    # If user has no profile, return popular tracks
    if user_profile is None:
        return tracks_df.sort_values('plays', ascending=False)['track_id'][:n_recommendations].tolist()
    
    # Get tracks the user hasn't listened to
    unlistened_tracks = tracks_df[~tracks_df['track_id'].isin(user_profile['listened_tracks'])]
    
    # Calculate genre and mood similarity scores
    track_scores = []
    
    for _, track in unlistened_tracks.iterrows():
        # Genre similarity
        genre_score = user_profile['genre_preferences'].get(track['genre'], 0)
        
        # Mood similarity
        mood_score = user_profile['mood_preferences'].get(track['mood'], 0)
        
        # Artist similarity (if user has listened to this artist before)
        artist_score = user_profile['favorite_artists'].get(track['artist'], 0) / max(user_profile['favorite_artists'].values()) if user_profile['favorite_artists'] else 0
        
        # BPM compatibility - prefer tracks within user's BPM range
        bpm_min, bpm_max = user_profile['bpm_range']
        bpm_score = 1.0 if bpm_min <= track['bpm'] <= bpm_max else 0.5
        
        # Duration compatibility - prefer tracks close to user's average duration
        duration_diff = abs(track['duration'] - user_profile['avg_duration']) / 240  # Normalize
        duration_score = max(0, 1 - duration_diff)
        
        # Combine scores (weighted)
        total_score = (0.4 * genre_score + 
                       0.2 * mood_score + 
                       0.2 * artist_score + 
                       0.1 * bpm_score + 
                       0.1 * duration_score)
        
        track_scores.append((track['track_id'], total_score))
    
    # Sort by score
    sorted_tracks = sorted(track_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N
    top_tracks = [track_id for track_id, _ in sorted_tracks[:n_recommendations]]
    
    return top_tracks

def hybrid_recommendations(user_id, user_profile, user_item_matrix, tracks_df, 
                          collab_weight=0.7, content_weight=0.3, n_recommendations=10):
    """Hybrid recommendation combining collaborative and content-based filtering"""
    # Get collaborative filtering recommendations
    cf_recs = collaborative_filtering(user_id, user_item_matrix, n_recommendations=20)
    
    # Get content-based recommendations
    cb_recs = content_based_filtering(user_profile, tracks_df, n_recommendations=20)
    
    # Combine recommendations with weights
    combined_scores = {}
    
    for i, track_id in enumerate(cf_recs):
        if track_id not in combined_scores:
            combined_scores[track_id] = 0
        combined_scores[track_id] += collab_weight * (1 - i/len(cf_recs))
    
    for i, track_id in enumerate(cb_recs):
        if track_id not in combined_scores:
            combined_scores[track_id] = 0
        combined_scores[track_id] += content_weight * (1 - i/len(cb_recs))
    
    # Sort by score
    sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_recommendations = [track for track, _ in sorted_recs[:n_recommendations]]
    
    return top_recommendations

def display_track_card(track):
    """Display a track in a nice card format"""
    st.markdown(f"""
    <div class="track-card">
        <div class="track-title">{track['title']}</div>
        <div class="track-artist">{track['artist']}</div>
        <div class="track-genre">{track['genre']} • {track['mood']} • {track['bpm']} BPM</div>
        <div class="track-stats">♬ {format(track['plays'], ',')} plays • ♥ {format(track['likes'], ',')} likes</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Title and description
    st.markdown('<h1 class="main-header">SoundCloud Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover new music you\'ll love</p>', unsafe_allow_html=True)
    
    # Load data
    tracks_df, user_track_df, user_track_matrix = generate_sample_data()
    
    # Create sidebar for user selection and options
    with st.sidebar:
        st.header("Settings")
        
        # User selection
        user_id = st.selectbox("Select User ID", sorted(user_track_df['user_id'].unique()))
        
        # Recommendation method
        rec_method = st.radio(
            "Recommendation Method",
            ["Collaborative Filtering", "Content-Based", "Hybrid (Recommended)"]
        )
        
        # Number of recommendations
        n_recommendations = st.slider("Number of Recommendations", 5, 20, 10)
        
        if rec_method == "Hybrid (Recommended)":
            collab_weight = st.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.7, 0.1)
            content_weight = st.slider("Content-Based Weight", 0.0, 1.0, 0.3, 0.1)
        else:
            collab_weight = 0.7
            content_weight = 0.3
        
        st.button("Get Recommendations", type="primary")
    
    # Create user profile
    user_profile = create_user_profile(user_id, user_track_df, tracks_df)
    user_item_matrix = get_user_item_matrix(user_track_df)
    
    # Show user profile
    st.header(f"User Profile (ID: {user_id})")
    
    if user_profile:
        # Display user stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tracks Listened", len(user_profile['listened_tracks']))
        
        with col2:
            top_genre = max(user_profile['genre_preferences'].items(), key=lambda x: x[1])[0]
            st.metric("Top Genre", top_genre)
        
        with col3:
            top_mood = max(user_profile['mood_preferences'].items(), key=lambda x: x[1])[0]
            st.metric("Favorite Mood", top_mood)
        
        # Display user preferences
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Genre Preferences")
            genre_df = pd.DataFrame({
                'Genre': user_profile['genre_preferences'].keys(),
                'Preference': user_profile['genre_preferences'].values()
            }).sort_values('Preference', ascending=False).head(5)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Preference', y='Genre', data=genre_df, palette='viridis', ax=ax)
            ax.set_xlim(0, 1)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Mood Preferences")
            mood_df = pd.DataFrame({
                'Mood': user_profile['mood_preferences'].keys(),
                'Preference': user_profile['mood_preferences'].values()
            }).sort_values('Preference', ascending=False).head(5)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Preference', y='Mood', data=mood_df, palette='magma', ax=ax)
            ax.set_xlim(0, 1)
            st.pyplot(fig)
    else:
        st.warning("No user profile available")
    
    # Generate recommendations
    st.header("Recommended Tracks")
    
    with st.spinner("Generating recommendations..."):
        time.sleep(1)  # Simulating processing time
        
        if rec_method == "Collaborative Filtering":
            rec_track_ids = collaborative_filtering(user_id, user_item_matrix, n_recommendations)
            rec_description = "Based on similar users' listening habits"
        elif rec_method == "Content-Based":
            rec_track_ids = content_based_filtering(user_profile, tracks_df, n_recommendations)
            rec_description = "Based on your genre and mood preferences"
        else:  # Hybrid
            rec_track_ids = hybrid_recommendations(
                user_id, user_profile, user_item_matrix, tracks_df, 
                collab_weight, content_weight, n_recommendations
            )
            rec_description = "Combining similar users' habits and your preferences"
    
    st.caption(f"**{rec_description}**")
    
    # Display recommendations
    recommended_tracks = tracks_df[tracks_df['track_id'].isin(rec_track_ids)]
    
    for _, track in recommended_tracks.iterrows():
        display_track_card(track)
    
    # Show recommendation explanation
    with st.expander("How were these recommendations generated?"):
        st.write("""
        This recommendation system uses multiple approaches to suggest tracks:
        
        **Collaborative Filtering**: This method finds users with similar listening habits to you, 
        then recommends tracks they've enjoyed that you haven't heard yet.
        
        **Content-Based Filtering**: This approach analyzes your genre preferences, favorite artists, 
        and musical attributes (BPM, mood, etc.) to find similar tracks.
        
        **Hybrid Approach**: The most effective method, combining both collaborative and content-based 
        recommendations for more personalized results.
        """)
        
        if rec_method == "Hybrid (Recommended)":
            st.write(f"Current weights: {collab_weight:.1f} for collaborative filtering and {content_weight:.1f} for content-based.")

if __name__ == "__main__":
    main()
