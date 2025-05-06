import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Set page configuration
st.set_page_config(
    page_title="SoundCloud Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add simpler CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #ff5506;
    }
    .track-card {
        background-color: #f9f9f0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ff5500;
    }
</style>
""", unsafe_allow_html=True)

# Create synthetic data (reduced size for better performance)
@st.cache_data
def generate_sample_data():
    # Create sample users
    users = list(range(1, 51))  # Reduced to 50 users
    
    # Create sample tracks with metadata
    genres = ["Hip Hop", "Rock", "Electronic", "Pop", "R&B"]
    moods = ["Energetic", "Calm", "Upbeat", "Relaxed", "Intense"]
    
    track_data = []
    for i in range(1, 101):  # Reduced to 100 tracks
        track = {
            'track_id': i,
            'title': f"Track {i}",
            'artist': f"Artist {random.randint(1, 25)}",
            'genre': random.choice(genres),
            'mood': random.choice(moods),
            'bpm': random.randint(60, 180),
            'duration': random.randint(120, 480),
            'plays': random.randint(1000, 100000),
            'likes': random.randint(100, 5000),
        }
        track_data.append(track)
    
    tracks_df = pd.DataFrame(track_data)
    
    # Create user listening history
    user_track_interactions = []
    for user_id in users:
        # Each user listens to 5-15 tracks
        num_tracks = random.randint(5, 15)
        listened_tracks = random.sample(list(tracks_df['track_id']), num_tracks)
        
        for track_id in listened_tracks:
            interaction = {
                'user_id': user_id,
                'track_id': track_id,
                'listen_count': random.randint(1, 20),
                'liked': random.random() > 0.7,
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
    """Create a simplified user profile based on their listening history"""
    user_data = user_track_df[user_track_df['user_id'] == user_id]
    
    if user_data.empty:
        return None
    
    # Merge with track information
    user_profile = user_data.merge(tracks_df, on='track_id')
    
    # Aggregate genre and mood preferences
    genre_counts = user_profile['genre'].value_counts().to_dict()
    mood_counts = user_profile['mood'].value_counts().to_dict()
    
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
    }

def collaborative_filtering(user_id, user_track_matrix, n_recommendations=5):
    """Simplified user-based collaborative filtering"""
    # Calculate user similarity
    user_sim = cosine_similarity(user_track_matrix)
    user_sim_df = pd.DataFrame(user_sim, 
                              index=user_track_matrix.index, 
                              columns=user_track_matrix.index)
    
    # Get similar users
    user_idx = user_track_matrix.index.get_loc(user_id)
    similar_users = user_sim_df.iloc[user_idx].sort_values(ascending=False)[1:6]  # Top 5 similar users
    
    # Get tracks listened by similar users but not by the target user
    user_tracks = set(user_track_matrix.columns[user_track_matrix.loc[user_id] > 0])
    
    recommendations = {}
    for sim_user_id, sim_score in similar_users.items():
        sim_user_tracks = set(user_track_matrix.columns[user_track_matrix.loc[sim_user_id] > 0])
        new_tracks = sim_user_tracks - user_tracks
        
        for track in new_tracks:
            if track not in recommendations:
                recommendations[track] = 0
            recommendations[track] += sim_score * user_track_matrix.loc[sim_user_id, track]
    
    # Sort recommendations by score
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_recommendations = [track for track, score in sorted_recs[:n_recommendations]]
    
    return top_recommendations

def content_based_filtering(user_profile, tracks_df, n_recommendations=5):
    """Simplified content-based filtering"""
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
        
        # Combine scores
        total_score = 0.6 * genre_score + 0.4 * mood_score
        
        track_scores.append((track['track_id'], total_score))
    
    # Sort by score
    sorted_tracks = sorted(track_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N
    top_tracks = [track_id for track_id, _ in sorted_tracks[:n_recommendations]]
    
    return top_tracks

def hybrid_recommendations(user_id, user_profile, user_track_matrix, tracks_df, 
                          collab_weight=0.6, content_weight=0.4, n_recommendations=5):
    """Simplified hybrid recommendation system"""
    # Get collaborative filtering recommendations
    cf_recs = collaborative_filtering(user_id, user_track_matrix, n_recommendations=10)
    
    # Get content-based recommendations
    cb_recs = content_based_filtering(user_profile, tracks_df, n_recommendations=10)
    
    # Combine recommendations with weights
    combined_scores = {}
    
    for i, track_id in enumerate(cf_recs):
        combined_scores[track_id] = collab_weight * (1 - i/len(cf_recs))
    
    for i, track_id in enumerate(cb_recs):
        if track_id in combined_scores:
            combined_scores[track_id] += content_weight * (1 - i/len(cb_recs))
        else:
            combined_scores[track_id] = content_weight * (1 - i/len(cb_recs))
    
    # Sort by score
    sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_recommendations = [track for track, _ in sorted_recs[:n_recommendations]]
    
    return top_recommendations

def display_track_card(track):
    """Display a simplified track card"""
    st.markdown(f"""
    <div class="track-card">
        <div><b>{track['title']}</b></div>
        <div>{track['artist']}</div>
        <div style="color: #ff5500;">{track['genre']} • {track['mood']} • {track['bpm']} BPM</div>
        <div>Plays: {format(track['plays'], ',')} • Likes: {format(track['likes'], ',')}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Title and description
    st.markdown('<h1 class="main-header">SoundCloud Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('Discover new music you\'ll love')
    
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
            ["Collaborative Filtering", "Content-Based", "Hybrid"]
        )
        
        # Number of recommendations
        n_recommendations = st.slider("Number of Recommendations", 3, 10, 5)
        
        if rec_method == "Hybrid":
            collab_weight = st.slider("Collaborative Weight", 0.0, 1.0, 0.6, 0.1)
            content_weight = st.slider("Content Weight", 0.0, 1.0, 0.4, 0.1)
        else:
            collab_weight = 0.6
            content_weight = 0.4
        
        st.button("Get Recommendations", type="primary")
    
    # Create user profile
    user_profile = create_user_profile(user_id, user_track_df, tracks_df)
    
    # Show user profile
    st.header(f"User Profile (ID: {user_id})")
    
    if user_profile:
        # Display user stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Tracks Listened", len(user_profile['listened_tracks']))
        
        with col2:
            top_genre = max(user_profile['genre_preferences'].items(), key=lambda x: x[1])[0]
            st.metric("Top Genre", top_genre)
        
        # Display user preferences
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Genre Preferences")
            genre_df = pd.DataFrame({
                'Genre': user_profile['genre_preferences'].keys(),
                'Preference': user_profile['genre_preferences'].values()
            }).sort_values('Preference', ascending=False)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x='Preference', y='Genre', data=genre_df, ax=ax)
            ax.set_xlim(0, 1)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Mood Preferences")
            mood_df = pd.DataFrame({
                'Mood': user_profile['mood_preferences'].keys(),
                'Preference': user_profile['mood_preferences'].values()
            }).sort_values('Preference', ascending=False)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x='Preference', y='Mood', data=mood_df, ax=ax)
            ax.set_xlim(0, 1)
            st.pyplot(fig)
    else:
        st.warning("No user profile available")
    
    # Generate recommendations
    st.header("Recommended Tracks")
    
    with st.spinner("Generating recommendations..."):
        if rec_method == "Collaborative Filtering":
            rec_track_ids = collaborative_filtering(user_id, user_track_matrix, n_recommendations)
            rec_description = "Based on similar users' listening habits"
        elif rec_method == "Content-Based":
            rec_track_ids = content_based_filtering(user_profile, tracks_df, n_recommendations)
            rec_description = "Based on your genre and mood preferences"
        else:  # Hybrid
            rec_track_ids = hybrid_recommendations(
                user_id, user_profile, user_track_matrix, tracks_df, 
                collab_weight, content_weight, n_recommendations
            )
            rec_description = "Combining similar users' habits and your preferences"
    
    st.caption(f"**{rec_description}**")
    
    # Display recommendations
    recommended_tracks = tracks_df[tracks_df['track_id'].isin(rec_track_ids)]
    
    for _, track in recommended_tracks.iterrows():
        display_track_card(track)

if __name__ == "__main__":
    main()
