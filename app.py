import streamlit as st
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import json

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_movie_recommendations(movie_name, movie_year, user_preferences=""):
    """Generate movie recommendations based on input movie and user preferences"""
    
    sys_instruct = f"""
You are an expert film analyst with comprehensive knowledge of movies across all genres and eras.
Your task is to recommend movies similar to {movie_name} ({movie_year}) based on:
1. Similar plot/story elements
2. Matching tone and atmosphere
3. Comparable themes and messages
4. Similar directorial style or cinematography
5. Shared cast members when relevant
{"" if not user_preferences else f"6. User-specified preferences: {user_preferences}"}

Provide exactly 5 recommendations with:
- Movie title (with release year)
- Brief similarity explanation (what makes it similar)
- Key shared elements (genre, director, cast, etc.)
- Where available to stream (Netflix, Prime, etc.)

Format the output in clear markdown with bold titles and bullet points.
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=sys_instruct
        ),
        contents=f"Recommend movies similar to {movie_name} ({movie_year}) with these preferences: {user_preferences}"
    )
    
    return response.text

# Streamlit app
st.title("🎬 Cinematic Companion - Movie Recommendation Engine")

col1, col2 = st.columns(2)
with col1:
    movie_name = st.text_input("Movie Name", placeholder="e.g., Inception")
with col2:
    movie_year = st.text_input("Release Year", placeholder="e.g., 2010")

user_preferences = st.text_area("What did you particularly enjoy about this movie? (Optional)", 
                              placeholder="e.g., the mind-bending plot twists, Hans Zimmer's score, the visual effects...",
                              height=100)

if st.button("Get Recommendations"):
    if movie_name and movie_year:
        with st.spinner("Analyzing your movie and finding perfect matches..."):
            recommendations = generate_movie_recommendations(movie_name, movie_year, user_preferences)
        
        st.markdown("## Your Personalized Movie Recommendations")
        st.markdown(recommendations, unsafe_allow_html=True)
        
        st.info("💡 Tip: The more specific your preferences, the better the recommendations!")
    else:
        st.error("Please provide at least the movie name and release year")

# Add some sample prompts
with st.expander("💡 Example Queries"):
    st.markdown("""
    - **Movie:** The Dark Knight (2008)  
      **Preferences:** Heath Ledger's performance, the gritty tone, moral complexity
    
    - **Movie:** Parasite (2019)  
      **Preferences:** social commentary, genre-blending, unexpected plot turns
    
    - **Movie:** Pride & Prejudice (2005)  
      **Preferences:** period romance, strong female lead, witty dialogue
    """)

# Add footer
st.markdown("---")
st.caption("Powered by Google Gemini - Finding your next favorite movie since 2024")
