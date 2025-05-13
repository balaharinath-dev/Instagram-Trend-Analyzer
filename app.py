import streamlit as st
import requests
import time
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from collections import Counter

# Load environment variables
load_dotenv()
subscription_key = st.secrets["SUBSCRIPTION_KEY"]
API_TOKEN = st.secrets["API_TOKEN"]
endpoint = st.secrets["AZURE_ENDPOINT"]
api_version = st.secrets["AZURE_API_VERSION"]
deployment = st.secrets["AZURE_DEPLOYMENT_NAME"]

# Initialize model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Streamlit page configuration with Instagram theme
st.set_page_config(page_title="Instagram Trend Analyzer", page_icon="📸", layout="wide")

# Custom CSS for Instagram theme with improved color adaptability
st.markdown("""
<style>
    /* Instagram color scheme - updated with more vibrant colors */
    :root {
        --instagram-pink: #E1306C;
        --instagram-purple: #5851DB; /* Updated from #833AB4 to more vibrant purple */
        --instagram-orange: #F77737;
        --instagram-yellow: #FCAF45;
        --instagram-red: #FD1D1D;
        --instagram-blue: #405DE6;
    }
    
    /* Global styles - removing the fixed text colors */
    body {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Header styling - using more vibrant purple */
    
    h1, h2, h3{
            color: #ffffff
            }
    
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, var(--instagram-purple), var(--instagram-pink), var(--instagram-orange));
        color: white !important;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }
    
    /* Card styling for posts with theme-aware text */
    .post-card {
        border: 1px solid #DBDBDB;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        background-color: white !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        height: 250px !important;
        overflow: auto !important;
    }
    
    /* Dark mode text adjustments */
    .dark-mode-text {
        color: black !important;
        
    }
    
    /* Light mode specific styles */
    @media (prefers-color-scheme: light) {
        .post-card {
            color: #262626;
            height: 200px !important;
            overflow: auto !important;
        }
        .content-text {
            color: #262626;
        }
    }
    
    /* Dark mode specific styles */
    @media (prefers-color-scheme: dark) {
        .post-card {
            background-color: #262626;
            color: black;
            height: 200px !important;
            overflow: auto !important;
        }
        .content-text {
            color: #FAFAFA;
        }
    }
    
    /* Instagram-like buttons */
    .instagram-button {
        display: inline-block;
        padding: 8px 16px;
        background: linear-gradient(45deg, var(--instagram-purple), var(--instagram-pink));
        color: white;
        text-decoration: none;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    
    /* Tab styling - improved spacing and appearance */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; /* Increased spacing between tabs */
        padding: 10px 0; /* Add vertical padding for tabs container */
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px; /* Slightly more rounded corners */
        font-weight: 600;
        padding: 8px 16px; /* Add more padding within each tab */
        margin-right: 5px; /* Additional spacing between tabs */
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #5851DB, var(--instagram-pink)) !important; /* Gradient background for selected tab */
        color: white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* Subtle shadow for selected tab */
    }
    
    /* Tab panel styling */
    .stTabs [data-baseweb="tab-panel"] {
        padding: 20px 10px; /* Add padding inside tab panels */
    }
</style>
""", unsafe_allow_html=True)

def find_all_key_values(data, target_key):
    results = []
    def recurse(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == target_key:
                    results.append(value)
                if isinstance(value, (dict, list)):
                    recurse(value)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)
    recurse(data)
    return results

def extract_posts_with_locations(items):
    """Extract posts with their corresponding locations"""
    posts_with_locations = []

    # Process the top-level location data and posts
    top_posts = find_all_key_values(items, "topPosts")

    # Extract location data from each item
    for i, item in enumerate(items):
        # Each item might have a locationName
        location_name = item.get("locationName", None)
        
        # Each item might have a list of top posts
        if "topPosts" in item and isinstance(item["topPosts"], list):
            for post in item["topPosts"]:
                post_data = {
                    "location": location_name,
                    "url": post.get("url", ""),
                    "caption": post.get("caption", ""),
                    "hashtags": post.get("hashtags", []),
                    "thumbnail": post.get("cover_artwork_thumbnail_uri", ""),
                    "likes": post.get("likesCount", 0)
                }
                posts_with_locations.append(post_data)

    # If we couldn't extract them properly, just use the raw data
    if not posts_with_locations and top_posts:
        # Fall back to plain extraction
        for post in top_posts:
            if isinstance(post, dict):
                posts_with_locations.append({
                    "location": None,
                    "url": post.get("url", ""),
                    "caption": post.get("caption", ""),
                    "hashtags": post.get("hashtags", []),
                    "thumbnail": post.get("thumbnailUrl", ""),
                    "likes": post.get("likesCount", 0)
                })

    return posts_with_locations

def run_analysis(searched_term, min_likes=0):
    run_url = f"https://api.apify.com/v2/acts/apify~instagram-scraper/runs?token={API_TOKEN}"
    payload = {
        "addParentData": False,
        "enhanceUserSearchWithFacebookPage": False,
        "isUserReelFeedURL": False,
        "isUserTaggedFeedURL": False,
        "resultsLimit": 200,  # Increased to get more results
        "resultsType": "details",
        "search": f"{searched_term}",
        "searchLimit": 5,  # Increased to get more results
        "searchType": "hashtag"
    }
    response = requests.post(run_url, json=payload, verify=False)
    run_data = response.json()
    run_id = run_data["data"]["id"]

    status_url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={API_TOKEN}"
    while True:
        status_response = requests.get(status_url, verify=False)
        status_data = status_response.json()
        status = status_data["data"]["status"]
        if status in ["SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"]:
            break
        time.sleep(5)

    default_dataset_id = status_data["data"]["defaultDatasetId"]

    dataset_url = f"https://api.apify.com/v2/datasets/{default_dataset_id}/items?token={API_TOKEN}"
    dataset_response = requests.get(dataset_url, verify=False)
    items = dataset_response.json()

    # Extract all available locations
    all_locations = find_all_key_values(items, "locationName")
    unique_locations = list(set([loc for loc in all_locations if loc]))

    # Extract posts with their locations
    posts_with_locations = extract_posts_with_locations(items)
    
    # Filter posts by minimum likes
    posts = [post for post in posts_with_locations if post.get("likes", 0) >= min_likes]

    # If no posts match our filter, fall back to all posts
    if not posts:
        posts = posts_with_locations

    # Sort posts by likes count (most likes first)
    posts = sorted(posts, key=lambda x: x.get("likes", 0), reverse=True)

    # Extract data from filtered posts
    captions_list = [post.get("caption", "") for post in posts if post.get("caption")]
    hashtag_lists = [post.get("hashtags", []) for post in posts if post.get("hashtags")]
    hashtag_list = [tag for sublist in hashtag_lists for tag in sublist]
    url_list = [post.get("url", "") for post in posts if post.get("url")]
    thumbnails = [post.get("thumbnail", "") for post in posts if post.get("thumbnail")]

    # Enhanced posts for visualization with thumbnails and URLs
    post_display_data = []
    for post in posts:
        if post.get("url") and post.get("caption"):
            post_display_data.append({
                "url": post.get("url", ""),
                "caption": post.get("caption", ""),
                "thumbnail": post.get("thumbnail", ""),
                "location": post.get("location", "Unknown"),
                "likes": post.get("likes", 0)
            })

    prompt1 = f'''
I've collected captions from top-trending Instagram posts related to the keyword: "{searched_term}".

🔍 Please analyze these captions to extract the **underlying trends** in the domain of "{searched_term}". Your goal is not to summarize the captions, but to identify:

* The popular activities, ideas, or content themes people are engaging with
* What's driving attention and interaction in this space
* How users are expressing or showcasing "{searched_term}"

🤔 What's trending in the world of "{searched_term}" based on these captions?
Make the output insightful, engaging, and formatted in en-US English.
'''

    prompt2 = f'''
Here is a list of hashtags from top-performing Instagram posts, all related to the keyword: "{searched_term}".

🎯 Analyze these hashtags to identify the **real-world trends** in the domain of "{searched_term}". Focus on:

* Common patterns or subtopics that appear frequently
* Emerging communities, movements, or events
* The kind of content these hashtags are attached to

💡 Based on these hashtags, what's trending in the "{searched_term}" space?
'''

    prompt3 = f'''
I have gathered:
📄 Captions from trending Instagram posts about "{searched_term}"
🔖 Hashtags from those same top-performing posts

🔍 Please combine both sources to generate a unified insight report showing **what's actually trending in the domain of "{searched_term}"** on Instagram.
📢 Final goal: Tell me what's trending in "{searched_term}" *based on real post behavior*, not just the words.
Make it comprehensive, engaging, and clearly structured in en-US English.
'''

    captions_text = "\n".join(captions_list)
    hashtag_text = "\n".join(hashtag_list)

    response1 = model.invoke(prompt1 + captions_text)
    response2 = model.invoke(prompt2 + hashtag_text)
    response_final = model.invoke(prompt3 + response1.content + response2.content)

    # Generate hashtag string for wordcloud
    most_common_hashtags = Counter(hashtag_list).most_common(15)
    top_15=[i for i, _ in most_common_hashtags]
    hashtag_string = " ".join(top_15)

    return {
        "searched_term": searched_term,
        "captions": captions_list,
        "hashtags": hashtag_list,
        "insight": response_final.content,
        "locations": unique_locations,
        "hashtag_string": hashtag_string,
        "post_display_data": post_display_data,
        "url_list": url_list
    }
    
# st.sidebar.markdown("""
#         <style>
#             .red-button {
#                 display: inline-block;
#                 padding: 0.5em 1em;
#                 color: white !important;
#                 background-color: orange !important;
#                 border-radius: 15px !important;
#                 text-decoration: none !important;
#                 font-weight: bold;
#                 text-align: center;
#                 width: 30%;
#                 margin-bottom: 10px;
#             }
#             .red-button:hover {
#                 background-color: orange !important; /* darker on hover */
#             }
#         </style>

#         <a href="http://localhost:8501" target="_self" class="red-button">Back</a>
#     """, unsafe_allow_html=True)

# Sidebar UI with Instagram theme
with st.sidebar:
    st.image("instagram.png", width=200)
    st.title("📲 InstaTrendz")
    
    # Create a container with gradient background
    with st.container():
        st.markdown("""
        <div style="padding: 15px; border-radius: 10px; background: linear-gradient(45deg, #833AB4, #E1306C, #F77737);">
        <h3 style="color: white; text-align: center;">Search Instagram Trends</h3>
        </div>
        """, unsafe_allow_html=True)
        
        searched_term = st.text_input("🔍 Search Hashtag", placeholder="#fashion, #travel, etc.")
        
        # Add minimum likes filter instead of location
        min_likes = st.slider("❤️ Minimum Likes", min_value=0, max_value=10000, value=0, step=100)
        
        # Styled button
        analyze = st.button("✨ Analyze Trends")

if analyze:
    with st.spinner("📱 Analyzing Instagram trends..."):
        result = run_analysis(searched_term, min_likes)
        st.session_state["result"] = result
        st.session_state["min_likes"] = min_likes

# Main content
if "result" in st.session_state:
    result = st.session_state["result"]
    
    # Re-analyze if minimum likes filter has changed
    if "min_likes" in st.session_state and st.session_state["min_likes"] != min_likes:
        with st.spinner(f"Filtering results for minimum {min_likes} likes..."):
            result = run_analysis(searched_term, min_likes)
            st.session_state["result"] = result
            st.session_state["min_likes"] = min_likes

    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, #833AB4, #E1306C, #F77737); color: white; border-radius: 10px; margin-bottom: 20px;">
        <h1>📸 Instagram Trends: #{result['searched_term']}</h1>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📢 Insight", "📸 Captions", "🏷️ Hashtags", "🔝 Posts"])

    with tab1:
        st.header(f"📈 Trends for: #{result['searched_term']}")
        
        # Display wordcloud in the final insights tab
        st.subheader("☁️ Trending Topics Word Cloud")
        # Generate wordcloud here instead of storing the object
        if result["hashtag_string"]:
            wordcloud = WordCloud(width=800, height=400, background_color="white", 
                                  colormap="plasma").generate(result["hashtag_string"])
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)
        else:
            st.info("No hashtags available to generate wordcloud.")
        
        st.subheader("🔍 Trend Analysis")
        
        st.markdown(
            f"<div style='color: var(--text-color); font-size: 16px; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>{result['insight']}</div>",
            unsafe_allow_html=True
        )

    with tab2:
        st.subheader("🔟 Top Trending Captions")
        
        # Create caption cards with Instagram styling
        top_captions = result["captions"][:10]
        for i, cap in enumerate(top_captions, 1):
            st.markdown(f"""
            <div class="post-card">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="background: linear-gradient(45deg, #833AB4, #E1306C, #F77737); color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                        {i}
                    </div>
                    <div class="dark-mode-text" style="font-weight: bold;">Post #{i}</div>
                </div>
                <div class="dark-mode-text" style="padding: 5px 0;">{cap}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.subheader("📊 Hashtag Analytics")
        
        # Interactive hashtag analysis
        hashtag_freq = pd.Series(result["hashtags"]).value_counts()
        
        # Top hashtags chart using Plotly with Instagram colors
        top_hashtags = hashtag_freq.head(10).reset_index()
        top_hashtags.columns = ['Hashtag', 'Count']
        
        fig = px.bar(
            top_hashtags, 
            x='Hashtag', 
            y='Count',
            color='Count',
            color_continuous_scale=['#833AB4', '#E1306C', '#F77737', '#FCAF45'],
            title="Top 10 Hashtags"
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("🔝 Top Trending Posts")
        
        # Add sorting options
        sort_by = st.radio("Sort by:", ["Most Likes", "Recently Added"], horizontal=True)
        
        if result["post_display_data"]:
            # Sort posts based on selection
            posts = result["post_display_data"]
            if sort_by == "Most Likes":
                posts = sorted(posts, key=lambda x: x.get("likes", 0), reverse=True)
            
            # Create 3 columns for posts display
            cols = st.columns(3)
            
            for i, post in enumerate(posts[:9]):  # Display up to 9 posts
                with cols[i % 3]:
                    # Fix HTML rendering issues by using Streamlit components instead
                    with st.container():
                        st.markdown(f"""
                            <div class="post-card" style="padding: 10px; border: 1px solid #ccc; border-radius: 8px; margin-bottom: 15px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <div class="dark-mode-text" style="font-weight: bold;">Post #{i+1}</div>
                                    <div style="color: #E1306C;">❤️ {post.get('likes', 0)}</div>
                                </div>
                                <div style="margin-top: 8px;">
                                    <span style="font-weight: 600;">Caption:</span>
                                    <span>{post["caption"][:100] + "..." if len(post["caption"]) > 100 else post["caption"]}</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Use Streamlit button for the link
                        st.markdown(f"""
                        <a href="{post['url']}" target="_blank" style="text-decoration: none;">
                            <div class="instagram-button" style="width: 100%; text-align: center; padding: 8px 16px; background: linear-gradient(45deg, #833AB4, #E1306C, #F77737); color: white; border-radius: 5px; font-weight: bold;">
                                View on Instagram
                            </div>
                        </a>
                        """, unsafe_allow_html=True)
                        
                        # Add space between posts
                        st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
        else:
            st.info("No post data available to display.")

else:
    # Welcome screen with Instagram styling
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(45deg, #833AB4, #E1306C, #F77737); color: white; border-radius: 10px; margin: 20px 0;">
        <h1>📸 Welcome to InstaTrendz!</h1>
        <p style="font-size: 18px;">Discover what's trending on Instagram in real-time</p>
    </div>
    
    <div class="post-card" style="padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <h3 style="color: #833AB4;">How it works:</h3>
        <ol class="dark-mode-text">
            <li>👈 Enter a hashtag in the sidebar (e.g., fashion, travel, fitness)</li>
            <li>❤️ Set minimum likes filter (optional)</li>
            <li>🔍 Click "Analyze Trends" to discover what's popular</li>
            <li>📊 View insights, trending hashtags, and top posts</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)