# ========================================================================================
# FYP-JESSLYN ANG KAR WEI-TP069272-APD3F2505CS(DA)
# ========================================================================================
# FYP Title: Detecting Fake News Using Machine Learning (ML) Algorithms for Internet Users
# ========================================================================================

# System Deployment (Streamlit)

# ----------------
# Import Libraries
# ----------------
import os
import zipfile
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
from textblob import TextBlob
import textstat
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import time
import plotly.express as px
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)


# ---------------------------
# Custom Styles for Streamlit
# ---------------------------
background_url = "https://wallpapers.com/images/hd/minimalist-gray-f8tfvmsugbfylndq.jpg"

st.markdown(
    f"""
    <style>
    /* App Background */
    [data-testid="stAppViewContainer"] {{
        background-image: url("{background_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}

    /* Button Styling */
    div.stButton > button:first-child {
        background-color: #6395EE !important;
        color: black !important;
        font-size: 20px !important; 
        font-weight: bold !important;
        height: 60px !important;  
        width: 250px !important;  
        border-radius: 10px !important;
        border: 2px solid black !important;
        transition: all 0.2s ease;
        text-decoration: none !important;
    }

    /* Center the button container */
    div.stButton {
        text-align: center;
        padding-top: 20px;
        padding-bottom: 20px;
    }

    /* Navy Blue Border and Glow when clicked */
    div.stButton > button:first-child:active {{
        border-color: #000080 !important; 
        box-shadow: 0px 0px 10px #000080 !important;
        transform: scale(0.95) !important; 
        background-color: #6395EE !important;
    }}
    
    /* Ensure no underline appears on hover/focus */
    div.stButton > button:first-child:hover, 
    div.stButton > button:first-child:focus {{
        text-decoration: none !important;
        color: black !important;
        border-color: black !important;
    }}

    /* Text Area Styling (Wider, Taller, and Larger Font) */
    /* Note the double braces {{ }} below to prevent Python errors */
    div.stTextArea > div > textarea {{
        background-color: white !important;
        color: black !important;
        font-size: 24px !important;
        font-weight: normal !important;
        min-height: 400px !important;
        width: 100% !important;
        border: 2px solid #6395EE !important;
        border-radius: 10px !important;
        line-height: 1.5 !important;
    }}

    /* Ensuring the container allows the width to expand */
    [data-testid="stTextArea"] {{
        width: 100% !important;
        padding: 0px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# Load Paths and Model + Preprocessors + Datasets
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "BEST_MODEL_FOR_DEPLOYMENT_Gradient_Boosting.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "models", "fitted_tfidf_vectorizer.joblib")
OHE_PATH = os.path.join(BASE_DIR, "models", "fitted_ohe_encoder.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "fitted_minmax_scaler.joblib")
MASTER_FEATURES_PATH = os.path.join(BASE_DIR, "models", "master_feature_names.joblib")

# Load artifacts
model = joblib.load(MODEL_PATH)
tfidf_vectorizer = joblib.load(TFIDF_PATH)
ohe_encoder = joblib.load(OHE_PATH)
scaler = joblib.load(SCALER_PATH)
master_features = joblib.load(MASTER_FEATURES_PATH)
master_features = master_features.tolist() 

# Deployment folder
DEPLOY_DIR = os.path.join(BASE_DIR, "deployment_data")
os.makedirs(DEPLOY_DIR, exist_ok=True)

# ZIP paths inside deployment_data folder
TRAIN_ZIP = os.path.join(DEPLOY_DIR, "Deployment_Fake_News_Train.zip")
TEST_ZIP  = os.path.join(DEPLOY_DIR, "Deployment_Fake_News_Test.zip")

# CSV paths after extraction
TRAIN_CSV = os.path.join(DEPLOY_DIR, "Deployment_Fake_News_Train.csv")
TEST_CSV  = os.path.join(DEPLOY_DIR, "Deployment_Fake_News_Test.csv")

# Extract ZIP if CSV doesn't exist
def extract_if_needed(zip_path, csv_path):
    if not os.path.exists(csv_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DEPLOY_DIR)
            print(f"Extracted {csv_path} from {zip_path}")

extract_if_needed(TRAIN_ZIP, TRAIN_CSV)
extract_if_needed(TEST_ZIP, TEST_CSV)

# Load CSVs
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)


# Combined dataset for proof + deployment metrics
master_lookup_df = pd.concat([train_df, test_df]).drop_duplicates(subset=["full_content"])
dep_df = pd.concat([train_df, test_df]).reset_index(drop=True)

categories = train_df["category"].unique()
countries = train_df["country"].unique()


# ----------------------------------------------
# Precompute Category TF-IDF for Fast Similarity
# ----------------------------------------------
# For each category, concatenate all full_content into one big document
category_corpus = (
    train_df
    .groupby("category")["full_content"]
    .apply(lambda x: " ".join(str(t) for t in x))
    .reindex(categories)  # keep order aligned with categories array
)

# Transform these concatenated texts into TF-IDF vectors
category_tfidf_matrix = tfidf_vectorizer.transform(category_corpus.fillna(""))


# ------------------
# Utility Functions
# ------------------
def find_dataset_truth(text):
    clean_text = text.strip()[:200]
    match = master_lookup_df[
        master_lookup_df["full_content"].str.contains(clean_text, case=False, na=False, regex=False)
    ]
    if not match.empty:
        res = match.iloc[0]
        return {
            "country": res["country"],
            "category": res["category"],
            "label": "Fake" if res["label"] == 1 else "Real",
            "found": True
        }
    return {"country": "Unknown", "category": "Politics", "label": "N/A", "found": False}


# -------------------------------------
# Feature Attribution / Explainability
# -------------------------------------
def extract_model_influential_phrases(raw_text, pred_label, top_n=10):
    """
    Extract top influential words contributing to Fake / Real prediction
    using TF-IDF activation weighted by model feature importance and master feature list.
    """

    # Transform input text using fitted TF-IDF
    tfidf_vector = tfidf_vectorizer.transform([raw_text]).toarray()[0]
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Align model importances with master features
    model_importances = []
    for feat in feature_names:
        if feat in master_features:
            idx = master_features.index(feat)
            model_importances.append(model.feature_importances_[idx])
        else:
            model_importances.append(0)  # For unseen words, assume zero importance

    # Word frequency
    word_counts = {}
    for w in raw_text.lower().split():
        word_counts[w] = word_counts.get(w, 0) + 1

    leaning = "Fake-leaning" if pred_label == 1 else "Real-leaning"
    phrase_scores = []
    total_score = 0

    for i, phrase in enumerate(feature_names):
        tfidf_value = tfidf_vector[i]
        if tfidf_value < 0.0001:
            continue

        score = tfidf_value * model_importances[i]
        total_score += score

        phrase_scores.append({
            "Word": phrase,
            "Frequency": word_counts.get(phrase, 0),
            "Score": round(score, 6)
        })

    if not phrase_scores:
        return []

    for phrase in phrase_scores:
        contribution = (phrase["Score"] / total_score * 100) if total_score > 0 else 0
        phrase["Contribution %"] = f"{contribution:.2f}%"

        if contribution < 1:
            phrase["Comment"] = f"⏬ Low Impact of {leaning}"
        elif contribution < 5:
            phrase["Comment"] = f"☑️ Moderate Impact of {leaning}"
        else:
            phrase["Comment"] = f"🚨 High Impact of {leaning}"

    return sorted(
        phrase_scores,
        key=lambda x: float(x["Contribution %"].replace("%", "")),
        reverse=True
    )[:top_n]


# -----------------------------------------------------------------------------------------------------------
# Generate the NLP Features such as Word Counts, Sentiment Score, Readability Score, and Average Word Length 
# -----------------------------------------------------------------------------------------------------------
def compute_features(text):
    words = text.split()
    word_count = len(words)
    sentiment_score = TextBlob(text).sentiment.polarity
    readability_score = textstat.flesch_reading_ease(text)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    return word_count, sentiment_score, readability_score, avg_word_length


# --------------------------------------
# Identify the Country from the Dataset 
# --------------------------------------
def detect_country_dataset(text):
    """
    Detect country strictly from TRAIN dataset
    by checking which country's articles contain similar content.
    """
    snippet = text[:200]
    for ctry in countries:
        subset = train_df[train_df["country"] == ctry]
        if subset["full_content"].str.contains(snippet, case=False, na=False, regex=False).any():
            return ctry
    return "Unknown"


# --------------------------------------------------------------------------------------------
# Identify the Category Similarity by Using the Keywords in the News Content
# --------------------------------------------------------------------------------------------
def detect_category_similarity_tfidf(text):
    """
    Fast category similarity using TF-IDF cosine similarity (A2-Lite):
    - Transform input text into TF-IDF vector.
    - Compute cosine similarity with each category's concatenated TF-IDF vector.
    - Return similarity percentages.
    """
    # Transform input text
    text_tfidf = tfidf_vectorizer.transform([text])

    # Cosine similarity between input and each category document
    sims = cosine_similarity(text_tfidf, category_tfidf_matrix)[0]  # shape: (n_categories,)

    # Convert to non-negative and normalize to percentages
    sims = np.maximum(sims, 0)
    total = sims.sum()
    if total == 0:
        return {cat: 0.0 for cat in categories}

    percents = (sims / total) * 100
    return {cat: round(p, 2) for cat, p in zip(categories, percents)}


# -----------------------------
# Plot the Category Similarity 
# -----------------------------
def plot_category_bar(category_percent):
    df = pd.DataFrame({
        "Category": list(category_percent.keys()),
        "Percentage": list(category_percent.values())
    })

    fig = px.bar(
        df,
        x="Category",
        y="Percentage",
        color="Percentage",
        color_continuous_scale="Plasma",
        text="Percentage",
    )

    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Similarity (%)",
        font=dict(size=18),  
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    st.plotly_chart(fig, use_container_width=True)

    final_category = df.loc[df["Percentage"].idxmax(), "Category"]
    return final_category


# -------------------------
# Plot the Map for Country
# -------------------------
def plot_country_map(detected_country):
    df = pd.DataFrame({
        "country": countries,
        "value": [1 if c == detected_country else 0 for c in countries]
    })

    fig = px.choropleth(
        df,
        locations="country",
        locationmode="country names",
        color="value",
        color_continuous_scale=["lightgrey", "orange"],
        range_color=(0, 1),
    )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------
# Preprocess the Text for Category and Country
# ----------------------------------------------
def preprocess_text(text, category, country):
    tfidf_vec = tfidf_vectorizer.transform([text])
    cat_input = pd.DataFrame([[category, country]], columns=["category", "country"])
    ohe_vec = ohe_encoder.transform(cat_input)
    wc, sent, read, avg_l = compute_features(text)
    numeric_vec = scaler.transform([[wc, sent, read, avg_l]])
    return hstack([tfidf_vec, ohe_vec, numeric_vec]), wc, sent, read, avg_l


# --------------------------------------------------------
# Plot the Most Frequent Words Found in the News Category
# --------------------------------------------------------
def most_frequent_words(text, top_n=10):
    # Tokenize and clean text
    words = [w.lower() for w in text.split() if w.isalpha()]
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]  # remove stopwords

    counter = Counter(words)
    total_words = sum(counter.values())
    most_common = counter.most_common(top_n)

    if most_common:
        words, counts = zip(*most_common)
        percentages = [count / total_words * 100 for count in counts]

        df_words = pd.DataFrame({
            "Word": [w.capitalize() for w in words],  
            "Percentage": percentages,
            "Count": counts
        }).sort_values(by="Percentage", ascending=True)

        fig = px.bar(
            df_words,
            x="Percentage",
            y="Word",
            orientation="h",
            color="Percentage",
            color_continuous_scale="Blues",  
            custom_data=["Count"] 
        )

        fig.update_traces(
            texttemplate='%{x:.1f}%', 
            textposition='outside',
            cliponaxis=False,
            hovertemplate='Word: %{y}<br>Count: %{customdata}<br>Percentage: %{x:.1f}%'
        )

        fig.update_layout(
            font=dict(size=18),
            height=500,
            xaxis_title=dict(text="Percentage of Total Words (%)", font=dict(size=18)),
            yaxis_title=dict(text="Word", font=dict(size=18)),
            margin=dict(l=120, r=50, t=80, b=80),
            showlegend=False,
            coloraxis_showscale=False,
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------
# Plot the Word Cloud for the News Content
# ----------------------------------------
def generate_wordcloud(text):
    # Remove stopwords
    stop_words = set(STOPWORDS)

    wc = WordCloud(
        width=2000,
        height=1000,
        background_color="white",
        colormap="viridis",  
        stopwords=stop_words,
        max_font_size=200,
        relative_scaling=0.5
    ).generate(text)

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10), dpi=120)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    # Center the figure in Streamlit
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig, clear_figure=True, use_container_width=False)


# --------------------------------------------------
# Plot the Word Count Analysis for the News Content
# --------------------------------------------------
def plot_word_count_analysis(word_count):
    # Define thresholds
    if word_count < 200:
        label = "Short Snippet"
        comment = "It is <strong>very short</strong>. Likely a social media post or a clickbait blurb."
        color = "#FF4B4B" # Red (Alert)
    elif 200 <= word_count <= 800:
        label = "Standard News"
        comment = "It is a <strong>typical length</strong> for a professional news article."
        color = "#23C552" # Green (Balanced)
    else:
        label = "Long-form / Detailed"
        comment = "It is <strong>high detail</strong>. Typical of deep-dive investigative journalism or academic papers."
        color = "#6395EE" # Blue

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = word_count,

        title = {'text': f"Length Category:  {label}", 'font': {'size': 24}},
        number = {'font': {'size': 40}}, 
        gauge = {
            'axis': {'range': [0, 1500], 'tickfont': {'size': 16}}, 
            'bar': {'color': color},
            'steps': [
                {'range': [0, 200], 'color': "rgba(255, 75, 75, 0.2)"},
                {'range': [200, 800], 'color': "rgba(35, 197, 82, 0.2)"},
                {'range': [800, 1500], 'color': "rgba(99, 149, 238, 0.2)"}
            ],
        }
    ))
    
    fig.update_layout(
        height=300, 
        margin=dict(l=30, r=30, t=80, b=20),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="
        background-color: {color}; 
        border-left: 10px solid #000000; 
        padding: 20px; 
        border-radius: 10px;
        margin-top: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    ">
        <p style="
            margin: 0; 
            font-size: 20px; 
            color: white; 
            line-height: 1.5;
            letter-spacing: 0.5px;
        ">
            📝 <strong>Word Insight</strong>: {comment}
        </p>
    </div>
""", unsafe_allow_html=True)


# --------------------------------
# Plot the Sentiment Score / Gauge
# --------------------------------
def plot_sentiment_gauge(sentiment_score):
    # Define Logic & Comments
    if sentiment_score > 0.3:
        status = "POSITIVE/PROMOTIONAL"
        comment = "This text uses <strong>biased, positive language</strong>. It might be propaganda or a 'puff piece'."
        color = "#23C552" # Green
    elif sentiment_score < -0.3:
        status = "NEGATIVE/INFLAMMATORY"
        comment = "This text contains high levels of <strong>negative or angry language—common</strong> in 'outrage' fake news."
        color = "#FF4B4B" # Red
    else:
        status = "NEUTRAL/OBJECTIVE"
        comment = "The tone is <strong>balanced and factual</strong>. This is typical of professional, high-quality journalism."
        color = "#825E00" # Grey

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=["Sentiment"], x=[2], base=-1,
        orientation='h',
        marker=dict(color='#F0F2F6'), 
        hoverinfo='skip'
    ))

    fig.add_trace(go.Bar(
        y=["Sentiment"], 
        x=[sentiment_score],
        base=0,
        orientation='h',
        marker=dict(color=color),
        text=f" Score: {sentiment_score:.2f} ",
        textposition='auto',
        textfont=dict(size=18, color="white") 
    ))

    fig.update_layout(

        font=dict(size=18),
        title=dict(
            text=f"Tone Detection: {status}",
            font=dict(size=18),
            x=0.5,        
            xanchor='center'
        ),
        xaxis=dict(
            range=[-1, 1], 
            tickvals=[-1, -0.5, 0, 0.5, 1], 

            ticktext=["Extr. Negative", "Negative", "Neutral", "Positive", "Extr. Positive"],
            tickfont=dict(size=18) 
        ),
        yaxis=dict(visible=False),
        height=220,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="
        background-color: {color}; 
        border-left: 10px solid #000000; 
        padding: 20px; 
        border-radius: 10px;
        margin-top: 10px;
    ">
        <p style="
            margin: 0; 
            font-size: 20px; 
            color: white; 
            line-height: 1.5;
        ">
            🎭 <strong>Sentiment Score Analysis</strong>: {comment}
        </p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------
# Plot the Readability Score for the News with Specific Categories
# -----------------------------------------------------------------
def plot_readability_steps(score):
    # Mapping the categories
    levels = ["Professional", "University", "College", "Teen", "Fairly Easy", "Middle School", "Primary School"]
    
    # 1. Logic for Index and Specific Comments
    if score >= 90: 
        current_idx = 6
        comment = "This is a <strong>primary school level</strong>. It is extremely simple language. While accessible, very short or simple sentences are often used in low-effort clickbait."
        box_color = "#23C552"
    elif score >= 80: 
        current_idx = 5
        comment = "This is a <strong>middle school level</strong>. It is easy to consume. Great for reaching a wide audience quickly."
        box_color = "#23C552"
    elif score >= 70: 
        current_idx = 4
        comment = "This is <strong>fairly easy</strong>. It is clear and concise. This is the sweet spot for general news reporting."
        box_color = "#6395EE" 
    elif score >= 60: 
        current_idx = 3
        comment = "This is <strong>standard readability</strong>. It is comparable to a typical high school tabloid or digest."
        box_color = "#6395EE"
    elif score >= 50: 
        current_idx = 2
        comment = "This is a <strong>college level</strong>. It requires focus. This is a typical of specialized journalism or detailed editorial pieces."
        box_color = "#FFA500" 
    elif score >= 30: 
        current_idx = 1
        comment = "This is a <strong>university level</strong>. It is difficult and contains complex sentence structures as well as academic vocabulary."
        box_color = "#AD7000"
    else: 
        current_idx = 0
        comment = "This is <strong>professional/academic</strong>. It is very difficult and is likely a white paper, legal document, or high-level scientific report."
        box_color = "#FF4B4B" 

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[1]*7, y=levels, orientation='h',
        marker=dict(color='rgba(220, 220, 220, 0.4)'),
        hoverinfo='skip'
    ))
 
    fig.add_trace(go.Bar(
        x=[1], y=[levels[current_idx]], orientation='h',
        marker=dict(color=box_color), # Matches the logic color
        text=f" Score: {score} - {levels[current_idx]} ", 
        textposition='inside',

        insidetextfont=dict(size=22, color='white', family="Arial Black")
    ))

    fig.update_layout(
        title=dict(
            text="", 
            font=dict(size=18)
        ),
        xaxis=dict(visible=False), 

        yaxis=dict(
            tickfont=dict(size=18), 
            title=None
        ),
        showlegend=False, 
        height=550,
        margin=dict(l=20, r=20, t=80, b=20),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="
        background-color: {box_color}; 
        border-left: 10px solid #000000; 
        padding: 20px; 
        border-radius: 10px;
        margin-top: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    ">
        <p style="
            margin: 0; 
            font-size: 20px; 
            color: white; 
            line-height: 1.5;
        ">
            📚 <strong>Readability Insight</strong>: {comment}
        </p>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------
# Plot the Lexical Meter for Average Word Length
# ----------------------------------------------
def plot_lexical_meter(avg_word_len):
    # Define the Logic
    if avg_word_len < 4.5:
        comment = "This news is <strong>low complexity</strong> where it uses very simple language, typical of social media or informal conversation."
        box_color = "#FFA500" 
    elif 4.5 <= avg_word_len <= 6.0:
        comment = "This news is <strong>standard complexity</strong> where it matches the vocabulary levels of professional journalism."
        box_color = "#11A33A"
    else:
        comment = "This news is <strong>high complexity</strong> where it uses sophisticated, academic, and technical vocabulary."
        box_color = "#6395EE"

    data = {
        'Category': ['Social Media', 'Conversational', 'Journalism', 'Academic/Legal', 'Your Article'],
        'Avg Length': [4.0, 4.8, 5.3, 6.5, avg_word_len],
    }
    df = pd.DataFrame(data)

    fig = px.scatter(
        df,
        x='Avg Length',
        y='Category',
        color='Category',
        color_discrete_map={
            'Your Article': '#6395EE', 
            'Social Media': 'grey', 'Conversational': 'grey',
            'Journalism': 'grey', 'Academic/Legal': 'grey'
        }
    )

    fig.update_layout(title={"text": ""}, margin=dict(l=160, r=50, t=20, b=100))

    fig.update_layout(margin=dict(l=160, r=50, t=20, b=100))
    
    fig.add_vline(x=avg_word_len, line_width=2, line_dash="dash", line_color="#6395EE")

    fig.update_traces(marker=dict(size=22)) 


    fig.update_layout(
        font=dict(size=18), 
        showlegend=False,
        height=480,        
        title=dict(
            font=dict(size=18)
        ),
        xaxis=dict(
            title=dict(text="Average Characters per Word", font=dict(size=18)),
            tickfont=dict(size=18),
            range=[3, 8], 
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            title="", 
            tickfont=dict(size=18) 
        ),

        margin=dict(
            l=160,
            r=50, 
            t=80, 
            b=100 
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="
        background-color: {box_color}; 
        border-left: 10px solid #000000; 
        padding: 20px; 
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 20px;
    ">
        <p style="
            margin: 0; 
            font-size: 20px; 
            color: white; 
            line-height: 1.5;
        ">
            🔍 <strong>Lexical Analysis</strong>: {comment}
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------
# Plot Entropy / Uncertainty 
# ---------------------------

# Entropy measures how uncertain the model is about its prediction.
# It looks at the probability distribution between Fake and Real.
# If probabilities are very skewed (e.g., 0.95 Fake, 0.05 Real), entropy is low → the model is confident.
# If probabilities are balanced (e.g., 0.50 Fake, 0.50 Real), entropy is high → the model is uncertain.

def plot_entropy_bar(pred_prob_fake, pred_prob_real):

    # Probabilities
    probs = np.array([pred_prob_fake, pred_prob_real])

    # Compute entropy
    entropy = -np.sum(probs * np.log(probs + 1e-12))  # add epsilon to avoid log(0)
    max_entropy = np.log(len(probs))  # maximum possible entropy for 2 classes = log(2)
    normalized_entropy = entropy / max_entropy

    fig = go.Figure(go.Bar(
        x=[normalized_entropy],
        y=["Entropy / Uncertainty"],
        orientation="h",
        marker=dict(color="rgba(255,165,0,0.8)"),
        text=[f"{normalized_entropy:.2f}"],
        textposition="outside",
        textfont_size=18  
    ))

    fig.update_layout(
        xaxis=dict(range=[0,1], title="Normalized Entropy", title_font=dict(size=18), tickfont=dict(size=18)),
        yaxis=dict(showticklabels=False, title_font=dict(size=18), tickfont=dict(size=18)),
        height=200,
        margin=dict(l=50, r=50, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_layout(legend=dict(font=dict(size=18)))

    st.plotly_chart(fig, use_container_width=True)

    if normalized_entropy < 0.3:
        comment = "This is <strong>low entropy</strong>. The model is confident and reliable for this text."
        color = "#0A9D33" 
    elif normalized_entropy < 0.7:
        comment = "This is <strong>medium entropy</strong>. The model shows moderate uncertainty."
        color = "#764D00"  
    else:
        comment = "This is <strong>high entropy</strong>. The model is highly uncertain and prediction may be unreliable."
        color = "#D20A0A"  

    st.markdown(f"""
    <div style="background-color:{color}; border-left: 10px solid #000000; padding:15px; border-radius:10px; margin-top:10px;">
        <p style="color:white; font-size:20px;">
        📊 <strong>Entropy Insight:</strong> {comment}
        </p>
    </div>
    """, unsafe_allow_html=True)

    return normalized_entropy


# ----------------------------------------
# Plot Nearest Neighbor Similarity
# ----------------------------------------
def plot_nearest_neighbor_similarity(raw_text, tfidf_vectorizer, train_df, top_n=10):
    """
    Compute nearest neighbors of the input text in the training set
    and show distribution of Fake vs Real among top N neighbors
    using a stacked bar chart, plus example text snippets.
    """

    # Transform input text
    text_vec = tfidf_vectorizer.transform([raw_text])
    train_vecs = tfidf_vectorizer.transform(train_df["full_content"].values)

    # Compute cosine similarity
    sims = cosine_similarity(text_vec, train_vecs)[0]

    # Attach similarity scores
    train_df = train_df.copy()
    train_df["Similarity"] = sims

    # Take top N neighbors
    top_neighbors = train_df.sort_values(by="Similarity", ascending=False).head(top_n)

    # Count Fake vs Real among top neighbors
    counts = top_neighbors["label"].value_counts().to_dict()
    fake_count = counts.get(1, 0)
    real_count = counts.get(0, 0)

    summary_df = pd.DataFrame({
        "Class": ["Fake ❌", "Real ✅"],
        "Count": [fake_count, real_count]
    })


    fig = px.bar(
        summary_df,
        x=["Top Neighbors"]*len(summary_df),
        y="Count",
        color="Class",
        text="Count",
        color_discrete_map={"Real ✅": "#23C552", "Fake ❌": "#FF4B4B"}
    )

    fig.update_traces(texttemplate='%{y}', textposition='inside', textfont_size=18)
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Count",
        xaxis=dict(title_font=dict(size=18), tickfont=dict(size=18)),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=18)),
        legend=dict(
            font=dict(size=18),
            title=dict(text="Class", font=dict(size=18))  
        ),
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    if fake_count > real_count:
        comment = f"The majority of the Top {top_n} neighbors are <strong>Fake ({fake_count} vs {real_count})</strong>. This supports a <strong>Fake prediction</strong>."
        color = "#D00000"
    elif real_count > fake_count:
        comment = f"The majority of the Top {top_n} neighbors are <strong>Real ({real_count} vs {fake_count})</strong>. This supports a <strong>Real prediction</strong>."
        color = "#019C2D"
    else:
        comment = f"The equal distribution of Fake and Real neighbors ({fake_count} vs {real_count}). Prediction reliability is <strong>weaker</strong>."
        color = "#BD7200"

    st.markdown(f"""
    <div style="background-color:{color}; border-left: 10px solid #000000; padding:15px; border-radius:10px; margin-top:10px;">
        <p style="color:white; font-size:20px;">
        🧭 <strong>Neighbor Insight:</strong> {comment}
        </p>
    </div>
    """, unsafe_allow_html=True)

 
    top_neighbors["Verdict"] = top_neighbors["label"].map({0: "Real ✅", 1: "Fake ❌"})
    top_neighbors["Snippet"] = top_neighbors["full_content"].str.slice(0, 120) + "..."

    display_df = top_neighbors.rename(columns={
        "Similarity": "Similarity Score",
        "Snippet": "Example Snippet"
    })


    st.markdown("---")
    st.subheader("🔎 Top Nearest Neighbor Examples")

    # Build table
    table_html = display_df[["Verdict", "Similarity Score", "Example Snippet"]].to_html(
        index=False,
        classes="custom-table",
        justify="center"
    )

    st.markdown("""
        <style>
            .custom-table {
                background-color: #023E8A;
                color: white;
                border-collapse: collapse;
                margin-left: auto;
                margin-right: auto;
                width: 95%;
            }
            .custom-table th, .custom-table td {
                text-align: center;
                padding: 10px;
                font-size: 20px; /* increased font size */
            }
            .custom-table th {
                background-color: #03045E;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(table_html, unsafe_allow_html=True)

    return summary_df


# -------------
#  Streamlit UI
# -------------
st.markdown("""
    <style>
    /* Radio Button Labels ("Paste text", etc.) */
    div[data-testid="stRadio"] label p {
        font-size: 20px !important;
    }
    
    /* Global Input Labels */
    .stWidgetLabel p {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    /* SPECIFIC target for the Text Area label to ensure it is 20px */
    div[data-testid="stTextArea"] label p {
        font-size: 20px !important;
    }

    /* Text Area Content (The text typed inside) */
    .stTextArea textarea {
        font-size: 20px !important;
    }

    /* File Uploader text */
    div[data-testid="stFileUploader"] section {
        font-size: 20px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Header 
st.markdown(
    """
    <div style="text-align: center; padding: 20px;">
        <h1 style='font-size: 70px; font-weight: 800; color: #55A9DC; text-shadow: 2px 2px 5px rgba(0,0,0,0.1); margin-bottom: 0px;'>
            📰 Fake News Detector 📰
        </h1>
        <hr style="border: 0; height: 3px; background: linear-gradient(to right, transparent, #6395EE, transparent); width: 60%; margin: auto;">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <p style='text-align: center; font-size: 20px; color: #FFFFFF; font-weight: 500;'>
        Paste article text or upload a <code>.txt</code> file to predict Fake or Real news.
    </p>
    """,
    unsafe_allow_html=True
)

input_option = st.radio("Input method:", ["Paste text", "Upload .txt file"])

if input_option == "Paste text":

    st.markdown("<p style='font-size: 20px; font-weight: 500;'>Paste your article text here:</p>", unsafe_allow_html=True)
    raw_text = st.text_area("", height=250)
    
elif input_option == "Upload .txt file":

    st.markdown("<p style='font-size: 20px; font-weight: 500;'>Upload a .txt file:</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="txt")
    if uploaded_file is not None:
        raw_text = uploaded_file.read().decode('utf-8')

        st.markdown("<p style='font-size: 20px; font-weight: 500;'>Content from uploaded file:</p>", unsafe_allow_html=True)
        st.text_area("", value=raw_text, height=250)
    else:
        raw_text = st.text_area("Paste article text here:", height=250)


# Prediction Button
if st.button("Predict") and raw_text.strip() != "":


    with st.spinner("Processing... Please Wait"):
        time.sleep(1.5)

    st.markdown("""
        <style>
        /* Center and wrap tabs */
        div[data-testid="stTabs"] {
            display: flex !important;
            justify-content: center !important;
            flex-wrap: wrap !important;
            gap: 10px;
            margin-top: 30px;
            margin-bottom: 50px;
        }

        /* Apply font size to all descendants of button */
        div[data-testid="stTabs"] button * {
            font-size: 20px !important;
            font-weight: bold !important;
        }

        div[data-testid="stTabs"] button {
            color: white !important;
            background-color: #cc10a6 !important;
            border-radius: 10px !important;
            padding: 10px 20px !important;
            border: none !important;
            transition: 0.3s;
        }

        div[data-testid="stTabs"] button:hover {
            background-color: #800066 !important;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            background-color: #cc1010 !important;
            color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)



    # ---------------
    # Navigation Tabs
    # ---------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 Prediction Outcome",
        "📝 Model Explanation",
        "🌍 News Context & Verification",
        "💡 Linguistic Analysis",
        "📈 Prediction Reliability"
    ])


    # Dataset proof
    truth = find_dataset_truth(raw_text)

    # Country detection
    detected_country = detect_country_dataset(raw_text)

    # Category similarity
    category_percent = detect_category_similarity_tfidf(raw_text)
    final_category_sim = max(category_percent, key=category_percent.get)

    # Decide which metadata to use
    used_country = truth["country"] if truth["found"] else detected_country
    used_category = truth["category"] if truth["found"] else final_category_sim

    # Preprocess text
    X_input, wc, sentiment, readability, avg_len = preprocess_text(
        raw_text,
        category=used_category,
        country=used_country
    )

    # Predict
    pred_label = model.predict(X_input)[0]
    pred_prob_fake = model.predict_proba(X_input)[0, 1]
    pred_prob_real = 1 - pred_prob_fake

    model_verdict = "Fake ❌" if pred_label == 1 else "Real ✅"
    color = "red" if pred_label == 1 else "#80EF80"


# --------------------------------
# News Accuracy (Model Confidence)
# --------------------------------
    if pred_label == 1:  # Fake
        accuracy_value = pred_prob_fake
    else:  # Real
        accuracy_value = 1 - pred_prob_fake

    accuracy_percent = accuracy_value * 100

    with tab1:
    # -----------------
    # Model Prediction
    # -----------------
        st.markdown("---")
        st.subheader("🔍 Model Prediction")
        st.markdown(
            f"<h2 style='color:{color}; font-size: 80px; text-align:center;'>{model_verdict}</h2>",
            unsafe_allow_html=True
        )


    # -----------------------
    # Fact Check Explorer URL
    # -----------------------
        st.markdown("---")
        fact_check_query = raw_text[:100].strip().replace(" ", "+")
        google_search_url = f"https://www.google.com/search?q={fact_check_query}+news"

        st.markdown("### 🛡️ External Verification")

        st.markdown("""
            <style>
            .google-btn {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
                height: 60px;
                background-color: #000000; /* Black Background */
                color: #ffffff !important; /* White Text */
                border: 2px solid #333333; 
                border-radius: 10px;
                

                text-decoration: none !important; 
                
                font-size: 20px !important;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            
            .google-btn:hover {
                background-color: #1a1a1a;
                border-color: #FF4B4B; /* Red border on hover */
                
                /* ENSURE NO UNDERLINE ON HOVER */
                text-decoration: none !important; 
                color: #ffffff !important;
            }

            .google-btn:active {
                border-color: #FF0000; /* Bright Red border when clicked */
                box-shadow: 0px 0px 15px #FF0000; /* Red glow */
                transform: scale(0.98);
                
                text-decoration: none !important; 
            }
            </style>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <a href="{google_search_url}" target="_blank" class="google-btn">
                🌐 Search Google News
            </a>
            """, unsafe_allow_html=True)

        st.markdown("---")


    # ---------------------------------
    # Breakdown of the Prediction News 
    # ---------------------------------
        st.subheader("📈 Breakdown of the Prediction")

        prob_fake = pred_prob_fake
        prob_real = 1 - pred_prob_fake

        # Display confidence percentages 
        st.markdown(f"""
        <div style="border: 2px solid silver; border-radius: 10px; padding: 20px; width: 100%; max-width: 700px; margin: 0 auto; background-color: #111; text-align: center;">
            <p style="color: white; font-size: 22px; font-weight: bold; margin: 10px 0;">Fake: <span style="color: #ff4c4c;">{round(prob_fake*100,2)}%</span></p>
            <p style="color: white; font-size: 22px; font-weight: bold; margin: 10px 0;">Real: <span style="color: #4caf50;">{round(prob_real*100,2)}%</span></p>
        </div>
        """, unsafe_allow_html=True)



    # -----------------------------------------------
    # Confidence Gauge / Meter for the News Accuracy
    # -----------------------------------------------
        st.markdown("<br><br>", unsafe_allow_html=True)

        confidence = prob_fake if pred_label == 1 else prob_real
        label_text = "Fake" if pred_label == 1 else "Real"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence*100,
            title={'text': f"The model is confident that this news is {label_text}"},
            
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if label_text=="Fake" else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence*100
                }
            }
        ))     
        st.plotly_chart(fig, use_container_width=True)


    with tab2:
    # -------------------------------------
    # Influential Words / Word Explanation
    # -------------------------------------
        st.markdown("---")
        st.subheader("🧩 Key Influential Words")

        # Extract influential words
        influential_phrases = extract_model_influential_phrases(
            raw_text, pred_label, top_n=10
        )

        if influential_phrases:
            df_phrases = pd.DataFrame(influential_phrases)

            table_html = df_phrases.to_html(
                index=False,
                classes="custom-table",
                justify="center"
            )

            st.markdown("""
                <style>
                    .custom-table {
                        background-color: #023E8A; /* table background red */
                        color: white;              /* text color */
                        border-collapse: collapse;
                        margin-left: auto;         /* center table */
                        margin-right: auto;
                        width: 80%;                /* table width */
                    }
                    .custom-table th, .custom-table td {
                        text-align: center;
                        padding: 10px;             /* padding */
                        font-size: 20px;           /* match chart font size */
                    }
                    .custom-table th {
                        background-color: #03045E; 
                    }
                </style>
            """, unsafe_allow_html=True)

            st.markdown(table_html, unsafe_allow_html=True)

            st.markdown("""
                <div style="background-color: #E75480; border-left: 10px solid #FFC0CB; padding: 20px; border-radius: 15px; margin-top: 15px;">
                    <p style="color: white; font-size: 20px; font-weight: bold; margin: 0;">
                        These words had the greatest impact on predicting whether the news is Fake or Real.
                    </p>
                </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
                <div style="background-color: #4A2233; border-left: 10px solid #CC8C9B; padding: 20px; border-radius: 15px; margin-top: 15px;">
                    <p style="color: white; font-size: 20px; font-weight: bold; margin: 0;">
                    ℹ️ No influential words detected.
                    </p>
                </div>
            """, unsafe_allow_html=True)


    with tab3:
    # -------------
    # Dataset Proof
    # -------------
        st.markdown("---")
        st.subheader("📜 Dataset Proof")

        if truth["found"]:

            st.markdown(f"""
                <div style="background-color: #d4edda; border-left: 8px solid #28a745; padding: 20px; border-radius: 10px;">
                    <p style="color: #155724; font-size: 20px; margin: 0;">
                        ✅ <strong>Match found in dataset</strong><br><br>
                        <strong>True Label:</strong> {truth['label']}<br>
                        <strong>True Country:</strong> {truth['country']}<br>
                        <strong>True Category:</strong> {truth['category']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background-color: #d1ecf1; border-left: 8px solid #17a2b8; padding: 20px; border-radius: 10px;">
                    <p style="color: #0c5460; font-size: 20px; margin: 0;">
                        ℹ️ <strong>No exact match found in the dataset.</strong> This appears to be new news.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
        st.markdown("---")

    # ------------------------------
    # Category Similarity Bar Chart 
    # ------------------------------
        st.subheader("🏷️ Category Similarity")
        final_category = plot_category_bar(category_percent)
        st.markdown("---")


    # ---------------------------
    # Country Detection with Map
    # ---------------------------
        st.subheader("🌍 Country Detection")

        plot_country_map(used_country)

        st.markdown(f"""
            <div style="
                background-color: #CC5500;  
                border-left: 10px solid #FFDAB3;  
                padding: 20px; 
                border-radius: 15px; 
                text-align: left;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            ">
                <p style="color: white; font-size: 20px; margin: 0;">
                    <span style="font-weight:bold;">Detected Country</span>: {used_country}
                </p>
            </div>
            """, unsafe_allow_html=True)
        

    with tab4:
    # ----------------------------
    # Word Count / Article Length
    # ----------------------------
        st.markdown("---")
        st.subheader("📊 Word Count / Article Length Analysis")
        plot_word_count_analysis(wc)  


    # -------------------------
    # Sentiment Gauge Analysis
    # -------------------------
        st.markdown("---")
        st.subheader("📘 Sentiment Score / Gauge Analysis")
        plot_sentiment_gauge(sentiment)


    # ---------------------------
    #  Readability Interpretation
    # ---------------------------
        st.markdown("---")
        st.subheader("⭐ Readability Score Interpretation (Education Level Match)")
        plot_readability_steps(readability)


    # --------------------------------------
    # Lexical Meter for Average Word Length
    # --------------------------------------
        st.markdown("---")
        st.subheader("📘 Average Word Length Analysis")
        plot_lexical_meter(avg_len)


    # ----------------------------------
    #  Top 10 Words Appeared in the News
    # ----------------------------------
        st.markdown("---")
        st.subheader("📝 Top 10 Most Frequent Words")
        most_frequent_words(raw_text)


    # ------------
    #  Word Cloud 
    # ------------
        st.markdown("---")
        st.subheader("☁️ Word Cloud")
        generate_wordcloud(raw_text)


    with tab5:   
    # ----------------------------
    # Entropy / Uncertainty Score
    # ----------------------------
        st.markdown("---")
        st.subheader("📊 Entropy / Uncertainty Score") 
        entropy_value = plot_entropy_bar(pred_prob_fake, pred_prob_real)

    
    # ----------------------------
    # Nearest Neighbor Similarity
    # ----------------------------
        st.markdown("---")
        st.subheader("🧭 Nearest Neighbor Similarity")
        neighbors = plot_nearest_neighbor_similarity(raw_text, tfidf_vectorizer, train_df, top_n=10)


else:

# -----------------------------------
# Display a Waiting for Input Message
# -----------------------------------
    st.markdown("""
        <div style="
            background-color: #000080; 
            border-left: 10px solid #6395EE; 
            padding: 30px; 
            border-radius: 15px; 
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        ">
            <p style="color: white; font-size: 20spx; margin: 0; font-weight: bold;">
                ⏳ Awaiting Input to Begin Analysis...
            </p>
        </div>
    """, unsafe_allow_html=True)




