import streamlit as st
import pickle
import re
import string
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="wide"
)

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()

# Clean text function

import re
import string

def handle_negation(text):
    negations = {
        "not good": "bad",
        "not happy": "sad",
        "not great": "terrible",
        "not having a good": "having a bad",
        "not like": "dislike",
        "not love": "hate",
        "not enjoy": "dislike",
        "not well": "unwell",
        "not nice": "unpleasant",
        "not fine": "bad",
        "isn't good": "bad",
        "wasn't good": "bad",
        "can't stand": "hate",
        "don't like": "dislike",
        "doesn't work": "broken",
        "didn't enjoy": "disliked"
    }
    text = text.lower()
    for phrase, replacement in negations.items():
        text = text.replace(phrase, replacement)
    return text

def clean_text(text):
    text = str(text)
    text = handle_negation(text)      # ← negation handled first
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip().lower()
    return text

# Test it
print(clean_text("I am not having a good day."))
# Should print: "i am having a bad day"

# Predict sentiment
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    classes = model.classes_
    return prediction, dict(zip(classes, probabilities))

# Color mapping
COLORS = {
    'Positive': '#6BCB77',
    'Negative': '#FF6B6B',
    'Neutral': '#FFD93D'
}
EMOJI = {
    'Positive': '😊',
    'Negative': '😔',
    'Neutral': '😐'
}

# Header
st.title("💬 Sentiment Analysis Dashboard")
st.caption("Analyze sentiment of text using ML — Positive, Negative or Neutral")
st.divider()

# Tabs
tab1, tab2, tab3 = st.tabs([
    "Single Text Analysis",
    "Bulk Analysis",
    "About"
])

# ─── Tab 1: Single Text ───
with tab1:
    st.subheader("Analyze a Single Text")
    user_input = st.text_area(
        "Enter your text here:",
        placeholder="Type something like: I love this product!",
        height=120
    )

    if st.button("Analyze", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text!")
        else:
            prediction, probabilities = predict_sentiment(user_input)

            # Result
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### Result")
                st.markdown(
                    f"<div style='background:{COLORS[prediction]};"
                    f"padding:20px;border-radius:10px;text-align:center;"
                    f"font-size:24px;font-weight:bold;color:white;'>"
                    f"{EMOJI[prediction]} {prediction}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Confidence:** {max(probabilities.values())*100:.1f}%")

            with col2:
                st.markdown("### Probability Breakdown")
                fig = go.Figure(go.Bar(
                    x=list(probabilities.keys()),
                    y=[v*100 for v in probabilities.values()],
                    marker_color=[COLORS[k] for k in probabilities.keys()],
                    text=[f"{v*100:.1f}%" for v in probabilities.values()],
                    textposition='auto'
                ))
                fig.update_layout(
                    yaxis_title="Probability (%)",
                    height=300,
                    margin=dict(t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

# ─── Tab 2: Bulk Analysis ───
with tab2:
    st.subheader("Analyze Multiple Texts at Once")
    st.caption("Enter one text per line")

    bulk_input = st.text_area(
        "Enter multiple texts (one per line):",
        placeholder="I love this!\nThis is terrible\nThe weather is okay",
        height=200,
        key="bulk_input"
    )

    # Live counter
    texts_entered = [t for t in bulk_input.split('\n') if t.strip()]
    count = len(texts_entered)

    if count == 0:
        st.caption("Texts entered: 0 / 100")
    elif count <= 80:
        st.markdown(f"✅ Texts entered: **{count} / 100**")
    elif count <= 100:
        st.markdown(f"⚠️ Texts entered: **{count} / 100** — approaching limit!")
    else:
        st.markdown(f"🔴 Texts entered: **{count} / 100** — limit exceeded!")

    if st.button("Analyze All", type="primary"):
        texts = [t.strip() for t in bulk_input.split('\n') if t.strip()]

        if len(texts) > 100:
            st.warning("Maximum 100 texts allowed. Please reduce your input.")
        elif len(texts) == 0:
            st.warning("Please enter some text!")
        else:
            results = []
            for text in texts:
                prediction, probabilities = predict_sentiment(text)
                results.append({
                    'Text': text,
                    'Sentiment': f"{EMOJI[prediction]} {prediction}",
                    'Confidence': f"{max(probabilities.values())*100:.1f}%",
                    'Positive %': f"{probabilities.get('Positive', 0)*100:.1f}%",
                    'Negative %': f"{probabilities.get('Negative', 0)*100:.1f}%",
                    'Neutral %': f"{probabilities.get('Neutral', 0)*100:.1f}%"
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            st.markdown("### Summary")
            sentiment_counts = pd.DataFrame(results)['Sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color_discrete_sequence=['#6BCB77', '#FF6B6B', '#FFD93D'],
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
# ─── Tab 3: About ───
with tab3:
    st.subheader("About this Project")
    st.markdown("""
    ### Sentiment Analysis Dashboard
    
    This project classifies text into **3 sentiment categories**:
    - 😊 **Positive** — Happy, satisfied, excited text
    - 😔 **Negative** — Sad, angry, disappointed text  
    - 😐 **Neutral** — Factual or objective text
    
    ### Model Details
    | Detail | Value |
    |---|---|
    | Algorithm | Logistic Regression |
    | Vectorization | TF-IDF (50k features, bigrams) |
    | Training Data | Twitter Entity Sentiment Dataset |
    | Accuracy | 90.51% |
    | Classes | Positive, Negative, Neutral |
    
    ### Tech Stack
    Python | Scikit-learn | TF-IDF | Streamlit | Plotly | Pandas
    
    ### Built by
    Mahak Paliwal
    """)