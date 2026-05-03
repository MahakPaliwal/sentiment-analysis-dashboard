# Sentiment Analysis Dashboard

An interactive 3-class sentiment analysis web app built with Machine Learning and Streamlit.

🚀 **Live Demo:** [Click here](https://sentiment-analysis-dashboard-9insq5xcndbpxunfv7dm7b.streamlit.app)

---

## What it does
- Classifies any text into **Positive 😊**, **Negative 😔**, or **Neutral 😐**
- Shows **confidence score** and **probability breakdown** for each prediction
- Supports **bulk analysis** — analyze multiple texts at once
- Displays **sentiment distribution** pie chart for bulk inputs

---

## Model Details
| Detail | Value |
|---|---|
| Algorithm | Logistic Regression |
| Vectorization | TF-IDF (50k features, bigrams) |
| Training Dataset | Twitter Entity Sentiment Analysis (Kaggle) |
| Training Samples | ~49,000 tweets |
| Test Accuracy | **90.50%** |
| Classes | Positive, Negative, Neutral |
| Negative F1-score | 0.92 |
| Neutral F1-score | 0.89 |
| Positive F1-score | 0.91 |

---

## Tech Stack
- **Python** — core language
- **Scikit-learn** — TF-IDF vectorization + Logistic Regression
- **Streamlit** — interactive web dashboard
- **Plotly** — probability bar charts and pie charts
- **Pandas** — data processing
- **NLTK** — text preprocessing

---

## Project Structure
sentiment-analysis-dashboard/
├── app.py                  # Streamlit dashboard
├── sentiment_model.pkl     # Trained Logistic Regression model
├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
---

## How to Run Locally

**Step 1 — Clone the repo**
```bash
git clone https://github.com/MahakPaliwal/sentiment-analysis-dashboard.git
cd sentiment-analysis-dashboard
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Run the app**
```bash
streamlit run app.py
```
**Step 4 — Open in browser**
---

## Screenshots

### Single Text Analysis
Upload any text and get instant sentiment prediction with confidence score and probability breakdown.

### Bulk Analysis
Analyze multiple texts at once and see overall sentiment distribution.

---

## Sample Predictions
| Text | Sentiment | Confidence |
|---|---|---|
| "I love this product!" | 😊 Positive | 94.2% |
| "This is the worst experience" | 😔 Negative | 91.5% |
| "The meeting was cancelled" | 😐 Neutral | 87.3% |
| "Great service, highly recommend!" | 😊 Positive | 96.1% |

---

## Key Highlights
- **90.50% accuracy** on 3-class classification
- Balanced F1-scores across all 3 sentiment classes
- Comprehensive EDA — word frequency, entity analysis, tweet length distribution
- Interactive visualizations using Plotly
- Deployed as a live web application

---

## Built by
**Mahak Paliwal**
- 📧 mahakpaliwal58@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/mahak-paliwal-a739841a1)
- 💻 [GitHub](https://github.com/MahakPaliwal)
