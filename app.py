import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="AeroStream Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    .sentiment-positive {
        color: #2ecc71;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #e74c3c;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #f39c12;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and encoder
@st.cache_resource
def load_models():
    """Load the trained model, label encoder, and embedding model"""
    try:
        model = joblib.load('models/airline_sentiment_lr.pkl')
        encoder = joblib.load('models/label_encoder.pkl')
        embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
        return model, encoder, embedding_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# Text preprocessing
def preprocess_text(text):
    """Clean and preprocess the input text"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.strip()
    return text

# Sentiment prediction
def predict_sentiment(text, model, encoder, embedding_model):
    """Predict sentiment for the given text"""
    try:
        # Preprocess
        cleaned_text = preprocess_text(text)
        
        # Generate embedding
        embedding = embedding_model.encode([cleaned_text], normalize_embeddings=True)
        
        # Predict
        prediction = model.predict(embedding)[0]
        probabilities = model.predict_proba(embedding)[0]
        
        # Decode label
        sentiment = encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary
        prob_dict = {
            encoder.classes_[i]: float(probabilities[i]) 
            for i in range(len(encoder.classes_))
        }
        
        return sentiment, prob_dict, cleaned_text
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

# Main app
def main():
    # Header
    st.title("✈️ AeroStream Analytics")
    st.markdown("### AI-Powered Airline Sentiment Analysis")
    st.markdown("---")
    
    # Load models
    model, encoder, embedding_model = load_models()
    
    if model is None or encoder is None or embedding_model is None:
        st.error("⚠️ Failed to load models. Please ensure model files exist in the 'models' folder.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/airplane-take-off.png", width=80)
        st.title("About")
        st.info(
            """
            **AeroStream Analytics** uses advanced machine learning to analyze 
            airline customer feedback and predict sentiment in real-time.
            
            **Features:**
            - 🎯 Real-time sentiment prediction
            - 📊 Confidence scores
            - 📈 Interactive visualizations
            - 🔄 Batch processing
            """
        )
        
        st.markdown("---")
        st.markdown("**Model Info:**")
        st.markdown("- Algorithm: Logistic Regression")
        st.markdown("- Embeddings: all-MiniLM-L12-v2")
        st.markdown("- Classes: Negative, Neutral, Positive")
        
        st.markdown("---")
        st.markdown("**Statistics:**")
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        st.metric("Total Predictions", len(st.session_state.predictions))
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📋 Batch Analysis", "📊 Analytics"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Analyze a Single Tweet")
        
        # Sample tweets
        sample_tweets = {
            "Positive": "@united Thanks for the great service! Flight was smooth and crew was amazing! 😊",
            "Negative": "@AmericanAir Worst experience ever. 3 hour delay and lost my luggage. Unacceptable!",
            "Neutral": "@SouthwestAir Flight from LAX to JFK departing at 2pm"
        }
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Text input
            user_input = st.text_area(
                "Enter airline tweet or feedback:",
                height=120,
                placeholder="Type or paste airline-related text here...",
                help="Enter any airline customer feedback, tweet, or review"
            )
        
        with col2:
            st.markdown("**Try examples:**")
            for sentiment, tweet in sample_tweets.items():
                if st.button(f"📌 {sentiment}", key=f"sample_{sentiment}"):
                    user_input = tweet
                    st.rerun()
        
        # Predict button
        if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
            if user_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    sentiment, probabilities, cleaned = predict_sentiment(
                        user_input, model, encoder, embedding_model
                    )
                    
                    if sentiment:
                        # Store prediction
                        st.session_state.predictions.append({
                            'text': user_input,
                            'sentiment': sentiment,
                            'timestamp': datetime.now(),
                            'probabilities': probabilities
                        })
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Analysis Results")
                        
                        # Sentiment badge
                        sentiment_colors = {
                            'positive': '🟢',
                            'negative': '🔴',
                            'neutral': '🟡'
                        }
                        sentiment_classes = {
                            'positive': 'sentiment-positive',
                            'negative': 'sentiment-negative',
                            'neutral': 'sentiment-neutral'
                        }
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Predicted Sentiment:**")
                            st.markdown(
                                f"<h2 class='{sentiment_classes[sentiment]}'>"
                                f"{sentiment_colors[sentiment]} {sentiment.upper()}</h2>",
                                unsafe_allow_html=True
                            )
                        
                        with col2:
                            st.markdown("**Confidence:**")
                            confidence = probabilities[sentiment] * 100
                            st.markdown(f"<h2>{confidence:.1f}%</h2>", unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("**Text Length:**")
                            st.markdown(f"<h2>{len(user_input)} chars</h2>", unsafe_allow_html=True)
                        
                        # Probability bars
                        st.markdown("---")
                        st.subheader("📈 Confidence Scores")
                        
                        # Create DataFrame for visualization
                        prob_df = pd.DataFrame({
                            'Sentiment': list(probabilities.keys()),
                            'Probability': [p * 100 for p in probabilities.values()]
                        }).sort_values('Probability', ascending=False)
                        
                        # Horizontal bar chart
                        fig = px.bar(
                            prob_df,
                            x='Probability',
                            y='Sentiment',
                            orientation='h',
                            color='Sentiment',
                            color_discrete_map={
                                'positive': '#2ecc71',
                                'negative': '#e74c3c',
                                'neutral': '#f39c12'
                            },
                            text=prob_df['Probability'].apply(lambda x: f'{x:.1f}%')
                        )
                        fig.update_layout(
                            showlegend=False,
                            height=250,
                            xaxis_title="Confidence (%)",
                            yaxis_title="",
                            xaxis_range=[0, 100]
                        )
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show cleaned text
                        with st.expander("🔍 View Preprocessed Text"):
                            st.code(cleaned, language=None)
                            st.caption("This is the cleaned version of your input used for prediction")
            else:
                st.warning("⚠️ Please enter some text to analyze")
    
    # Tab 2: Batch Analysis
    with tab2:
        st.header("Batch Tweet Analysis")
        st.markdown("Analyze multiple tweets at once by entering them line by line.")
        
        batch_input = st.text_area(
            "Enter multiple tweets (one per line):",
            height=200,
            placeholder="Tweet 1\nTweet 2\nTweet 3...",
            help="Each line will be analyzed separately"
        )
        
        if st.button("🚀 Analyze Batch", type="primary"):
            if batch_input.strip():
                tweets = [t.strip() for t in batch_input.split('\n') if t.strip()]
                
                if tweets:
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, tweet in enumerate(tweets):
                        status_text.text(f"Analyzing tweet {i+1}/{len(tweets)}...")
                        sentiment, probabilities, cleaned = predict_sentiment(
                            tweet, model, encoder, embedding_model
                        )
                        
                        if sentiment:
                            results.append({
                                'Tweet': tweet[:100] + '...' if len(tweet) > 100 else tweet,
                                'Sentiment': sentiment.upper(),
                                'Confidence': f"{probabilities[sentiment]*100:.1f}%",
                                'Negative': f"{probabilities.get('negative', 0)*100:.1f}%",
                                'Neutral': f"{probabilities.get('neutral', 0)*100:.1f}%",
                                'Positive': f"{probabilities.get('positive', 0)*100:.1f}%"
                            })
                        
                        progress_bar.progress((i + 1) / len(tweets))
                    
                    status_text.text("Analysis complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Batch Results")
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True, height=400)
                    
                    # Summary statistics
                    st.markdown("---")
                    st.subheader("📈 Summary Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    sentiment_counts = results_df['Sentiment'].value_counts()
                    
                    with col1:
                        st.metric("Total Tweets", len(results))
                    with col2:
                        st.metric("Positive", sentiment_counts.get('POSITIVE', 0), 
                                 delta=f"{sentiment_counts.get('POSITIVE', 0)/len(results)*100:.1f}%")
                    with col3:
                        st.metric("Neutral", sentiment_counts.get('NEUTRAL', 0),
                                 delta=f"{sentiment_counts.get('NEUTRAL', 0)/len(results)*100:.1f}%")
                    with col4:
                        st.metric("Negative", sentiment_counts.get('NEGATIVE', 0),
                                 delta=f"{sentiment_counts.get('NEGATIVE', 0)/len(results)*100:.1f}%")
                    
                    # Pie chart
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'POSITIVE': '#2ecc71',
                            'NEGATIVE': '#e74c3c',
                            'NEUTRAL': '#f39c12'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("⚠️ Please enter at least one tweet to analyze")
    
    # Tab 3: Analytics
    with tab3:
        st.header("📊 Session Analytics")
        
        if st.session_state.predictions:
            predictions = st.session_state.predictions
            
            # Create DataFrame
            df = pd.DataFrame([
                {
                    'Text': p['text'][:80] + '...' if len(p['text']) > 80 else p['text'],
                    'Sentiment': p['sentiment'],
                    'Confidence': p['probabilities'][p['sentiment']],
                    'Timestamp': p['timestamp']
                }
                for p in predictions
            ])
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            sentiment_counts = df['Sentiment'].value_counts()
            avg_confidence = df['Confidence'].mean()
            
            with col1:
                st.metric("Total Predictions", len(df))
            with col2:
                st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
            with col3:
                most_common = sentiment_counts.index[0] if len(sentiment_counts) > 0 else 'N/A'
                st.metric("Most Common", most_common.upper())
            with col4:
                st.metric("Unique Texts", df['Text'].nunique())
            
            st.markdown("---")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution
                fig1 = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'positive': '#2ecc71',
                        'negative': '#e74c3c',
                        'neutral': '#f39c12'
                    }
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Confidence distribution
                fig2 = px.histogram(
                    df,
                    x='Confidence',
                    nbins=20,
                    title="Confidence Score Distribution",
                    labels={'Confidence': 'Confidence Score'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Timeline
            if len(df) > 1:
                st.markdown("---")
                st.subheader("📈 Prediction Timeline")
                
                df_timeline = df.copy()
                df_timeline['Timestamp'] = pd.to_datetime(df_timeline['Timestamp'])
                df_timeline = df_timeline.sort_values('Timestamp')
                df_timeline['Index'] = range(1, len(df_timeline) + 1)
                
                fig3 = px.scatter(
                    df_timeline,
                    x='Index',
                    y='Confidence',
                    color='Sentiment',
                    title="Confidence Scores Over Time",
                    labels={'Index': 'Prediction Number', 'Confidence': 'Confidence Score'},
                    color_discrete_map={
                        'positive': '#2ecc71',
                        'negative': '#e74c3c',
                        'neutral': '#f39c12'
                    },
                    hover_data=['Text']
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            # Recent predictions table
            st.markdown("---")
            st.subheader("📝 Recent Predictions")
            st.dataframe(
                df[['Text', 'Sentiment', 'Confidence', 'Timestamp']].head(10),
                use_container_width=True
            )
            
            # Clear history button
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.predictions = []
                st.rerun()
        else:
            st.info("📭 No predictions yet. Start analyzing tweets in the other tabs!")
            st.markdown("""
                **Get started by:**
                1. Go to the "Single Prediction" tab
                2. Enter an airline tweet or feedback
                3. Click "Analyze Sentiment"
                4. Come back here to see your analytics!
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; padding: 2rem 0;'>
            <p>✈️ AeroStream Analytics | Powered by Machine Learning</p>
            <p>Built with Streamlit, Scikit-learn & Sentence-Transformers</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
