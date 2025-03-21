import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta
from nlp_utils import nlp_processor
from db_utils import db

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsAnalytics:
    @staticmethod
    def create_sentiment_chart(articles: List[Dict]) -> go.Figure:
        """Create sentiment distribution pie chart"""
        try:
            sentiments = [art.get('sentiment', 'neutral') for art in articles]
            sentiment_counts = pd.Series(sentiments).value_counts()
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Article Sentiment Distribution",
                color_discrete_map={
                    'positive': '#2ecc71',
                    'neutral': '#95a5a6',
                    'negative': '#e74c3c'
                }
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            return fig
        except Exception as e:
            logger.error(f"Error creating sentiment chart: {str(e)}")
            return None

    @staticmethod
    def create_source_distribution(articles: List[Dict]) -> go.Figure:
        """Create news source distribution bar chart"""
        try:
            sources = [art.get('source', {}).get('name', 'Unknown') for art in articles]
            source_counts = pd.Series(sources).value_counts().head(10)
            
            fig = px.bar(
                x=source_counts.index,
                y=source_counts.values,
                title="Top News Sources",
                labels={'x': 'Source', 'y': 'Number of Articles'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            return fig
        except Exception as e:
            logger.error(f"Error creating source distribution: {str(e)}")
            return None

    @staticmethod
    def create_category_distribution(articles: List[Dict]) -> go.Figure:
        """Create category distribution chart"""
        try:
            categories = [art.get('category', 'Uncategorized') for art in articles]
            category_counts = pd.Series(categories).value_counts()
            
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Article Category Distribution"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            return fig
        except Exception as e:
            logger.error(f"Error creating category distribution: {str(e)}")
            return None

    @staticmethod
    def generate_wordcloud(articles: List[Dict]) -> Optional[WordCloud]:
        """Generate word cloud from article texts"""
        try:
            # Combine all article texts
            all_text = " ".join(
                f"{art.get('title', '')} {art.get('description', '')}" 
                for art in articles
            )
            
            if not all_text.strip():
                return None
                
            # Generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                collocations=False
            ).generate(all_text)
            
            return wordcloud
        except Exception as e:
            logger.error(f"Error generating word cloud: {str(e)}")
            return None

    @staticmethod
    def create_trending_topics(articles: List[Dict], top_n: int = 10) -> go.Figure:
        """Create trending topics bar chart"""
        try:
            # Extract keywords from all articles
            all_keywords = []
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                keywords = nlp_processor.extract_keywords(text)
                all_keywords.extend([kw for kw, _ in keywords])
            
            # Count keyword frequencies
            keyword_counts = Counter(all_keywords).most_common(top_n)
            
            if not keyword_counts:
                return None
            
            # Create bar chart
            fig = px.bar(
                x=[count for _, count in keyword_counts],
                y=[word for word, _ in keyword_counts],
                orientation='h',
                title="Trending Topics",
                labels={'x': 'Frequency', 'y': 'Topic'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            return fig
        except Exception as e:
            logger.error(f"Error creating trending topics: {str(e)}")
            return None

    @staticmethod
    def create_publication_timeline(articles: List[Dict]) -> go.Figure:
        """Create publication timeline chart"""
        try:
            # Extract publication dates
            dates = [
                datetime.strptime(art.get('publishedAt', ''), '%Y-%m-%dT%H:%M:%SZ')
                for art in articles
                if art.get('publishedAt')
            ]
            
            date_counts = pd.Series(dates).value_counts().sort_index()
            
            fig = px.line(
                x=date_counts.index,
                y=date_counts.values,
                title="Publication Timeline",
                labels={'x': 'Date', 'y': 'Number of Articles'}
            )
            return fig
        except Exception as e:
            logger.error(f"Error creating publication timeline: {str(e)}")
            return None

    @staticmethod
    def create_user_reading_patterns(user_id: int) -> Dict[str, go.Figure]:
        """Create visualizations for user reading patterns"""
        try:
            # Get user's reading history
            history = db.get_reading_history(user_id)
            if not history:
                return {}
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Reading time distribution
            time_dist = df['timestamp'].dt.hour.value_counts().sort_index()
            time_chart = px.bar(
                x=time_dist.index,
                y=time_dist.values,
                title="Reading Time Distribution",
                labels={'x': 'Hour of Day', 'y': 'Number of Articles Read'}
            )
            
            # Category preferences
            cat_dist = df['category'].value_counts()
            category_chart = px.pie(
                values=cat_dist.values,
                names=cat_dist.index,
                title="Category Preferences"
            )
            
            # Reading trend over time
            daily_counts = df.groupby(df['timestamp'].dt.date).size()
            trend_chart = px.line(
                x=daily_counts.index,
                y=daily_counts.values,
                title="Reading Trend",
                labels={'x': 'Date', 'y': 'Articles Read'}
            )
            
            return {
                'time_distribution': time_chart,
                'category_preferences': category_chart,
                'reading_trend': trend_chart
            }
        except Exception as e:
            logger.error(f"Error creating user reading patterns: {str(e)}")
            return {}

    @staticmethod
    def create_feedback_analysis(user_id: int) -> Dict[str, go.Figure]:
        """Analyze user feedback patterns"""
        try:
            conn = db.get_connection()
            df = pd.read_sql_query("""
                SELECT feedback, rating, timestamp 
                FROM article_feedback 
                WHERE user_id = ?
            """, conn, params=(user_id,))
            conn.close()
            
            if df.empty:
                return {}
            
            # Feedback distribution
            feedback_dist = df['feedback'].value_counts()
            feedback_chart = px.pie(
                values=feedback_dist.values,
                names=feedback_dist.index,
                title="Feedback Distribution"
            )
            
            # Rating distribution
            rating_dist = df['rating'].value_counts().sort_index()
            rating_chart = px.bar(
                x=rating_dist.index,
                y=rating_dist.values,
                title="Rating Distribution",
                labels={'x': 'Rating', 'y': 'Count'}
            )
            
            return {
                'feedback_distribution': feedback_chart,
                'rating_distribution': rating_chart
            }
        except Exception as e:
            logger.error(f"Error creating feedback analysis: {str(e)}")
            return {}

# Initialize global analytics instance
analytics = NewsAnalytics()