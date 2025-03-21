import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from config import Config

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_name: str = None):
        self.db_name = db_name or Config.DB_NAME

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection"""
        return sqlite3.connect(self.db_name)

    def execute_query(self, query: str, params: tuple = None, fetch: bool = False) -> Optional[List]:
        """Execute SQL query with error handling"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            if fetch:
                result = cursor.fetchall()
            else:
                result = None
                
            conn.commit()
            return result
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    # User Management
    def create_user(self, username: str, password: str) -> Optional[int]:
        """Create new user and return user ID"""
        try:
            query = "INSERT INTO users (username, password) VALUES (?, ?)"
            self.execute_query(query, (username, password))
            
            # Get the created user's ID
            query = "SELECT id FROM users WHERE username = ?"
            result = self.execute_query(query, (username,), fetch=True)
            return result[0][0] if result else None
            
        except sqlite3.IntegrityError:
            logger.error(f"Username {username} already exists")
            return None

    def get_user(self, username: str) -> Optional[Dict]:
        """Get user details by username"""
        query = "SELECT id, username, created_at FROM users WHERE username = ?"
        result = self.execute_query(query, (username,), fetch=True)
        
        if result:
            return {
                'id': result[0][0],
                'username': result[0][1],
                'created_at': result[0][2]
            }
        return None

    def verify_user(self, username: str, password: str) -> bool:
        """Verify user credentials"""
        query = "SELECT id FROM users WHERE username = ? AND password = ?"
        result = self.execute_query(query, (username, password), fetch=True)
        return bool(result)

    # User Preferences
    def get_user_preferences(self, user_id: int) -> Optional[Dict]:
        """Get user preferences"""
        query = """
            SELECT categories, countries, language, sentiment 
            FROM user_preferences 
            WHERE user_id = ?
        """
        result = self.execute_query(query, (user_id,), fetch=True)
        
        if result:
            return {
                'categories': result[0][0].split(',') if result[0][0] else [],
                'countries': result[0][1].split(',') if result[0][1] else [],
                'language': result[0][2],
                'sentiment': result[0][3]
            }
        return None

    def update_user_preferences(self, user_id: int, preferences: Dict) -> bool:
        """Update user preferences"""
        query = """
            UPDATE user_preferences 
            SET categories = ?, 
                countries = ?, 
                language = ?, 
                sentiment = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
        """
        try:
            self.execute_query(query, (
                ','.join(preferences['categories']),
                ','.join(preferences['countries']),
                preferences['language'],
                preferences['sentiment'],
                user_id
            ))
            return True
        except Exception as e:
            logger.error(f"Error updating preferences: {str(e)}")
            return False

    # Reading History
    def add_reading_history(self, user_id: int, article_data: Dict) -> bool:
        """Record article reading history"""
        query = """
            INSERT INTO reading_history 
            (user_id, article_url, title, category) 
            VALUES (?, ?, ?, ?)
        """
        try:
            self.execute_query(query, (
                user_id,
                article_data['url'],
                article_data.get('title'),
                article_data.get('category')
            ))
            return True
        except Exception as e:
            logger.error(f"Error recording reading history: {str(e)}")
            return False

    def get_reading_history(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Get user's reading history"""
        query = """
            SELECT article_url, title, category, read_timestamp 
            FROM reading_history 
            WHERE user_id = ? 
            ORDER BY read_timestamp DESC 
            LIMIT ?
        """
        results = self.execute_query(query, (user_id, limit), fetch=True)
        
        return [{
            'url': row[0],
            'title': row[1],
            'category': row[2],
            'timestamp': row[3]
        } for row in results] if results else []

    # Article Feedback
    def add_article_feedback(self, user_id: int, article_url: str, 
                           feedback: str = None, rating: int = None) -> bool:
        """Record article feedback"""
        query = """
            INSERT INTO article_feedback 
            (user_id, article_url, feedback, rating) 
            VALUES (?, ?, ?, ?)
        """
        try:
            self.execute_query(query, (user_id, article_url, feedback, rating))
            return True
        except Exception as e:
            logger.error(f"Error recording feedback: {str(e)}")
            return False

    def get_article_feedback(self, article_url: str) -> Dict[str, Any]:
        """Get aggregated feedback for an article"""
        query = """
            SELECT 
                COUNT(*) as total_feedback,
                AVG(rating) as avg_rating,
                SUM(CASE WHEN feedback = 'Helpful' THEN 1 ELSE 0 END) as helpful_count,
                SUM(CASE WHEN feedback = 'Not Helpful' THEN 1 ELSE 0 END) as not_helpful_count
            FROM article_feedback 
            WHERE article_url = ?
        """
        result = self.execute_query(query, (article_url,), fetch=True)
        
        if result:
            return {
                'total_feedback': result[0][0],
                'average_rating': float(result[0][1]) if result[0][1] else None,
                'helpful_count': result[0][2],
                'not_helpful_count': result[0][3]
            }
        return {
            'total_feedback': 0,
            'average_rating': None,
            'helpful_count': 0,
            'not_helpful_count': 0
        }

    # Cache Management
    def cache_article(self, url: str, data: Dict) -> bool:
        """Cache article data"""
        query = """
            INSERT OR REPLACE INTO article_cache 
            (url, title, content, summary, sentiment) 
            VALUES (?, ?, ?, ?, ?)
        """
        try:
            self.execute_query(query, (
                url,
                data.get('title'),
                data.get('content'),
                data.get('summary'),
                data.get('sentiment')
            ))
            return True
        except Exception as e:
            logger.error(f"Error caching article: {str(e)}")
            return False

    def get_cached_article(self, url: str) -> Optional[Dict]:
        """Get cached article data"""
        query = """
            SELECT title, content, summary, sentiment, cached_at 
            FROM article_cache 
            WHERE url = ? AND cached_at > datetime('now', '-24 hours')
        """
        result = self.execute_query(query, (url,), fetch=True)
        
        if result:
            return {
                'title': result[0][0],
                'content': result[0][1],
                'summary': result[0][2],
                'sentiment': result[0][3],
                'cached_at': result[0][4]
            }
        return None

    def cleanup_old_cache(self) -> None:
        """Remove cached articles older than 24 hours"""
        query = "DELETE FROM article_cache WHERE cached_at < datetime('now', '-24 hours')"
        self.execute_query(query)

# Initialize global database manager
db = DatabaseManager()