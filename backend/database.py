"""
Database module for storing user data, interview sessions, and chat history.
Uses PostgreSQL for scalable, cloud-based database storage with UTC timestamps.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone
from typing import List, Dict, Optional
import hashlib
import secrets
import os

class InterviewDatabase:
    def __init__(self, database_url: str = None):
        """Initialize database connection and create tables if they don't exist."""
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is not set. Please add it to your .env file.")
        self.init_database()

    def get_connection(self):
        """Create and return a database connection."""
        return psycopg2.connect(self.database_url)

    def init_database(self):
        """Create database tables if they don't exist."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                salt TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'UTC')
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interview_sessions (
                session_id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                interview_type TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                num_questions INTEGER NOT NULL,
                started_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'UTC'),
                completed_at TIMESTAMP WITH TIME ZONE,
                status TEXT DEFAULT 'in_progress',
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id SERIAL PRIMARY KEY,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'UTC'),
                FOREIGN KEY (session_id) REFERENCES interview_sessions(session_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                resume_id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'UTC'),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()

    # ─── User Management ───────────────────────────────────────────────────

    def _hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt. Returns (hash, salt)."""
        if salt is None:
            salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000
        ).hex()
        return password_hash, salt

    def create_user(self, username: str, password: str) -> tuple:
        """Create a new user. Returns (user_id, success, message)."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute("SELECT user_id, password_hash FROM users WHERE username = %s", (username,))
            existing_user = cursor.fetchone()
            if existing_user:
                if existing_user['password_hash'] is None:
                    password_hash, salt = self._hash_password(password)
                    cursor.execute(
                        "UPDATE users SET password_hash = %s, salt = %s WHERE user_id = %s",
                        (password_hash, salt, existing_user['user_id'])
                    )
                    conn.commit()
                    return existing_user['user_id'], True, "Password set successfully for existing account"
                return None, False, "Username already exists"
            password_hash, salt = self._hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, password_hash, salt) VALUES (%s, %s, %s) RETURNING user_id",
                (username, password_hash, salt)
            )
            user_id = cursor.fetchone()['user_id']
            conn.commit()
            return user_id, True, "User created successfully"
        except Exception as e:
            return None, False, f"Error creating user: {str(e)}"
        finally:
            cursor.close()
            conn.close()

    def authenticate_user(self, username: str, password: str) -> tuple:
        """Authenticate user. Returns (user_id, success, message)."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT user_id, password_hash, salt FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if not result:
            return None, False, "Username not found"
        if result['password_hash'] is None or result['salt'] is None:
            return None, False, "Account needs password setup. Please use Sign Up."
        password_hash, _ = self._hash_password(password, result['salt'])
        if password_hash == result['password_hash']:
            return result['user_id'], True, "Login successful"
        return None, False, "Incorrect password"

    def get_user_id(self, username: str) -> Optional[int]:
        """Get user_id for a given username."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result['user_id'] if result else None

    # ─── Profile Management ────────────────────────────────────────────────

    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """Get user profile info."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            "SELECT user_id, username, created_at FROM users WHERE user_id = %s", (user_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return dict(result) if result else None

    def update_username(self, user_id: int, new_username: str) -> tuple:
        """Update username. Returns (success, message)."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute("SELECT user_id FROM users WHERE username = %s", (new_username,))
            if cursor.fetchone():
                return False, "Username already taken"
            cursor.execute("UPDATE users SET username = %s WHERE user_id = %s", (new_username, user_id))
            conn.commit()
            return True, "Username updated successfully"
        except Exception as e:
            return False, str(e)
        finally:
            cursor.close()
            conn.close()

    def update_password(self, user_id: int, current_password: str, new_password: str) -> tuple:
        """Update password after verifying current. Returns (success, message)."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT password_hash, salt FROM users WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if not result:
            return False, "User not found"
        current_hash, _ = self._hash_password(current_password, result['salt'])
        if current_hash != result['password_hash']:
            return False, "Current password is incorrect"
        new_hash, new_salt = self._hash_password(new_password)
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password_hash = %s, salt = %s WHERE user_id = %s",
            (new_hash, new_salt, user_id)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return True, "Password updated successfully"

    def delete_user(self, user_id: int) -> tuple:
        """Delete user and all associated data. Returns (success, message)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                DELETE FROM chat_messages WHERE session_id IN (
                    SELECT session_id FROM interview_sessions WHERE user_id = %s
                )
            """, (user_id,))
            cursor.execute("DELETE FROM interview_sessions WHERE user_id = %s", (user_id,))
            cursor.execute("DELETE FROM resumes WHERE user_id = %s", (user_id,))
            cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
            conn.commit()
            return True, "Account deleted successfully"
        except Exception as e:
            conn.rollback()
            return False, str(e)
        finally:
            cursor.close()
            conn.close()

    # ─── Interview Session Management ──────────────────────────────────────

    def create_session(self, user_id: int, interview_type: str,
                       difficulty: str, num_questions: int) -> int:
        """Create a new interview session and return session_id."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            INSERT INTO interview_sessions
            (user_id, interview_type, difficulty, num_questions)
            VALUES (%s, %s, %s, %s)
            RETURNING session_id
        """, (user_id, interview_type, difficulty, num_questions))
        session_id = cursor.fetchone()['session_id']
        conn.commit()
        cursor.close()
        conn.close()
        return session_id

    def update_session_status(self, session_id: int, status: str):
        """Update session status (in_progress, completed, abandoned)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE interview_sessions
            SET status = %s, completed_at = NOW() AT TIME ZONE 'UTC'
            WHERE session_id = %s
        """, (status, session_id))
        conn.commit()
        cursor.close()
        conn.close()

    def get_user_sessions(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get recent interview sessions for a user."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT session_id, interview_type, difficulty,
                   num_questions, started_at, completed_at, status
            FROM interview_sessions
            WHERE user_id = %s
            ORDER BY started_at DESC
            LIMIT %s
        """, (user_id, limit))
        sessions = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return sessions

    # ─── Chat Message Management ───────────────────────────────────────────

    def save_message(self, session_id: int, role: str, content: str):
        """Save a chat message to the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_messages (session_id, role, content)
            VALUES (%s, %s, %s)
        """, (session_id, role, content))
        conn.commit()
        cursor.close()
        conn.close()

    def get_session_messages(self, session_id: int) -> List[Dict]:
        """Get all messages for a specific session."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT message_id, role, content, timestamp
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY timestamp ASC
        """, (session_id,))
        messages = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return messages

    def get_session_details(self, session_id: int) -> Optional[Dict]:
        """Get complete session details including all messages."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT s.session_id, s.interview_type, s.difficulty,
                   s.num_questions, s.started_at, s.completed_at, s.status,
                   u.username
            FROM interview_sessions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.session_id = %s
        """, (session_id,))
        session = cursor.fetchone()
        if not session:
            cursor.close()
            conn.close()
            return None
        session_dict = dict(session)
        cursor.execute("""
            SELECT role, content, timestamp
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY timestamp ASC
        """, (session_id,))
        session_dict['messages'] = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return session_dict

    # ─── Statistics and Analytics ──────────────────────────────────────────

    def get_user_stats(self, user_id: int) -> Dict:
        """Get statistics for a user's interview history."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT COUNT(*) as total_sessions,
                   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_sessions
            FROM interview_sessions
            WHERE user_id = %s
        """, (user_id,))
        stats = dict(cursor.fetchone())
        cursor.execute("""
            SELECT difficulty, COUNT(*) as count
            FROM interview_sessions
            WHERE user_id = %s GROUP BY difficulty
        """, (user_id,))
        stats['by_difficulty'] = {row['difficulty']: row['count'] for row in cursor.fetchall()}
        cursor.close()
        conn.close()
        return stats

    def get_completed_sessions_with_messages(self, user_id: int, limit: int = 30) -> List[Dict]:
        """Get completed sessions with messages for dashboard analytics."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT session_id, interview_type, difficulty, started_at, completed_at
            FROM interview_sessions
            WHERE user_id = %s AND status = 'completed'
            ORDER BY completed_at DESC NULLS LAST, started_at DESC
            LIMIT %s
        """, (user_id, limit))
        sessions = [dict(row) for row in cursor.fetchall()]

        for session in sessions:
            cursor.execute("""
                SELECT role, content, timestamp
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY timestamp ASC
            """, (session["session_id"],))
            session["messages"] = [dict(row) for row in cursor.fetchall()]

        cursor.close()
        conn.close()
        return sessions

    # ─── Resume Management ─────────────────────────────────────────────────

    def upload_resume(self, user_id: int, filename: str, content: str) -> tuple:
        """Upload a new resume for a user. Returns (resume_id, success, message)."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute("""
                INSERT INTO resumes (user_id, filename, content)
                VALUES (%s, %s, %s)
                RETURNING resume_id
            """, (user_id, filename, content))
            resume_id = cursor.fetchone()['resume_id']
            conn.commit()
            return resume_id, True, "Resume uploaded successfully"
        except Exception as e:
            conn.rollback()
            return None, False, str(e)
        finally:
            cursor.close()
            conn.close()

    def get_user_resumes(self, user_id: int) -> List[Dict]:
        """Get all resumes for a user without content (metadata only)."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT resume_id, filename, uploaded_at
            FROM resumes
            WHERE user_id = %s
            ORDER BY uploaded_at DESC
        """, (user_id,))
        resumes = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return resumes

    def get_resume(self, user_id: int, resume_id: int) -> Optional[Dict]:
        """Get full resume details (including content)."""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT resume_id, filename, content, uploaded_at
            FROM resumes
            WHERE user_id = %s AND resume_id = %s
        """, (user_id, resume_id))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return dict(result) if result else None

    def delete_resume(self, user_id: int, resume_id: int) -> tuple:
        """Delete a resume for a user. Returns (success, message)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM resumes WHERE user_id = %s AND resume_id = %s", (user_id, resume_id))
            if cursor.rowcount == 0:
                return False, "Resume not found or not authorized to delete"
            conn.commit()
            return True, "Resume deleted successfully"
        except Exception as e:
            conn.rollback()
            return False, str(e)
        finally:
            cursor.close()
            conn.close()
