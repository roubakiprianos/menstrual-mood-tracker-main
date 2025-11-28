import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import psycopg2
from psycopg2 import sql
import datetime
import warnings
import bcrypt

# Import pipeline with error handling
try:
    from transformers import pipeline
except ImportError:
    try:
        from transformers.pipelines import pipeline
    except ImportError as e:
        st.error(f"Failed to import transformers: {e}")
        st.stop()

# Must be the first Streamlit command
st.set_page_config(
    layout="wide",
    page_title="Menstrual Mood Tracker",
    page_icon="🌸",
    initial_sidebar_state="expanded"
)

# Suppress all warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

# =======================================================
# 1. CONFIGURATION AND AUTHENTICATION SETUP
# =======================================================

# --- Initialize Database Connection (needs to be before auth to load users) ---
@st.cache_resource
def init_db_connection():
    try:
        # Connect using the DATABASE_URL secret
        conn = psycopg2.connect(st.secrets["DATABASE_URL"])
        return conn
    except Exception as e:
        st.error("Database connection failed. Please check your 'DATABASE_URL' secret/environment variable.")
        st.error(f"Details: {e}")
        st.stop()

# --- Load Authentication Configuration from config.yaml or secrets ---
try:
    # Try to load from Streamlit secrets first (for cloud deployment)
    if "credentials" in st.secrets:
        config = {
            'credentials': st.secrets["credentials"].to_dict(),
            'cookie': st.secrets["cookie"].to_dict()
        }
    else:
        # Fall back to config.yaml for local development
        with open('config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("Configuration file (config.yaml) not found. Please create it.")
    st.stop()
except Exception as e:
    st.error(f"Error loading configuration: {e}")
    st.stop()

# --- Load NLP Pipeline (Hugging Face) ---
@st.cache_resource
def load_emotion_pipeline():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

# --- Core Functions ---

def get_emotion_content(emotion, cycle_day):
    """Returns personalized quotes, tips, and advice based on emotion and cycle day."""

    emotion_data = {
        'joy': {
            'quote': "Your joy is your sorrow unmasked. - Kahlil Gibran",
            'tip': "Celebrate this feeling! Consider journaling about what brought you joy today.",
            'color': '#FFD700',
            'emoji': '😊'
        },
        'sadness': {
            'quote': "It's okay to not be okay. Be gentle with yourself today.",
            'tip': "Try gentle movement like stretching or a short walk. Reach out to someone you trust.",
            'color': '#4169E1',
            'emoji': '💙'
        },
        'anger': {
            'quote': "Your feelings are valid. Take time to understand what you need.",
            'tip': "Try deep breathing exercises. Count to 10 before reacting. Physical activity can help release tension.",
            'color': '#FF6B35',
            'emoji': '🔥'
        },
        'fear': {
            'quote': "Courage is not the absence of fear, but the triumph over it.",
            'tip': "Ground yourself with the 5-4-3-2-1 technique. Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
            'color': '#7B68EE',
            'emoji': '💜'
        },
        'surprise': {
            'quote': "Life is full of surprises. Embrace the unexpected with curiosity.",
            'tip': "Take a moment to reflect on what surprised you and what you can learn from it.",
            'color': '#FF8C00',
            'emoji': '✨'
        },
        'disgust': {
            'quote': "Listen to your boundaries. They're protecting you.",
            'tip': "It's okay to step away from what doesn't feel right. Honor your feelings and set healthy boundaries.",
            'color': '#32CD32',
            'emoji': '🌿'
        },
        'neutral': {
            'quote': "Sometimes the most productive thing you can do is rest.",
            'tip': "Neutral days are perfectly normal. Use this calm to check in with yourself.",
            'color': '#9E9E9E',
            'emoji': '🌸'
        }
    }

    # Cycle-specific advice
    cycle_tips = {
        1: "Day 1 can be challenging. Rest is productive. Stay hydrated and be extra kind to yourself.",
        2: "Your body is working hard. Gentle movement and warm compresses can help with discomfort.",
        3: "You're past the hardest part. Notice if your energy is starting to shift.",
        4: "Energy may be returning. Listen to your body's signals.",
        5: "Notice how you're feeling. Many people start feeling lighter around now.",
        6: "You might notice increased energy. It's a great time for activities you enjoy.",
        7: "The final stretch. Reflect on your cycle and what you've learned about yourself."
    }

    content = emotion_data.get(emotion.lower(), emotion_data['neutral'])
    content['cycle_advice'] = cycle_tips.get(cycle_day, "Remember to listen to your body and honor your needs.")

    return content

def analyze_emotion(text, emotion_analyzer):
    """Analyzes text using the HuggingFace model and returns the result."""
    results = emotion_analyzer(text)[0]
    top_emotion = max(results, key=lambda x: x['score'])
    return top_emotion['label'], top_emotion['score'], results

def create_table_if_not_exists(conn):
    """Ensures the required database tables exist."""
    # Journal entries table
    CREATE_JOURNAL_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS journal_entries (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(50) NOT NULL,
        entry_date TIMESTAMP NOT NULL,
        period_day INTEGER,
        summary TEXT,
        emotion_label VARCHAR(50),
        confidence_score NUMERIC,
        joy_score NUMERIC,
        sadness_score NUMERIC,
        anger_score NUMERIC,
        fear_score NUMERIC,
        surprise_score NUMERIC,
        disgust_score NUMERIC,
        neutral_score NUMERIC
    );
    """

    # Users table for registration
    CREATE_USERS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        name VARCHAR(100) NOT NULL,
        password_hash VARCHAR(100) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    with conn.cursor() as cur:
        cur.execute(CREATE_JOURNAL_TABLE_SQL)
        cur.execute(CREATE_USERS_TABLE_SQL)
        conn.commit()

def register_user(conn, username, email, name, password_hash):
    """Register a new user in the database."""
    query = """
    INSERT INTO users (username, email, name, password_hash)
    VALUES (%s, %s, %s, %s)
    RETURNING id;
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, [username, email, name, password_hash])
            user_id = cur.fetchone()[0]
            conn.commit()
        return True, user_id
    except Exception as e:
        conn.rollback()
        if "duplicate key" in str(e).lower():
            if "username" in str(e).lower():
                return False, "Username already exists"
            elif "email" in str(e).lower():
                return False, "Email already exists"
        return False, str(e)

def load_users_from_db(conn):
    """Load all users from database and format for streamlit-authenticator."""
    query = "SELECT username, email, name, password_hash FROM users;"
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

        # Format for streamlit-authenticator
        users_dict = {}
        for username, email, name, password_hash in rows:
            users_dict[username] = {
                'email': email,
                'name': name,
                'password': password_hash
            }
        return users_dict
    except Exception:
        # If table doesn't exist or error, return empty dict
        return {}

def calculate_streak(df):
    """Calculate the current logging streak (consecutive days)."""
    if df.empty:
        return 0

    # Get unique dates (date only, not time)
    dates = pd.to_datetime(df['Date']).dt.date.unique()
    dates = sorted(dates, reverse=True)

    if len(dates) == 0:
        return 0

    # Check if there's an entry today or yesterday
    from datetime import date, timedelta
    today = date.today()
    yesterday = today - timedelta(days=1)

    if dates[0] not in [today, yesterday]:
        return 0  # Streak broken

    # Count consecutive days
    streak = 1
    for i in range(len(dates) - 1):
        diff = (dates[i] - dates[i + 1]).days
        if diff == 1:
            streak += 1
        else:
            break

    return streak

def get_insights(df):
    """Generate insights from the user's mood data."""
    if df.empty or len(df) < 3:
        return None

    insights = {}

    # Most common emotion
    insights['most_common_emotion'] = df['Emotion Label'].mode()[0] if len(df['Emotion Label'].mode()) > 0 else None

    # Best day (highest average confidence for positive emotions)
    if 'Period Day' in df.columns:
        day_avg = df.groupby('Period Day')['Confidence Score'].mean()
        insights['best_day'] = int(day_avg.idxmax()) if not day_avg.empty else None

    # Total entries
    insights['total_entries'] = len(df)

    return insights

def get_period_fact_of_day():
    """Returns a period fact based on the current day."""
    from datetime import date

    facts = [
        {
            'fact': "The average menstrual cycle lasts 28 days, but anywhere from 21 to 35 days is considered normal.",
            'icon': '📅',
            'tip': 'Track your cycle to understand your unique pattern!'
        },
        {
            'fact': "Period blood isn't actually just blood - it's a mix of blood, tissue from the uterine lining, and vaginal secretions.",
            'icon': '🔬',
            'tip': 'Changes in color and consistency are usually normal.'
        },
        {
            'fact': "You lose about 2-3 tablespoons of blood during your entire period, though it can feel like much more!",
            'icon': '💧',
            'tip': 'Heavy bleeding (more than 80ml) should be discussed with a doctor.'
        },
        {
            'fact': "Period cramps happen because your uterus contracts to shed its lining. Prostaglandins are the chemicals responsible.",
            'icon': '💪',
            'tip': 'Heat, exercise, and anti-inflammatory medications can help!'
        },
        {
            'fact': "Your metabolism can increase slightly during your period, which is why you might feel hungrier!",
            'icon': '🍽️',
            'tip': 'Listen to your body and nourish it with what it needs.'
        },
        {
            'fact': "PMS symptoms can start up to 2 weeks before your period and affect up to 90% of menstruating people.",
            'icon': '🧠',
            'tip': 'Tracking your symptoms can help you prepare and cope better.'
        },
        {
            'fact': "Exercise during your period can actually help reduce cramps and improve your mood through endorphin release.",
            'icon': '🏃‍♀️',
            'tip': 'Even gentle movement like walking or stretching counts!'
        },
        {
            'fact': "The first day of your period is considered Day 1 of your menstrual cycle.",
            'icon': '🌟',
            'tip': 'This is when hormones are at their lowest before starting to rise again.'
        },
        {
            'fact': "Chocolate cravings during your period are real! Your body needs more magnesium, and chocolate is rich in it.",
            'icon': '🍫',
            'tip': 'Dark chocolate is a great source of magnesium and iron.'
        },
        {
            'fact': "Your sense of smell can be heightened during certain phases of your menstrual cycle.",
            'icon': '👃',
            'tip': 'This is linked to hormonal changes throughout your cycle.'
        },
        {
            'fact': "Period pain that interferes with daily activities could be a sign of endometriosis or other conditions.",
            'icon': '⚠️',
            'tip': 'Don\'t ignore severe pain - consult a healthcare provider.'
        },
        {
            'fact': "Your period can affect your sleep quality due to hormonal fluctuations, especially progesterone levels.",
            'icon': '😴',
            'tip': 'Prioritize rest and maintain good sleep hygiene during your cycle.'
        },
        {
            'fact': "The menstrual cycle is divided into 4 phases: menstruation, follicular, ovulation, and luteal.",
            'icon': '🔄',
            'tip': 'Each phase has unique hormonal patterns and potential mood effects.'
        },
        {
            'fact': "Stress can affect your menstrual cycle, potentially causing it to be late, early, or skipped entirely.",
            'icon': '🧘‍♀️',
            'tip': 'Stress management techniques can help regulate your cycle.'
        },
        {
            'fact': "Period products have evolved significantly - from pads and tampons to menstrual cups and period underwear!",
            'icon': '🌸',
            'tip': 'Find what works best for your body and lifestyle.'
        },
        {
            'fact': "Your energy levels naturally fluctuate throughout your cycle - it's not just in your head!",
            'icon': '⚡',
            'tip': 'Plan important tasks during your high-energy phases when possible.'
        },
        {
            'fact': "The color of your period blood can tell you things about your health - bright red is fresh, dark is older blood.",
            'icon': '🎨',
            'tip': 'Very pale or gray discharge should be checked by a doctor.'
        },
        {
            'fact': "You can still get pregnant during your period, though it's less likely. Ovulation timing varies!",
            'icon': '💡',
            'tip': 'Use contraception consistently if pregnancy prevention is important.'
        },
        {
            'fact': "Orgasms can help relieve menstrual cramps by releasing endorphins and relaxing the uterine muscles.",
            'icon': '💕',
            'tip': 'Self-care comes in many forms - do what feels right for you!'
        },
        {
            'fact': "The average person will menstruate for about 7 years of their lifetime!",
            'icon': '⏰',
            'tip': 'That\'s why understanding and tracking your cycle is so valuable.'
        },
        {
            'fact': "Hydration is extra important during your period - it can help reduce bloating and headaches.",
            'icon': '💦',
            'tip': 'Aim for at least 8 glasses of water throughout the day.'
        },
        {
            'fact': "Iron levels can drop during menstruation due to blood loss, which may cause fatigue.",
            'icon': '🥬',
            'tip': 'Eat iron-rich foods like leafy greens, beans, and lean meats.'
        },
        {
            'fact': "Period apps and trackers can help predict your next period and identify patterns in your cycle.",
            'icon': '📱',
            'tip': 'You\'re already doing this - great job taking charge of your health!'
        },
        {
            'fact': "Hormonal birth control works by preventing ovulation, which is why some people don't get periods on it.",
            'icon': '💊',
            'tip': 'Talk to your doctor about what\'s right for your body.'
        },
        {
            'fact': "Mood changes during your cycle are linked to fluctuating levels of estrogen and progesterone.",
            'icon': '🎭',
            'tip': 'Tracking your moods can help you understand and prepare for these changes.'
        },
        {
            'fact': "Ancient cultures celebrated menstruation as a sign of fertility and feminine power.",
            'icon': '🏛️',
            'tip': 'Your body is capable of amazing things!'
        },
        {
            'fact': "The word 'menstruation' comes from Latin 'mensis' meaning 'month' - linked to lunar cycles.",
            'icon': '🌙',
            'tip': 'Many cultures have connected menstrual cycles to moon phases.'
        },
        {
            'fact': "Everyone's period is different - what's normal for you might not be normal for someone else.",
            'icon': '✨',
            'tip': 'Trust your body and speak up if something feels wrong.'
        },
        {
            'fact': "Fiber-rich foods can help with period symptoms by regulating hormones and reducing bloating.",
            'icon': '🥦',
            'tip': 'Include whole grains, fruits, and vegetables in your diet.'
        },
        {
            'fact': "Your pain tolerance can actually decrease during menstruation due to hormonal changes.",
            'icon': '🌡️',
            'tip': 'Be extra gentle with yourself during this time.'
        },
        {
            'fact': "Regular exercise can help regulate your menstrual cycle and reduce PMS symptoms.",
            'icon': '🤸‍♀️',
            'tip': 'Find activities you enjoy to make it sustainable long-term.'
        }
    ]

    # Use day of year to get consistent fact for the day
    day_of_year = date.today().timetuple().tm_yday
    fact_index = day_of_year % len(facts)

    return facts[fact_index]

def load_user_history(conn, user_id):
    """Loads all historical data for the logged-in user."""
    query = "SELECT * FROM journal_entries WHERE user_id = %s ORDER BY entry_date DESC;"
    try:
        # Rename columns to match the application's expected DataFrame columns
        df = pd.read_sql(query, conn, params=[user_id])
        df.columns = [
            'id', 'User ID', 'Date', 'Period Day', 'Summary', 'Emotion Label', 'Confidence Score',
            'Joy_Score', 'Sadness_Score', 'Anger_Score', 'Fear_Score', 'Surprise_Score', 'Disgust_Score', 'Neutral_Score'
        ]
        # Ensure correct data types and format datetime
        df['Date'] = pd.to_datetime(df['Date'])
        # Create a formatted date column for display (without seconds)
        df['Date_Display'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M')
        return df.drop(columns=['id', 'User ID']) # Drop internal columns for app display
    except Exception as e:
        # If table doesn't exist yet or other load error, return an empty structure
        st.warning(f"No history found or error loading data. Start logging! ({e})")
        return pd.DataFrame(columns=[
            'Date', 'Period Day', 'Summary', 'Emotion Label', 'Confidence Score',
            'Joy_Score', 'Sadness_Score', 'Anger_Score', 'Fear_Score', 'Surprise_Score', 'Disgust_Score', 'Neutral_Score'
        ])

def delete_all_user_entries(conn, user_id):
    """Deletes all journal entries for the specified user."""
    query = "DELETE FROM journal_entries WHERE user_id = %s;"
    try:
        with conn.cursor() as cur:
            cur.execute(query, [user_id])
            deleted_count = cur.rowcount
            conn.commit()
        return True, deleted_count
    except Exception as e:
        conn.rollback()
        return False, str(e)

# =======================================================
# INITIALIZE DATABASE AND LOAD USERS
# =======================================================

# Initialize database connection and create tables
conn = init_db_connection()
create_table_if_not_exists(conn)

# Load users from database and merge with config users
db_users = load_users_from_db(conn)
if db_users:
    # Merge database users with config users (config users take precedence)
    for username, user_data in db_users.items():
        if username not in config['credentials']['usernames']:
            config['credentials']['usernames'][username] = user_data

# Re-initialize authenticator with updated user list
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# =======================================================
# 2. LOGIN WIDGET AND STATUS CHECK
# =======================================================

# Check if user is not authenticated
if st.session_state.get("authentication_status") != True:
    # Show login and registration tabs
    tab1, tab2 = st.tabs(["🔐 Login", "✨ Create Account"])

    with tab1:
        st.markdown("### Login to Your Account")
        authenticator.login(location='main')

    with tab2:
        st.markdown("### Create a New Account")
        st.markdown("Join to track your menstrual cycle and mood patterns!")

        with st.form("registration_form"):
            col1, col2 = st.columns(2)

            with col1:
                new_name = st.text_input("Full Name *", placeholder="Enter your name")
                new_username = st.text_input("Username *", placeholder="Choose a username")

            with col2:
                new_email = st.text_input("Email *", placeholder="your.email@example.com")
                new_password = st.text_input("Password *", type="password", placeholder="Choose a password")

            st.markdown("*All fields are required")

            register_button = st.form_submit_button("Create Account", type="primary")

            if register_button:
                if not all([new_name, new_username, new_email, new_password]):
                    st.error("❌ Please fill in all fields")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters long")
                elif "@" not in new_email:
                    st.error("❌ Please enter a valid email address")
                else:
                    # Hash the password using bcrypt
                    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

                    # Register user in database
                    success, result = register_user(conn, new_username, new_email, new_name, hashed_password)

                    if success:
                        st.success(f"✅ Account created successfully! User ID: {result}")
                        st.info("👉 Please switch to the Login tab to sign in with your new account.")
                        # Force a rerun to reload users from database
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ Registration failed: {result}")

# --- Stop execution if user is not authenticated ---
if st.session_state.get("authentication_status") == False:
    st.error('Username/password is incorrect')
    st.stop()

if st.session_state.get("authentication_status") == None:
    st.warning('Please enter your username and password to proceed.')
    st.stop()

# =======================================================
# 3. AUTHENTICATED APPLICATION START
# =======================================================

if st.session_state.get("authentication_status"):
    # Initialize all resources once login is successful
    # (conn is already initialized at the top for user authentication)
    emotion_analyzer = load_emotion_pipeline()

    # --- Custom CSS for Orange & Purple Styling with Animations ---
    st.markdown("""
    <style>
        /* Orange and Purple color theme */
        :root {
            --primary-color: #FF6B35;
            --secondary-color: #7B68EE;
            --accent-color: #FFA07A;
        }

        /* Smooth fade-in animation */
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Pulse animation for metrics */
        @keyframes pulse {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
        }

        /* Colorful form styling with animation */
        .stForm {
            background: linear-gradient(135deg, #FFF4E6 0%, #F3E5F5 100%);
            padding: 2rem;
            border-radius: 12px;
            border: 2px solid #FFB366;
            animation: fadeIn 0.5s ease-out;
            transition: all 0.3s ease;
        }

        .stForm:hover {
            box-shadow: 0 8px 16px rgba(255, 107, 53, 0.15);
            transform: translateY(-2px);
        }

        /* Styled text inputs with focus effect */
        .stTextArea textarea {
            border-radius: 8px;
            border: 2px solid #FFB366;
            font-size: 15px;
            background-color: white;
            transition: all 0.3s ease;
        }

        .stTextArea textarea:focus {
            border-color: #7B68EE;
            box-shadow: 0 0 0 3px rgba(123, 104, 238, 0.1);
        }

        /* Orange-Purple gradient buttons with enhanced hover */
        .stButton > button {
            background: linear-gradient(135deg, #FF6B35 0%, #7B68EE 100%);
            color: white;
            border-radius: 20px;
            padding: 0.7rem 2rem;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4);
        }

        .stButton > button:active {
            transform: translateY(-1px);
        }

        /* Vibrant metrics with pulse on hover */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #FF6B35;
            transition: all 0.3s ease;
        }

        [data-testid="stMetric"]:hover [data-testid="stMetricValue"] {
            animation: pulse 1s ease-in-out;
        }

        /* Colorful headers with smooth appearance */
        h1 {
            color: #7B68EE;
            text-align: center;
            font-weight: 700;
            animation: fadeIn 0.6s ease-out;
        }

        h2, h3 {
            color: #FF6B35;
            font-weight: 600;
            animation: fadeIn 0.5s ease-out;
        }

        /* Gradient sidebar with smooth transition */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FFF4E6 0%, #F3E5F5 100%);
        }

        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.6);
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
        }

        [data-testid="stSidebar"] [data-testid="stMetric"]:hover {
            background: rgba(255, 255, 255, 0.9);
            transform: translateX(5px);
        }

        /* Success messages with slide-in */
        .stSuccess {
            background-color: #D4EDDA;
            border-radius: 10px;
            padding: 1rem;
            border-left: 4px solid #28A745;
            animation: fadeIn 0.4s ease-out;
        }

        /* Info messages */
        .stInfo {
            animation: fadeIn 0.4s ease-out;
        }

        /* Expander styling with hover effect */
        .streamlit-expanderHeader {
            background-color: #FFF4E6;
            border-radius: 8px;
            font-weight: 600;
            border: 1px solid #FFB366;
            transition: all 0.3s ease;
        }

        .streamlit-expanderHeader:hover {
            background-color: #FFE4CC;
            border-color: #FF6B35;
        }

        /* Clean white background */
        .main {
            background-color: #FFFFFF;
        }

        /* Better text colors */
        p, label, span {
            color: #333333;
        }

        /* Dataframe hover effect */
        .stDataFrame {
            animation: fadeIn 0.5s ease-out;
        }

        /* Chart containers with fade-in */
        [data-testid="stPlotlyChart"] {
            animation: fadeIn 0.6s ease-out;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- UI Elements for Logged-in User ---
    authenticator.logout('Logout', 'sidebar')

    # Sidebar header with decorative elements
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem 0; background: linear-gradient(135deg, #FFB366 0%, #FF6B35 100%);
                border-radius: 10px; margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0; font-size: 1.5rem;'>🌸</h2>
        <p style='color: white; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Cycle Tracker</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"### 👋 Welcome, {st.session_state['name']}!")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")

    # Load data specific to the current user
    if 'history_df' not in st.session_state:
        st.session_state.history_df = load_user_history(conn, st.session_state['username'])

    # Add sidebar stats
    if not st.session_state.history_df.empty:
        # Calculate streak
        streak = calculate_streak(st.session_state.history_df)
        st.sidebar.metric("🔥 Current Streak", f"{streak} days")

        # Show insights
        insights = get_insights(st.session_state.history_df)
        if insights:
            st.sidebar.metric("📝 Total Entries", insights['total_entries'])
            if insights['most_common_emotion']:
                emotion_content = get_emotion_content(insights['most_common_emotion'], 1)
                st.sidebar.metric("💭 Most Common Emotion",
                                f"{emotion_content['emoji']} {insights['most_common_emotion'].capitalize()}")

        # Insights section
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💡 Insights")
        if insights and insights['total_entries'] >= 3:
            if insights['best_day']:
                st.sidebar.info(f"📊 Day {insights['best_day']} tends to have your highest confidence scores!")
            if streak >= 3:
                st.sidebar.success(f"🎉 Amazing! You've logged {streak} days in a row!")
        else:
            st.sidebar.info("Log more entries to unlock personalized insights!")

    # --- Reset Data Section ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")

    with st.sidebar.expander("🗑️ Reset All Data"):
        st.warning("⚠️ This will permanently delete ALL your journal entries. This action cannot be undone!")

        # Confirmation checkbox
        confirm_reset = st.checkbox("I understand that this will delete all my data")

        if st.button("🔴 Delete All Entries", disabled=not confirm_reset, type="secondary"):
            if confirm_reset:
                success, result = delete_all_user_entries(conn, st.session_state['username'])
                if success:
                    st.success(f"✅ Successfully deleted {result} entries!")
                    # Clear session state to reload empty data
                    st.session_state.history_df = pd.DataFrame(columns=[
                        'Date', 'Period Day', 'Summary', 'Emotion Label', 'Confidence Score',
                        'Joy_Score', 'Sadness_Score', 'Anger_Score', 'Fear_Score', 'Surprise_Score', 'Disgust_Score', 'Neutral_Score'
                    ])
                    st.rerun()
                else:
                    st.error(f"❌ Error deleting entries: {result}")

    # --- App Structure ---
    st.title("🌸 Menstrual Mood Tracker")
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem;
                background: linear-gradient(135deg, #FFF4E6 0%, #F3E5F5 100%);
                border-radius: 12px; margin-bottom: 2rem; border: 2px solid #FFB366;'>
        <p style='font-size: 1.1rem; color: #5D4E60; margin: 0; line-height: 1.7; font-weight: 500;'>
            Track your mood and symptoms during your cycle to understand patterns over time ✨
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Display Fact of the Day
    fact_data = get_period_fact_of_day()
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #E8F5E9 0%, #FCE4EC 100%);
                padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;
                border-left: 6px solid #FF6B35; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h3 style='color: #FF6B35; margin-top: 0; font-size: 1.3rem;'>
            {fact_data['icon']} Did You Know? - Fact of the Day
        </h3>
        <p style='font-size: 1.05rem; color: #2D3436; line-height: 1.8; margin: 0.8rem 0;'>
            <strong>{fact_data['fact']}</strong>
        </p>
        <div style='background: rgba(255,255,255,0.7); padding: 0.8rem;
                    border-radius: 8px; margin-top: 1rem;'>
            <p style='margin: 0; color: #7B68EE; font-weight: 500;'>
                💡 <em>{fact_data['tip']}</em>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("❓ How the Emotion Analysis Works"):
        st.markdown("""
        The application uses a **Hugging Face** pre-trained language model to analyze your text and determine the probability (Confidence Score) of **seven core emotions**. The score ranges from **0.0** (no evidence) to **1.0** (absolute certainty).
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Input and Submission Form ---
    st.subheader("📝 Log Your Entry")
    with st.form(key='period_form'):
        col_date, col_time, col_day = st.columns([1, 1, 1])

        with col_date:
            entry_date = st.date_input("📅 Date", value="today")

        with col_time:
            entry_time = st.time_input("🕐 Time", value="now")

        with col_day:
            period_day = st.slider("🗓️ Cycle Day", min_value=1, max_value=7, value=1, step=1)
            st.caption(f"Day {period_day} of your cycle")

        st.markdown("<br>", unsafe_allow_html=True)
        user_summary = st.text_area(
            "💭 How are you feeling today?",
            max_chars=300,
            height=120,
            placeholder="Describe your mood and any symptoms you're experiencing...",
            help="Write 2-3 sentences about how you're feeling"
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.form_submit_button(label='✨ Analyze and Save Entry', use_container_width=True)

    # --- Analysis and Logging ---
    if submit_button and user_summary:

        emotion_label, confidence_score, results = analyze_emotion(user_summary, emotion_analyzer)
        # Combine date and time
        from datetime import datetime as dt
        log_datetime = dt.combine(entry_date, entry_time)
        log_date_str = log_datetime.strftime("%Y-%m-%d %H:%M")
        
        # Prepare data for dictionary (required for insertion)
        entry_dict = {
            'User ID': st.session_state['username'], # Used for DB insertion
            'Date': log_date_str,
            'Period Day': period_day,
            'Summary': user_summary,
            'Emotion Label': emotion_label,
            'Confidence Score': confidence_score,
        }
        for item in results:
            entry_dict[f"{item['label'].capitalize()}_Score"] = item['score']

        # --- SAVE DATA TO POSTGRESQL ---
        try:
            with conn.cursor() as cur:
                INSERT_SQL = """
                INSERT INTO journal_entries 
                (user_id, entry_date, period_day, summary, emotion_label, confidence_score, 
                 joy_score, sadness_score, anger_score, fear_score, surprise_score, disgust_score, neutral_score) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """
                
                data = (
                    st.session_state['username'],
                    log_datetime,
                    period_day, 
                    user_summary, 
                    emotion_label, 
                    confidence_score,
                    entry_dict['Joy_Score'], entry_dict['Sadness_Score'], entry_dict['Anger_Score'], 
                    entry_dict['Fear_Score'], entry_dict['Surprise_Score'], entry_dict['Disgust_Score'], 
                    entry_dict['Neutral_Score']
                )
                
                cur.execute(INSERT_SQL, data)
                conn.commit()
            
            # After successful DB insert, reload the user's data to update the session state
            st.session_state.history_df = load_user_history(conn, st.session_state['username'])

            # --- Immediate Feedback (Aesthetic Update) ---
            st.markdown("<br>", unsafe_allow_html=True)

            # Confetti celebration!
            st.balloons()

            st.success(f"✅ Entry saved successfully for {log_date_str} (Day {period_day})")

            # Get personalized content
            emotion_content = get_emotion_content(emotion_label, period_day)

            st.markdown("""
            <div style='background: linear-gradient(135deg, #FFF4E6 0%, #F3E5F5 100%);
                        padding: 1.5rem; border-radius: 12px;
                        margin: 1rem 0; border: 2px solid #FFB366;'>
            """, unsafe_allow_html=True)

            col_emotion, col_confidence = st.columns(2)

            with col_emotion:
                st.metric(label="🎭 Dominant Emotion", value=f"{emotion_content['emoji']} {emotion_label.capitalize()}")

            with col_confidence:
                confidence_pct = f"{confidence_score*100:.1f}%"
                st.metric(label="📊 Confidence Score", value=confidence_pct)

            st.markdown("</div>", unsafe_allow_html=True)

            # Personalized Wellness Content
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #FFFBF0 0%, #F8F4FF 100%);
                        padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                        border-left: 5px solid {emotion_content['color']};'>
                <h4 style='color: {emotion_content['color']}; margin-top: 0;'>💭 A Moment of Reflection</h4>
                <p style='font-style: italic; color: #5D4E60; font-size: 1.05rem; line-height: 1.6;'>
                    "{emotion_content['quote']}"
                </p>
                <hr style='border: none; border-top: 1px solid #E0D8E8; margin: 1rem 0;'>
                <h4 style='color: #FF6B35; margin-bottom: 0.5rem;'>🌿 Wellness Tip</h4>
                <p style='color: #5D4E60; line-height: 1.6;'>{emotion_content['tip']}</p>
                <hr style='border: none; border-top: 1px solid #E0D8E8; margin: 1rem 0;'>
                <h4 style='color: #7B68EE; margin-bottom: 0.5rem;'>📅 Cycle Day {period_day} Insight</h4>
                <p style='color: #5D4E60; line-height: 1.6;'>{emotion_content['cycle_advice']}</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Failed to save entry to database. Details: {e}")
            conn.rollback()
        
        st.markdown("---")

    # =======================================================
    # 4. HISTORY AND VISUALIZATION
    # =======================================================
    if not st.session_state.history_df.empty:

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("---")
        plot_df = st.session_state.history_df.copy()

        # --- Line Chart: Confidence Trend ---
        st.header("📈 Mood Trend Over Time")

        # Vibrant color palette for emotions
        emotion_colors = {
            'joy': '#FFD700',
            'sadness': '#4169E1',
            'anger': '#FF6B35',
            'fear': '#7B68EE',
            'surprise': '#FF8C00',
            'disgust': '#32CD32',
            'neutral': '#9E9E9E'
        }

        fig = px.line(
            plot_df,
            x='Date',
            y='Confidence Score',
            color='Emotion Label',
            title='Your Emotional Journey ✨',
            markers=True,
            line_shape='spline',
            color_discrete_map=emotion_colors
        )

        fig.update_layout(
            yaxis_range=[0, 1.1],
            plot_bgcolor='#FFF9F5',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=13, color='#5D4E60'),
            title_font=dict(size=19, color='#7B68EE', family='Arial'),
            legend=dict(
                bgcolor='#FFFFFF',
                bordercolor='#FFB366',
                borderwidth=2
            ),
            xaxis=dict(
                tickformat='%Y-%m-%d %H:%M'
            ),
            hovermode='x unified'
        )

        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)

        # --- Grouped Bar Chart: Aggregation by Period Day ---
        st.markdown("---")
        st.header("🗓️ Emotional Patterns by Cycle Day")

        emotion_cols = [
            'Joy_Score', 'Sadness_Score', 'Anger_Score',
            'Fear_Score', 'Surprise_Score', 'Disgust_Score', 'Neutral_Score'
        ]

        # Check if necessary columns exist and we have enough data
        if all(col in st.session_state.history_df.columns for col in emotion_cols) and len(st.session_state.history_df) >= 2:

            plot_df['Period Day'] = plot_df['Period Day'].astype(int)

            melted_df = plot_df.melt(
                id_vars=['Period Day'],
                value_vars=emotion_cols,
                var_name='Emotion',
                value_name='Average Confidence'
            )

            period_agg = melted_df.groupby(['Period Day', 'Emotion'])['Average Confidence'].mean().reset_index()

            fig_period = px.bar(
                period_agg,
                x='Period Day',
                y='Average Confidence',
                color='Emotion',
                barmode='group',
                title='How Emotions Vary Throughout Your Cycle 💫',
                color_discrete_map={
                    'Joy_Score': '#FFD700',
                    'Sadness_Score': '#4169E1',
                    'Anger_Score': '#FF6B35',
                    'Fear_Score': '#7B68EE',
                    'Surprise_Score': '#FF8C00',
                    'Disgust_Score': '#32CD32',
                    'Neutral_Score': '#9E9E9E'
                }
            )

            fig_period.update_layout(
                yaxis_range=[0, 0.7],
                legend_title_text='Emotion Type',
                xaxis_title="Day of Menstruation",
                yaxis_title="Average Confidence Score",
                plot_bgcolor='#FFF9F5',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=13, color='#5D4E60'),
                title_font=dict(size=19, color='#7B68EE', family='Arial'),
                legend=dict(
                    bgcolor='#FFFFFF',
                    bordercolor='#FFB366',
                    borderwidth=2
                ),
                xaxis=dict(dtick=1)
            )
            st.plotly_chart(fig_period, use_container_width=True)
        else:
            st.info("Log at least two entries with different cycle days to see pattern comparisons.")

        # --- Raw History Table ---
        st.markdown("---")

        # Header with export button
        col_header, col_export = st.columns([3, 1])
        with col_header:
            st.header("📖 Your Journal History")
        with col_export:
            # Prepare display dataframe with formatted dates
            display_df = st.session_state.history_df.copy()
            if 'Date_Display' in display_df.columns:
                display_columns = ['Date_Display', 'Period Day', 'Summary', 'Emotion Label', 'Confidence Score',
                                 'Joy_Score', 'Sadness_Score', 'Anger_Score', 'Fear_Score',
                                 'Surprise_Score', 'Disgust_Score', 'Neutral_Score']
                display_df = display_df[display_columns]
                display_df = display_df.rename(columns={'Date_Display': 'Date & Time'})

            # Export button
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export CSV",
                data=csv,
                file_name=f"mood_journal_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Download your journal entries as a CSV file"
            )

        st.dataframe(
            display_df.iloc[::-1],
            width='stretch',
            height=400
        )
    else:
        st.info("✨ No entries yet! Start logging your moods to see beautiful visualizations and patterns.")

    # --- Footer Credit ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p style='color: #7B68EE; font-size: 1rem; font-weight: 500;'>
            💜 Developed with care by Rouba 🌸
        </p>
        <p style='color: #FF6B35; font-size: 0.9rem;'>
            Understanding your cycle, one day at a time ✨
        </p>
    </div>
    """, unsafe_allow_html=True)
