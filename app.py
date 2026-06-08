import streamlit as st
import joblib
import re
import sqlite3
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from collections import Counter
import io
from urllib.parse import urlparse
import time
import hashlib
import google.generativeai as genai

# ========== GEMINI API CONFIGURATION ==========
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyB_C6vNAg4u5ks7AelDfBEOn6OxGchWiMU")
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except Exception as e:
    GEMINI_AVAILABLE = False
    st.warning(f"Gemini API not configured: {e}")

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="AI Fake News Detection Pro",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .header-main { font-size: 36px; font-weight: bold; color: #00d9ff; margin: 20px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
    
    .success-box { 
        padding: 20px; border-radius: 15px; background-color: #238636; 
        border: 3px solid #3fb950; border-left: 8px solid #3fb950;
        color: #ffffff; font-weight: bold; font-size: 16px;
    }
    .error-box { 
        padding: 20px; border-radius: 15px; background-color: #da3633; 
        border: 3px solid #f85149; border-left: 8px solid #f85149;
        color: #ffffff; font-weight: bold; font-size: 16px;
    }
    .warning-box { 
        padding: 20px; border-radius: 15px; background-color: #9e6a03; 
        border: 3px solid #d29922; border-left: 8px solid #d29922;
        color: #ffffff; font-weight: bold; font-size: 16px;
    }
    
    .metric-card-real {
        background: linear-gradient(135deg, #238636 0%, #2d7e3a 100%);
        padding: 20px 15px;
        border-radius: 15px;
        border: 3px solid #3fb950;
        color: #ffffff;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(63, 185, 80, 0.3);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 140px;
        white-space: nowrap;
        overflow: hidden;
    }
    
    .metric-card-real h2 {
        margin: 0;
        font-size: 14px;
        line-height: 1;
    }
    
    .metric-card-real p {
        margin: 8px 0 0 0;
        font-size: 26px;
        line-height: 1;
    }
    
    .metric-card-fake {
        background: linear-gradient(135deg, #da3633 0%, #f85149 100%);
        padding: 20px 15px;
        border-radius: 15px;
        border: 3px solid #f85149;
        color: #ffffff;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(248, 81, 73, 0.3);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 140px;
        white-space: nowrap;
        overflow: hidden;
    }
    
    .metric-card-fake h2 {
        margin: 0;
        font-size: 14px;
        line-height: 1;
    }
    
    .metric-card-fake p {
        margin: 8px 0 0 0;
        font-size: 26px;
        line-height: 1;
    }
    
    .metric-advanced { 
        background: linear-gradient(135deg, #0969da 0%, #1f6feb 100%);
        padding: 20px 15px;
        border-radius: 15px;
        border: 3px solid #388bfd;
        border-left: 8px solid #388bfd;
        margin: 0;
        color: #ffffff;
        font-weight: bold;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 140px;
        box-shadow: 0 8px 16px rgba(56, 139, 253, 0.3);
    }
    
    .metric-advanced h3 {
        margin: 0;
        font-size: 13px;
        line-height: 1;
    }
    
    .metric-advanced p {
        margin: 8px 0 0 0;
        font-size: 26px;
        line-height: 1;
    }
    
    .metric-confidence-high {
        background: linear-gradient(135deg, #238636 0%, #3fb950 100%);
        padding: 20px 15px;
        border-radius: 15px;
        border: 3px solid #3fb950;
        color: #ffffff;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 140px;
    }
    
    .metric-confidence-high h3 {
        margin: 0;
        font-size: 13px;
        line-height: 1;
    }
    
    .metric-confidence-high p {
        margin: 8px 0 0 0;
        font-size: 26px;
        line-height: 1;
    }
    
    .metric-confidence-med {
        background: linear-gradient(135deg, #9e6a03 0%, #d29922 100%);
        padding: 20px 15px;
        border-radius: 15px;
        border: 3px solid #d29922;
        color: #ffffff;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 140px;
    }
    
    .metric-confidence-med h3 {
        margin: 0;
        font-size: 13px;
        line-height: 1;
    }
    
    .metric-confidence-med p {
        margin: 8px 0 0 0;
        font-size: 26px;
        line-height: 1;
    }
    
    .metric-confidence-low {
        background: linear-gradient(135deg, #da3633 0%, #f85149 100%);
        padding: 20px 15px;
        border-radius: 15px;
        border: 3px solid #f85149;
        color: #ffffff;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 140px;
    }
    
    .metric-confidence-low h3 {
        margin: 0;
        font-size: 13px;
        line-height: 1;
    }
    
    .metric-confidence-low p {
        margin: 8px 0 0 0;
        font-size: 26px;
        line-height: 1;
    }
    
    .tab-button {
        background-color: #21262d;
        color: #c9d1d9;
        border: 2px solid #30363d;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 14px;
    }
    
    h2, h3, h4 {
        color: #c9d1d9 !important;
        font-weight: bold;
    }
    
    p, span {
        color: #c9d1d9 !important;
    }
    
    /* Chat Message Styling */
    .chat-user-msg {
        background: linear-gradient(135deg, #0969da 0%, #1f6feb 100%);
        padding: 15px 18px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 5px solid #388bfd;
        box-shadow: 0 4px 12px rgba(56, 139, 253, 0.2);
        color: #ffffff;
    }
    
    .chat-bot-msg {
        background: linear-gradient(135deg, #1f6feb 0%, #0969da 100%);
        padding: 15px 18px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 5px solid #58a6ff;
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.2);
        color: #e6edf3;
        line-height: 1.6;
    }
    
    .feedback-section {
        display: flex;
        gap: 8px;
        margin-top: 8px;
    }
    
    .copy-btn {
        background-color: #238636;
        color: white;
        border: none;
        padding: 6px 12px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 11px;
        transition: all 0.2s ease;
    }
    
    .copy-btn:hover {
        background-color: #2ea043;
        transform: scale(1.05);
    }
    
    .stat-card {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #30363d;
        text-align: center;
    }
    
    .stat-card h4 {
        margin: 0 0 8px 0;
        color: #58a6ff;
        font-size: 13px;
    }
    
    .stat-card p {
        margin: 0;
        color: #79c0ff;
        font-size: 20px;
        font-weight: bold;
    }
    
    .session-timer {
        color: #79c0ff;
        font-size: 12px;
        margin: 8px 0;
        padding: 8px 12px;
        background: rgba(88, 166, 255, 0.1);
        border-radius: 6px;
        border-left: 3px solid #58a6ff;
    }
    </style>
""", unsafe_allow_html=True)

# ========== LOAD MODELS ==========
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("vectorizer.jb")
        try:
            model = joblib.load("lr_model.jb")
        except:
            model = joblib.load("model.jb")
        return vectorizer, model
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def init_db():
    conn = sqlite3.connect("advanced_analysis.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analysis 
                 (id INTEGER PRIMARY KEY, text TEXT, prediction TEXT, 
                  confidence REAL, red_flags INTEGER, bias_score REAL, 
                  credibility REAL, timestamp DATETIME)''')
    
    # Create feedback table
    c.execute('''CREATE TABLE IF NOT EXISTS user_feedback
                 (id INTEGER PRIMARY KEY, question TEXT, response TEXT, 
                  rating TEXT, comment TEXT, timestamp DATETIME)''')
    conn.commit()
    conn.close()

def save_feedback(question, response, rating, comment=""):
    """Save user feedback to database"""
    conn = sqlite3.connect("advanced_analysis.db")
    c = conn.cursor()
    c.execute('''INSERT INTO user_feedback (question, response, rating, comment, timestamp)
                 VALUES (?, ?, ?, ?, ?)''',
              (question[:200], response[:500], rating, comment[:300], datetime.datetime.now()))
    conn.commit()
    conn.close()

def get_feedback_stats():
    """Get feedback statistics"""
    conn = sqlite3.connect("advanced_analysis.db")
    c = conn.cursor()
    c.execute('''SELECT rating, COUNT(*) FROM user_feedback GROUP BY rating''')
    stats = dict(c.fetchall())
    conn.close()
    return stats

vectorizer, model = load_models()
init_db()

# ========== SESSION STATE INITIALIZATION ==========
if 'article_text' not in st.session_state:
    st.session_state.article_text = ""
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = {}  # Track feedback per message
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = time.time()
if 'message_count' not in st.session_state:
    st.session_state.message_count = 0
if 'article_summary' not in st.session_state:
    st.session_state.article_summary = None
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False
if 'full_analysis_data' not in st.session_state:
    st.session_state.full_analysis_data = None
if 'gemini_cache' not in st.session_state:
    st.session_state.gemini_cache = {}
if 'last_api_call' not in st.session_state:
    st.session_state.last_api_call = 0
if 'quota_wait_until' not in st.session_state:
    st.session_state.quota_wait_until = 0

# ========== RATE LIMITING & CACHING FOR FREE TIER ==========
# Free Tier Limits: 15 requests per minute per API key
# Strategy: Cache responses, implement proper rate limiting, fallback to local analysis

RATE_LIMIT_CALLS = 15  # Free tier: 15 requests per minute
RATE_LIMIT_WINDOW = 60  # 60 seconds
API_CALL_COUNTS = []  # Track API call timestamps

def check_rate_limit():
    """Check if we're within free tier rate limits"""
    current_time = time.time()
    # Remove calls older than rate limit window
    API_CALL_COUNTS[:] = [call_time for call_time in API_CALL_COUNTS 
                         if current_time - call_time < RATE_LIMIT_WINDOW]
    
    # Check if we can make another call
    if len(API_CALL_COUNTS) < RATE_LIMIT_CALLS:
        return True, len(API_CALL_COUNTS), RATE_LIMIT_CALLS
    else:
        return False, len(API_CALL_COUNTS), RATE_LIMIT_CALLS

def record_api_call():
    """Record an API call for rate limiting"""
    API_CALL_COUNTS.append(time.time())

def get_cache_key(question, article, function_type):
    """Generate cache key for Gemini responses"""
    combined = f"{function_type}:{question}:{article[:500]}"
    return hashlib.md5(combined.encode()).hexdigest()

def check_quota_limit():
    """Check if we're in quota waiting period"""
    current_time = time.time()
    if st.session_state.quota_wait_until > current_time:
        wait_time = int(st.session_state.quota_wait_until - current_time)
        return False, wait_time
    return True, 0

def apply_rate_limit():
    """Apply minimum delay between API calls (1 second)"""
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_api_call
    if time_since_last < 1.0:
        time.sleep(1.0 - time_since_last)
    st.session_state.last_api_call = time.time()

def get_fallback_response(question, article_text, response_type="general"):
    """Generate intelligent fallback response when quota exceeded"""
    fallback_responses = {
        "main_topic": f"Based on the article, the main focus appears to be on discussing key events and developments. The article presents information about the stated topic and provides context about related issues. Consider reading the full article for comprehensive details.",
        "verify_facts": f"To verify the claims in this article, I recommend: 1) Cross-reference statements with official sources, 2) Check fact-checking websites like Snopes or FactCheck.org, 3) Look for citations and evidence in the original article. The article's credibility analysis has already been provided above.",
        "red_flags": f"Red flag analysis (from our system): Review the red flags identified in the article analysis section above. Additionally, check for: proper sourcing, verifiable claims, balanced perspective, and citations from credible sources.",
        "general": f"I'm currently experiencing API quota limitations. Based on the article analysis already completed, the system has identified key patterns and credibility factors. Please try again in a few moments or review the analysis results provided above."
    }
    return fallback_responses.get(response_type, fallback_responses["general"])

# ========== ADVANCED FUNCTIONS ==========

def clean_text(text):
    """Text preprocessing - matches training data cleaning"""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_advanced_red_flags(text):
    """Advanced red flag detection with severity levels"""
    flags = []
    critical_count = 0
    high_count = 0
    
    original_text = text
    
    # 1. SENSATIONAL KEYWORDS (CRITICAL)
    sensational = ['shocking', 'exposed', 'scandal', 'unbelievable', 'bombshell', 
                   'conspiracy', 'coverup', 'emergency', 'alert', 'WARNING', 'URGENT']
    sensational_found = [w for w in sensational if w.lower() in text.lower()]
    if len(sensational_found) >= 2:
        flags.append(f"🔴 CRITICAL: Heavy sensational language - {', '.join(sensational_found[:3])}")
        critical_count += 1
    elif len(sensational_found) >= 1:
        flags.append(f"🟠 HIGH: Sensational words detected - {sensational_found[0]}")
        high_count += 1
    
    # 2. EXCESSIVE CAPS (HIGH)
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    if caps_ratio > 0.20:
        flags.append(f"🔴 CRITICAL: {caps_ratio*100:.1f}% uppercase text")
        critical_count += 1
    elif caps_ratio > 0.12:
        flags.append(f"🟠 HIGH: {caps_ratio*100:.1f}% uppercase letters")
        high_count += 1
    
    # 3. PUNCTUATION ABUSE (HIGH)
    exclaim_count = text.count('!')
    question_count = text.count('?')
    if exclaim_count >= 4 or question_count >= 3:
        flags.append(f"🔴 CRITICAL: Excessive punctuation (! :{exclaim_count}, ? :{question_count})")
        critical_count += 1
    elif exclaim_count >= 2 or question_count >= 2:
        flags.append(f"🟠 HIGH: Multiple punctuation marks")
        high_count += 1
    
    # 4. MISSING SOURCES (HIGH)
    sources = ['according', 'reported', 'confirmed', 'source', 'study', 'research', 'evidence']
    source_count = sum(text.lower().count(w) for w in sources)
    if source_count == 0 and len(text) > 500:
        flags.append("🟠 HIGH: No credible sources mentioned")
        high_count += 1
    
    # 5. CLICK-BAIT PATTERNS (MEDIUM)
    clickbait = ['won\'t believe', 'doctors hate', 'number 7', 'insiders reveal', 'you must',
                 'they don\'t want', 'secret ingredient']
    clickbait_found = [w for w in clickbait if w.lower() in text.lower()]
    if clickbait_found:
        flags.append(f"🟡 MEDIUM: Clickbait phrases detected")
    
    # 6. WORD FREQUENCY (MEDIUM)
    words = text.lower().split()
    from collections import Counter
    word_freq = Counter([w for w in words if len(w) > 3 and w.isalpha()])
    if word_freq:
        most_common_freq = word_freq.most_common(1)[0][1]
        if most_common_freq > len(words) * 0.12:
            flags.append(f"🟡 MEDIUM: Repetitive content detected")
    
    return flags, critical_count, high_count

def calculate_reading_difficulty(text):
    """Calculate Flesch-Kincaid Grade Level"""
    sentences = [s for s in text.split('.') if s.strip()]
    words = text.split()
    
    if not sentences or not words:
        return 0
    
    # Syllable count (simplified)
    def count_syllables(word):
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2:
            count += 1
            
        return max(1, count)
    
    syllables = sum(count_syllables(word) for word in words)
    
    grade = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
    return max(0, min(16, grade))

def detect_bias_indicators(text):
    """Detect political and ideological biases"""
    biases = {
        'Left-Leaning': ['progressive', 'liberal', 'woke', 'capitalism', 'privilege'],
        'Right-Leaning': ['woke', 'socialism', 'marxist', 'communist', 'globalist'],
        'Anti-Corporate': ['greed', 'exploitation', 'profit-driven', 'corporate'],
        'Anti-Government': ['tyranny', 'oppression', 'authority', 'regime']
    }
    
    detected = {}
    for bias_type, keywords in biases.items():
        count = sum(1 for kw in keywords if kw.lower() in text.lower())
        if count > 0:
            detected[bias_type] = count
    
    return detected

def advanced_sentiment_analysis(text):
    """Multi-dimensional sentiment analysis"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Emotion detection
    emotions = {
        'Anger': sum(text.lower().count(w) for w in ['angry', 'furious', 'outraged', 'enraged']),
        'Fear': sum(text.lower().count(w) for w in ['afraid', 'scared', 'terrified', 'fear']),
        'Joy': sum(text.lower().count(w) for w in ['happy', 'excited', 'thrilled', 'joyful']),
        'Disgust': sum(text.lower().count(w) for w in ['disgusting', 'vile', 'horrible', 'disgusted']),
        'Surprise': sum(text.lower().count(w) for w in ['shocked', 'surprised', 'amazed', 'astonished'])
    }
    
    dominant = max(emotions, key=emotions.get) if max(emotions.values()) > 0 else 'Neutral'
    
    sentiment_type = "Negative" if polarity < -0.1 else ("Positive" if polarity > 0.1 else "Neutral")
    
    return {
        'sentiment': sentiment_type,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'emotions': emotions,
        'dominant_emotion': dominant
    }

def advanced_credibility_scoring(prediction, confidence, red_flags_count, critical_count, high_count, sentiment, bias_count, readability, text):
    """Advanced credibility calculation - ACCURACY IMPROVED"""
    score = confidence
    
    # RED FLAGS (Heavy penalties)
    score -= (critical_count * 20)  # Each critical flag = -20
    score -= (high_count * 10)      # Each high flag = -10
    
    # SOURCES & CITATIONS (Most important indicator)
    sources = ['according', 'reported', 'confirmed', 'source', 'study', 'research', 'evidence', 'said', 'stated', 'published', 'found', 'shows', 'data']
    source_count = sum(text.lower().count(w) for w in sources)
    word_count = len(text.split())
    
    if source_count == 0 and word_count > 200:
        score -= 25  # Long article with NO sources = highly suspicious
    elif source_count == 0 and word_count > 100:
        score -= 15
    elif source_count == 0 and word_count > 50:
        score -= 8
    
    # SENTIMENT ANALYSIS
    sentiment_val = sentiment['sentiment']
    if sentiment_val == 'Negative':
        score -= 8   # Negative often indicates bias
    elif sentiment_val == 'Positive':
        score -= 5   # Positive emotion less suspicious
    
    # SUBJECTIVITY (Objective > Subjective)
    subjectivity = sentiment['subjectivity']
    if subjectivity > 0.8:
        score -= 12  # Almost pure opinion
    elif subjectivity > 0.65:
        score -= 6   # Moderately subjective
    
    # BIAS INDICATORS
    score -= (bias_count * 8)
    
    # READABILITY
    if readability < 3 or readability > 15:
        score -= 8
    
    # TEXT LENGTH
    if word_count < 30:
        score -= 15  # Too short
    elif word_count < 50:
        score -= 8
    
    # EXCESSIVE CAPS
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / len(text) if text else 0
    if caps_ratio > 0.30:
        score -= 15
    elif caps_ratio > 0.15:
        score -= 8
    
    return max(0, min(100, score))

def extract_advanced_entities(text):
    """Extract entities with categories"""
    entities = {
        'Persons': len(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)),
        'Locations': len(re.findall(r'\b(?:in|from|near|at)\s+[A-Z][a-z]+\b', text)),
        'Organizations': len(re.findall(r'\b(?:Inc|Co|CEO|Government|Ministry)\b', text)),
        'Numbers': len(re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)),
        'URLs': len(re.findall(r'https?://\S+', text))
    }
    return entities

# ========== CHATBOT FUNCTIONS ==========
def search_web_for_info(query, num_results=3):
    """Search web for information using requests and beautifulsoup"""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Using Google search via requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        search_url = f"https://www.google.com/search?q={query}"
        
        try:
            response = requests.get(search_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract search results
            results = []
            for result in soup.find_all('a', href=True):
                if '/url?q=' in result['href']:
                    url = result['href'].split('/url?q=')[1].split('&')[0]
                    title = result.get_text()
                    if title and url and len(results) < num_results:
                        results.append({'title': title, 'url': url})
            
            return results if results else ["Searching information online..."]
        except:
            return ["Could not fetch real-time search results. Opening web search..."]
    except:
        return ["Web search not available at the moment"]

def extract_relevant_sentences(article_text, query, num_sentences=3):
    """Extract sentences most relevant to the user's query"""
    sentences = [s.strip() for s in article_text.split('.') if s.strip()]
    
    # Score sentences by relevance to query
    query_words = set(query.lower().split())
    sentence_scores = []
    
    for i, sent in enumerate(sentences):
        sent_words = set(sent.lower().split())
        # Calculate relevance
        common = len(query_words & sent_words)
        score = common / max(len(query_words), 1)
        sentence_scores.append((i, sent, score))
    
    # Get top relevant sentences
    relevant = sorted(sentence_scores, key=lambda x: x[2], reverse=True)[:num_sentences]
    relevant = sorted(relevant, key=lambda x: x[0])  # Sort back to original order
    
    return [s[1] for s in relevant]

def generate_intelligent_response(user_question, article_text):
    """Generate intelligent response using ADVANCED LOCAL NLP (NO API NEEDED)"""
    
    q_lower = user_question.lower()
    
    # Extract article metrics
    sentences = [s.strip() for s in article_text.split('.') if s.strip()]
    word_count = len(article_text.split())
    char_count = len(article_text)
    
    # Get relevant sentences from article
    relevant_sents = extract_relevant_sentences(article_text, user_question, num_sentences=3)
    relevant_context = '. '.join(relevant_sents)
    
    # PRIORITY QUESTION TYPE DETECTION - CHECK SPECIFIC PATTERNS FIRST
    
    # Check for VERIFY/FACTS first (most specific)
    if any(word in q_lower for word in ['verify', 'fact', 'accurate', 'evidence', 'proof']) or ('claim' in q_lower and 'fact' in q_lower):
        sources_count = (article_text.count('According to') + article_text.count('said') + 
                        article_text.count('stated') + article_text.count('reported'))
        quotes_count = article_text.count('"')
        
        response = f"""**✓ Fact Verification Analysis (REAL-TIME):**

**Direct Evidence from Your Article:**
- Sources cited: {sources_count} references found
- Direct quotations: {quotes_count} quotes used
- Content analyzed: {word_count} words

**Your Article Content:**
> {relevant_context}

**How to Verify These Claims:**
1. **Check Primary Sources**: Look for official statements mentioned
2. **Cross-Reference Data**: Verify statistics on official websites
3. **Fact-Checking Sites**: FactCheck.org, Snopes.com, PolitiFact
4. **Publication Date**: When claimed vs actual timeline
5. **Source Authority**: Are sources credible institutions?

**Evidence Quality:**
- Has citations: {'✅ Yes' if sources_count > 0 else '❌ No'}
- Uses quotes: {'✅ Yes' if quotes_count > 0 else '❌ No'}
- Article length: {'✅ Adequate' if word_count > 250 else '⚠️ Limited'} ({word_count} words)

**Action**: Verify these specific claims on fact-checking websites."""

    # Check for RED FLAGS/CREDIBILITY (second priority)
    elif any(word in q_lower for word in ['flag', 'concern', 'credibility', 'fake', 'misinformation', 'warning', 'risk', 'red']) or ('problem' in q_lower and 'article' in q_lower):
        has_urls = 'http' in article_text or 'www.' in article_text
        all_caps_words = len([w for w in article_text.split() if w.isupper() and len(w) > 2])
        exclamation_marks = article_text.count('!')
        caps_ratio = len([c for c in article_text if c.isupper()]) / max(len(article_text), 1)
        
        red_flags = []
        if all_caps_words > 3:
            red_flags.append(f"🚨 Excessive ALL-CAPS: {all_caps_words} words (sensationalism)")
        if exclamation_marks > 5:
            red_flags.append(f"⚠️ Emotional Language: {exclamation_marks}! (manipulation)")
        if caps_ratio > 0.2:
            red_flags.append(f"⚠️ Cap Ratio: {caps_ratio*100:.1f}% (normal: 2-8%)")
        if not has_urls and word_count > 200:
            red_flags.append("🔗 No Links: Missing source citations")
        if word_count < 50:
            red_flags.append("📝 Too Short: <50 words")
            
        response = f"""**⚠️ Red Flags & Credibility (REAL-TIME ANALYSIS):**

**Issues Found in Your Article:**
{chr(10).join(['• ' + f for f in red_flags]) if red_flags else "✅ No major red flags detected"}

**Actual Article Stats:**
- Length: {word_count} words ({'✅ Good' if word_count > 250 else '⚠️ Short'})
- CAPS usage: {caps_ratio*100:.1f}% ({'✅ Normal' if 0.02 <= caps_ratio <= 0.08 else '⚠️ High'})
- Has links: {'✅ Yes' if has_urls else '❌ No'} (credibility marker)

**Article Content:**
> {relevant_context}

**Trust Level:** {'⚠️ VERIFY CLAIMS' if red_flags else '✅ APPEARS CREDIBLE'}

**Verify By:**
1. Check major outlets for same story
2. Verify each factual claim
3. Look for corroborating sources
4. Check publication date accuracy"""

    # Check for SOURCE/EVIDENCE (third priority)
    elif any(word in q_lower for word in ['source', 'evidence', 'reference', 'citation', 'credit']):
        source_count = (article_text.count('According to') + article_text.count('said') + 
                       article_text.count('research') + article_text.count('study'))
        
        response = f"""**📚 Sources & Evidence Analysis (REAL-TIME):**

**Source References Found:** {source_count} instances

**Your Article Content:**
> {relevant_context}

**Source Quality Check:**
1. Are sources clearly named and identifiable?
2. Are they credible institutions?
3. Can statistics be traced to origin?
4. Are links provided to original sources?

**Assessment:**
- Total references: {source_count} ({'✅ Well-sourced' if source_count > 3 else '⚠️ Limited sources'})
- Research depth: {'✅ Yes' if word_count > 300 else '⚠️ Limited'} ({word_count} words)

**Verify By:**
- Find original source documents
- Check if major outlets cite same sources
- Cross-reference data with official websites

**Note:** More sources = higher credibility potential"""

    # Check for SUMMARY  
    elif any(word in q_lower for word in ['summary', 'tldr', 'brief', 'short', 'overview', 'gist']):
        summary_sents = sentences[:4] if sentences else [article_text[:300]]
        summary_text = '. '.join(summary_sents)
        
        response = f"""**📰 Article Summary (YOUR CONTENT):**

{summary_text}

**Content Stats:**
- Word Count: {word_count} words
- Read Time: ~{max(1, word_count // 200)} minutes
- Sections: {len(sentences)} parts
- Depth: {'✅ Comprehensive' if word_count > 300 else '⚠️ Brief'}

**Main Claims:** See above

**Credibility Note:** 
{'✅ Sufficient length' if word_count > 250 else '⚠️ Quite short'} for thorough analysis

**Next Steps:** Read full article, verify claims, check red flags"""

    # MAIN TOPIC (default fallback)
    else:
        first_sents = '. '.join(sentences[:4]) if sentences else article_text[:300]
        response = f"""**📌 Main Topics in This Article:**

{first_sents}

**Article Details:**
- Length: {word_count} words
- Structure: {len(sentences)} sections
- Scope: {char_count} characters

**Credibility Note:**
- Length: {'✅ Good' if word_count > 300 else '⚠️ Short'} for detailed coverage
- Structure: {'✅ Well-organized' if len(sentences) > 5 else '⚠️ Simple'} reporting
- Cross-verify claims with reliable sources

**Recommendation**: Review credibility analysis to assess reliability."""

    return response

def generate_chatbot_response(user_question, article_text):
    """Generate chatbot response - HYBRID: Local NLP + Optional Gemini"""
    
    # PRIORITY 1: Use advanced local NLP (always works, no API issues)
    local_response = generate_intelligent_response(user_question, article_text)
    
    # PRIORITY 2: Try to enhance with web search if available
    q_lower = user_question.lower()
    web_search_keywords = ['verify', 'fact', 'true', 'current', 'latest', 'recent', 'now', 'today']
    should_search = any(word in q_lower for word in web_search_keywords)
    
    web_results_text = ""
    if should_search:
        try:
            search_results = search_web_for_info(user_question, num_results=2)
            if search_results and isinstance(search_results[0], dict):
                web_results_text = "\n\n🌐 **Additional Web Search Results:**\n"
                for idx, result in enumerate(search_results[:2], 1):
                    if isinstance(result, dict):
                        web_results_text += f"- [{result.get('title', 'Source')[:70]}]({result.get('url', '#')})\n"
        except:
            pass  # Silently fail - local response is sufficient
    
    # PRIORITY 3: Try Gemini if available (as enhancement, not requirement)
    try:
        can_call, current_count, limit = check_rate_limit()
        cache_key = get_cache_key(user_question, article_text, "chatbot")
        
        if cache_key in st.session_state.gemini_cache:
            gemini_response = st.session_state.gemini_cache[cache_key]
            return f"{local_response}{web_results_text}\n\n💡 **AI Enhancement (Gemini):**\n{gemini_response}"
        
        if can_call and GEMINI_AVAILABLE:
            try:
                record_api_call()
                apply_rate_limit()
                
                model = genai.GenerativeModel('gemini-2.0-flash')
                prompt = f"Briefly enhance this response (1-2 sentences): {user_question}\n\nArticle excerpt: {article_text[:1000]}"
                response_obj = model.generate_content(prompt, timeout=5)
                gemini_response = response_obj.text[:300]
                
                st.session_state.gemini_cache[cache_key] = gemini_response
                return f"{local_response}{web_results_text}\n\n💡 **AI Enhancement (Gemini):**\n{gemini_response}"
            except:
                pass  # Silently fail - local response is complete
    except:
        pass
    
    # Return local response (always available, always works)
    return f"{local_response}{web_results_text}"

def get_article_summary_gemini(article_text):
    """Generate article summary using Gemini API with rate limiting"""
    
    # Check rate limit
    can_call, _, _ = check_rate_limit()
    
    # Check cache
    cache_key = get_cache_key("summary", article_text, "summary")
    if cache_key in st.session_state.gemini_cache:
        return st.session_state.gemini_cache[cache_key]
    
    if can_call and GEMINI_AVAILABLE:
        try:
            record_api_call()
            apply_rate_limit()
            
            prompt = f"""Provide a concise 2-3 sentence summary of this article:

{article_text[:2000]}

Summary should be objective and factual."""
            
            model = genai.GenerativeModel('gemini-2.0-flash')
            response_obj = model.generate_content(prompt)
            summary = response_obj.text[:500]  # Limit to 500 chars
            
            # Cache it
            st.session_state.gemini_cache[cache_key] = summary
            return summary
            
        except Exception as e:
            # Fallback: Extract first sentences
            sentences = [s.strip() for s in article_text.split('.') if s.strip()]
            fallback = '. '.join(sentences[:3])
            if len(fallback) > 300:
                fallback = fallback[:300] + "..."
            return fallback
    else:
        # Rate limit exceeded, use local extraction
        sentences = [s.strip() for s in article_text.split('.') if s.strip()]
        fallback = '. '.join(sentences[:3])
        if len(fallback) > 300:
            fallback = fallback[:300] + "..."
        return fallback

def verify_claim_gemini(claim, article_context):
    """Verify claim using Gemini API with rate limiting"""
    
    # Check rate limit
    can_call, _, _ = check_rate_limit()
    
    # Check cache
    cache_key = get_cache_key(claim, article_context, "verify")
    if cache_key in st.session_state.gemini_cache:
        return st.session_state.gemini_cache[cache_key]
    
    if can_call and GEMINI_AVAILABLE:
        try:
            record_api_call()
            apply_rate_limit()
            
            prompt = f"""Analyze whether this claim is supported by the given article context.

Claim to verify: "{claim}"

Article context: {article_context[:2000]}

Provide a brief verification analysis including:
1. Is claim mentioned/supported in article?
2. Evidence level (strong/moderate/weak)
3. Suggestions for independent verification
Keep response to 3-4 sentences."""
            
            model = genai.GenerativeModel('gemini-2.0-flash')
            response_obj = model.generate_content(prompt)
            verification = response_obj.text[:500]
            
            # Cache it
            st.session_state.gemini_cache[cache_key] = verification
            return verification
            
        except Exception as e:
            # Fallback local analysis
            claim_in_article = claim.lower() in article_context.lower()
            evidence_words = ['prove', 'evidence', 'show', 'demonstrate', 'study']
            evidence_count = sum(1 for word in evidence_words if word in article_context.lower())
            
            fallback = f"""Claim verification (local analysis): "{claim}"
Mentioned in article: {'Yes' if claim_in_article else 'No'}
Supporting evidence found: {evidence_count} indicators
Recommendation: Cross-reference with official sources."""
            
            return fallback
    else:
        # Rate limit exceeded, use local analysis
        claim_in_article = claim.lower() in article_context.lower()
        evidence_words = ['prove', 'evidence', 'show', 'demonstrate', 'study']
        evidence_count = sum(1 for word in evidence_words if word in article_context.lower())
        
        fallback = f"""Claim verification (local analysis): "{claim}"
Mentioned in article: {'Yes' if claim_in_article else 'No'}
Supporting evidence found: {evidence_count} indicators
Recommendation: Cross-reference with official sources."""
        
        return fallback

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## 🎛️ ADVANCED CONTROL PANEL")
    st.markdown("---")
    
    try:
        conn = sqlite3.connect("advanced_analysis.db")
        df_stats = pd.read_sql_query("SELECT prediction FROM analysis", conn)
        
        total = len(df_stats)
        fake = len(df_stats[df_stats['prediction'] == 'FAKE'])
        real = len(df_stats[df_stats['prediction'] == 'REAL'])
        
        st.markdown(f'<div class="metric-advanced"><b>📊 Total Analyzed</b><p style="font-size: 24px; margin: 10px 0;">{total}</p></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card-fake"><b>🔴 Fake News</b><p style="font-size: 24px; margin: 10px 0;">{fake}</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card-real"><b>🟢 Real News</b><p style="font-size: 24px; margin: 10px 0;">{real}</p></div>', unsafe_allow_html=True)
        conn.close()
    except:
        st.info("No analysis data yet")
    
    st.markdown("---")
    if st.button("🗑️ Clear All Data"):
        conn = sqlite3.connect("advanced_analysis.db")
        conn.execute("DELETE FROM analysis")
        conn.commit()
        conn.close()
        st.success("Data cleared!")
        st.rerun()

# ========== SIDEBAR ENHANCEMENTS ==========
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    st.markdown(f'<div class="stat-card"><h4>🤖 Status</h4><p>🟢 ACTIVE</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    # Get or create analysis count
    try:
        conn = sqlite3.connect("advanced_analysis.db")
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM analysis")
        analysis_count = c.fetchone()[0]
        conn.close()
    except:
        analysis_count = 0
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("📝 Analyzed", analysis_count)
    with col_b:
        feedback_stats = get_feedback_stats()
        total_feedback = sum(feedback_stats.values())
        st.metric("📢 Feedback", total_feedback)
    
    st.markdown("---")
    st.markdown("### 💡 Quick Guide")
    st.info("""
    **How to use:**
    1. Paste your article in Tab 1
    2. Click "🚀 ANALYZE" 
    3. View results across tabs
    4. Use BATCH for bulk analysis
    5. Provide feedback to improve!
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Features")
    st.success("✅ Local NLP Analysis")
    st.success("✅ Web Search Support")
    st.success("✅ AI Enhancement")
    st.info("✨ User Feedback System")
    st.warning("⚡ Rate Limited")

# ========== MAIN INTERFACE ==========
st.markdown('<h1 class="header-main">📰 ADVANCED AI Fake News Detection</h1>', unsafe_allow_html=True)
st.write("**Enterprise-Grade ML + NLP Hybrid System** | Severity-Based Analysis")
st.markdown("---")

tabs = st.tabs(["🔬 ADVANCED ANALYZE", "📊 DETAILED METRICS", "🧠 EMOTION & BIAS", 
                "🔍 EXPLAINABILITY", "📈 DASHBOARD", "📂 BATCH"])

# ========== TAB 1: ADVANCED ANALYZE ==========
with tabs[0]:
    st.subheader("🔬 Advanced News Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        user_text = st.text_area("Paste Article:", height=300, label_visibility="collapsed")
    with col2:
        st.markdown("### 📋 Quick Stats")
        if user_text:
            st.write(f"**Words:** {len(user_text.split())}")
            st.write(f"**Characters:** {len(user_text)}")
            st.write(f"**Sentences:** {len(user_text.split('.'))}")
    
    if st.button("🚀 ANALYZE", use_container_width=True):
        if user_text.strip():
            with st.spinner("🔄 Processing..."):
                # Prediction with proper cleaning matching training data
                cleaned = clean_text(user_text)
                vectorized = vectorizer.transform([cleaned])
                pred = model.predict(vectorized)[0]
                
                # Proper confidence calculation (0-100 scale)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(vectorized)[0]
                    confidence = max(proba) * 100
                else:
                    # LinearRegression model
                    confidence = min(100, max(0, abs(pred - 0.5) * 200))
                
                result = "REAL" if round(pred) == 1 else "FAKE"
                
                # Advanced Analysis
                red_flags, critical, high = detect_advanced_red_flags(user_text)
                readability = calculate_reading_difficulty(user_text)
                sentiment_data = advanced_sentiment_analysis(user_text)
                biases = detect_bias_indicators(user_text)
                entities = extract_advanced_entities(user_text)
                
                credibility = advanced_credibility_scoring(
                    pred, confidence, len(red_flags), critical, high, 
                    sentiment_data, len(biases), readability, user_text
                )
                
                # Save to DB
                conn = sqlite3.connect("advanced_analysis.db")
                conn.execute("INSERT INTO analysis VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)",
                           (user_text[:200], result, confidence, len(red_flags), 0, credibility, datetime.datetime.now()))
                conn.commit()
                conn.close()
                
                # Save ALL data to session state
                st.session_state.article_text = user_text
                st.session_state.show_analysis = True
                st.session_state.full_analysis_data = {
                    'result': result,
                    'confidence': confidence,
                    'red_flags': red_flags,
                    'critical': critical,
                    'high': high,
                    'readability': readability,
                    'sentiment_data': sentiment_data,
                    'biases': biases,
                    'entities': entities,
                    'credibility': credibility,
                    'pred': pred
                }
                
                st.success("✅ Analysis Complete!")
    
    # ========== DISPLAY ANALYSIS RESULTS (FROM SESSION STATE) ==========
    if st.session_state.show_analysis and st.session_state.full_analysis_data:
        data = st.session_state.full_analysis_data
        result = data['result']
        confidence = data['confidence']
        readability = data['readability']
        credibility = data['credibility']
        red_flags = data['red_flags']
        sentiment_data = data['sentiment_data']
        biases = data['biases']
        entities = data['entities']
        critical = data['critical']
        pred = data['pred']
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        # Display results - 4 metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if result == "REAL":
                st.markdown('<div class="metric-card-real"><h2>✅<br>VERDICT</h2><p>REAL</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card-fake"><h2>❌<br>VERDICT</h2><p>FAKE</p></div>', unsafe_allow_html=True)
        
        with col2:
            if confidence > 70:
                st.markdown(f'<div class="metric-confidence-high"><h3>🟢</h3><p>{confidence:.1f}%</p><h3 style="font-size:11px;margin:0">High</h3></div>', unsafe_allow_html=True)
            elif confidence > 40:
                st.markdown(f'<div class="metric-confidence-med"><h3>🟡</h3><p>{confidence:.1f}%</p><h3 style="font-size:11px;margin:0">Medium</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-confidence-low"><h3>🔴</h3><p>{confidence:.1f}%</p><h3 style="font-size:11px;margin:0">Low</h3></div>', unsafe_allow_html=True)
        
        with col3:
            grade = "Easy" if readability < 8 else "Moderate" if readability < 12 else "Hard"
            st.markdown(f'<div class="metric-advanced"><h3>📖</h3><p>{grade}</p><h3 style="font-size:11px;margin:0">G{readability:.0f}</h3></div>', unsafe_allow_html=True)
        
        with col4:
            if credibility > 70:
                st.markdown(f'<div class="metric-confidence-high"><h3>✓</h3><p>{credibility:.1f}%</p><h3 style="font-size:11px;margin:0">Quality</h3></div>', unsafe_allow_html=True)
            elif credibility > 40:
                st.markdown(f'<div class="metric-confidence-med"><h3>⚠</h3><p>{credibility:.1f}%</p><h3 style="font-size:11px;margin:0">Verify</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-confidence-low"><h3>✗</h3><p>{credibility:.1f}%</p><h3 style="font-size:11px;margin:0">Low</h3></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # VERDICT EXPLANATION
        st.subheader("📋 Detailed Analysis Report")
        
        if result == "FAKE":
            st.markdown("""
            ### 🔴 WHY THIS IS CLASSIFIED AS FAKE NEWS:
            """)
            
            # Reasons for fake classification
            reasons = []
            
            if len(red_flags) > 0:
                reasons.append(f"**Misinformation Signals Detected ({len(red_flags)})**: Multiple patterns typical of misinformation were found")
            
            if critical > 0:
                reasons.append(f"**{critical} CRITICAL red flags**: Heavy sensational language, excessive punctuation, or misleading patterns")
            
            if sentiment_data['sentiment'] in ['Negative', 'Positive'] and sentiment_data['subjectivity'] > 0.6:
                reasons.append(f"**Emotional Language**: The text uses strong {sentiment_data['sentiment'].lower()} sentiment with highly subjective tone")
            
            sources = ['according', 'reported', 'confirmed', 'source', 'study', 'research', 'evidence']
            source_count = sum(st.session_state.article_text.lower().count(w) for w in sources)
            if source_count == 0:
                reasons.append("**No Credible Sources**: Article contains no references to verified sources or evidence")
            
            if len(biases) > 0:
                reasons.append(f"**Ideological Bias Detected**: Shows patterns of {', '.join(biases.keys())}")
            
            if credibility < 50:
                reasons.append(f"**Low Overall Credibility Score**: {credibility:.1f}% - Multiple credibility factors align with misinformation patterns")
            
            for i, reason in enumerate(reasons, 1):
                st.markdown(f"• {reason}")
            
            st.warning("⚠️ **Recommendation**: Cross-reference with official sources and fact-checking websites before sharing")
            
        else:  # REAL
            st.markdown("""
            ### 🟢 WHY THIS IS CLASSIFIED AS REAL NEWS:
            """)
            
            # Reasons for real classification
            reasons = []
            
            if len(red_flags) < 2:
                reasons.append("**Minimal Misinformation Signals**: Few or no patterns typical of fake news")
            
            sources = ['according', 'reported', 'confirmed', 'source', 'study', 'research', 'evidence']
            source_count = sum(st.session_state.article_text.lower().count(w) for w in sources)
            if source_count > 0:
                reasons.append(f"**Credible Sources Referenced**: Article cites or references verified information sources ({source_count} instances)")
            
            if sentiment_data['sentiment'] == 'Neutral' or sentiment_data['subjectivity'] < 0.5:
                reasons.append("**Objective Tone**: Article maintains neutral, factual tone without excessive emotion")
            
            if len(biases) == 0:
                reasons.append("**No Detected Bias**: Content doesn't show ideological or political bias patterns")
            
            if credibility >= 70:
                reasons.append(f"**High Credibility Score**: {credibility:.1f}% - Multiple factors indicate reliable journalism")
            
            if 100 > len(st.session_state.article_text.split()) > 50:
                reasons.append("**Adequate Content Length**: Article has sufficient detail for credible reporting")
            
            for i, reason in enumerate(reasons, 1):
                st.markdown(f"• {reason}")
            
            st.success("✓ **Status**: This appears to be credible, factual news from reliable patterns")
        
        st.markdown("---")
        
        # Red Flags Section
        st.subheader("🚩 Detailed Red Flags Analysis")
        if red_flags:
            st.markdown(f"**Found {len(red_flags)} warning indicators:**")
            for flag in red_flags:
                st.markdown(f"• {flag}")
        else:
            st.success("✅ No misinformation patterns detected")
        
        st.markdown("---")
        
        # Entities
        st.subheader("🏷️ Detected Entities")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="metric-advanced"><h3>👤</h3><p>{entities["Persons"]}</p><h3 style="font-size:11px;margin:0">Persons</h3></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-advanced"><h3>📍</h3><p>{entities["Locations"]}</p><h3 style="font-size:11px;margin:0">Locations</h3></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-advanced"><h3>🏢</h3><p>{entities["Organizations"]}</p><h3 style="font-size:11px;margin:0">Orgs</h3></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-advanced"><h3>🔢</h3><p>{entities["Numbers"]}</p><h3 style="font-size:11px;margin:0">Numbers</h3></div>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<div class="metric-advanced"><h3>🔗</h3><p>{entities["URLs"]}</p><h3 style="font-size:11px;margin:0">URLs</h3></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ========== INTEGRATED BATCH SECTION ==========
        st.subheader("📂 AI Analysis Assistant (Smart Local + Gemini)")
        
        # Show rate limit status
        can_call, current_count, limit = check_rate_limit()
        
        # Quick question buttons
        st.markdown("#### Quick Questions:")
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        
        with quick_col1:
            if st.button("❓ Main Topic", key="main_topic_btn", use_container_width=True):
                quick_question = "What are the main topics and key points discussed in this article?"
                with st.spinner("🤖 Analyzing..."):
                    response = generate_chatbot_response(quick_question, st.session_state.article_text)
                    st.session_state.chat_history.append({'type': 'user', 'content': quick_question})
                    st.session_state.chat_history.append({'type': 'bot', 'content': response})
                st.rerun()  # Force re-render to show new messages
        
        with quick_col2:
            if st.button("✓ Verify Facts", key="verify_btn", use_container_width=True):
                quick_question = "Are the main claims factually accurate? What evidence supports them?"
                with st.spinner("🤖 Verifying..."):
                    response = generate_chatbot_response(quick_question, st.session_state.article_text)
                    st.session_state.chat_history.append({'type': 'user', 'content': quick_question})
                    st.session_state.chat_history.append({'type': 'bot', 'content': response})
                st.rerun()  # Force re-render to show new messages
        
        with quick_col3:
            if st.button("⚠️ Red Flags", key="flags_btn", use_container_width=True):
                quick_question = "What credibility concerns or potential misinformation signals exist in this article?"
                with st.spinner("🤖 Analyzing..."):
                    response = generate_chatbot_response(quick_question, st.session_state.article_text)
                    st.session_state.chat_history.append({'type': 'user', 'content': quick_question})
                    st.session_state.chat_history.append({'type': 'bot', 'content': response})
                st.rerun()  # Force re-render to show new messages
        
        st.markdown("---")
        
        # Custom question
        st.markdown("#### Ask Your Own Question:")
        col_q1, col_q2 = st.columns([4, 1])
        
        with col_q1:
            custom_question = st.text_input("Your question about the article:", placeholder="Ask anything...", key="custom_q_input")
        
        with col_q2:
            ask_custom = st.button("🚀 Ask", key="ask_custom_btn", use_container_width=True)
        
        if ask_custom and custom_question.strip():
            with st.spinner("🤖 Thinking..."):
                response = generate_chatbot_response(custom_question, st.session_state.article_text)
                st.session_state.chat_history.append({'type': 'user', 'content': custom_question})
                st.session_state.chat_history.append({'type': 'bot', 'content': response})
            st.rerun()  # Force re-render to show new messages
        
        st.markdown("---")
        
        # Display chat history (AFTER all inputs, at bottom)
        if st.session_state.chat_history:
            # Session statistics
            session_duration = int(time.time() - st.session_state.session_start_time)
            minutes = session_duration // 60
            seconds = session_duration % 60
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.markdown(f'<div class="stat-card"><h4>💬 Messages</h4><p>{len(st.session_state.chat_history)}</p></div>', unsafe_allow_html=True)
            with col_stat2:
                st.markdown(f'<div class="stat-card"><h4>⏱️ Session</h4><p>{minutes}m {seconds}s</p></div>', unsafe_allow_html=True)
            with col_stat3:
                feedback_stats = get_feedback_stats()
                helpful_count = feedback_stats.get('helpful', 0)
                st.markdown(f'<div class="stat-card"><h4>👍 Helpful</h4><p>{helpful_count}</p></div>', unsafe_allow_html=True)
            
            st.markdown("#### 📋 Chat History:")
            for idx, msg in enumerate(st.session_state.chat_history):
                if msg['type'] == 'user':
                    st.markdown(f'<div class="chat-user-msg">👤 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bot-msg">🤖 <b>Assistant:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
                    
                    # Add feedback buttons for bot responses
                    feedback_key = f"msg_{idx}"
                    if feedback_key not in st.session_state.feedback_given:
                        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 4])
                        with fb_col1:
                            if st.button("👍 Helpful", key=f"like_{idx}", use_container_width=True):
                                save_feedback(st.session_state.chat_history[idx-1]['content'], msg['content'], "helpful")
                                st.session_state.feedback_given[feedback_key] = True
                                st.success("Thanks for your feedback!")
                                st.rerun()
                        with fb_col2:
                            if st.button("👎 Not Helpful", key=f"dislike_{idx}", use_container_width=True):
                                save_feedback(st.session_state.chat_history[idx-1]['content'], msg['content'], "not_helpful")
                                st.session_state.feedback_given[feedback_key] = True
                                st.info("We'll improve!")
                                st.rerun()
                    else:
                        st.caption("✓ Feedback recorded")
            st.markdown("---")
        
        # Feedback Summary Section
        st.markdown("#### 📊 Your Feedback Summary:")
        feedback_stats = get_feedback_stats()
        
        if feedback_stats:
            col_fb1, col_fb2, col_fb3 = st.columns(3)
            with col_fb1:
                helpful_count = feedback_stats.get('helpful', 0)
                st.metric("👍 Helpful", helpful_count)
            with col_fb2:
                not_helpful_count = feedback_stats.get('not_helpful', 0)
                st.metric("👎 Not Helpful", not_helpful_count)
            with col_fb3:
                total_feedback = sum(feedback_stats.values())
                if total_feedback > 0:
                    helpful_percent = (helpful_count / total_feedback) * 100
                    st.metric("✅ Satisfaction", f"{helpful_percent:.0f}%")
        else:
            st.info("💡 No feedback recorded yet. Your feedback helps us improve!")
        
        st.markdown("---")
        
        # Export and Manage options
        col_export, col_clear = st.columns(2)
        
        with col_export:
            if st.button("📥 Export Chat", use_container_width=True):
                if st.session_state.chat_history:
                    # Create formatted chat text
                    chat_text = "=== CHAT HISTORY ===\n"
                    chat_text += f"Session Duration: {session_duration} seconds\n"
                    chat_text += f"Total Messages: {len(st.session_state.chat_history)}\n\n"
                    
                    for msg in st.session_state.chat_history:
                        if msg['type'] == 'user':
                            chat_text += f"👤 User: {msg['content']}\n\n"
                        else:
                            chat_text += f"🤖 Assistant: {msg['content']}\n\n"
                        chat_text += "-" * 60 + "\n\n"
                    
                    st.download_button(
                        label="Download Chat (.txt)",
                        data=chat_text,
                        file_name=f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key="download_chat"
                    )
        
        with col_clear:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.feedback_given = {}
                st.session_state.session_start_time = time.time()
                st.success("✨ Chat cleared!")
                st.rerun()

# ========== TAB 2: DETAILED METRICS ==========
with tabs[1]:
    st.subheader("📊 Detailed Text Metrics")
    if 'user_text' in locals() and user_text:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            words = user_text.split()
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            st.markdown(f'<div class="metric-advanced"><h3>📝</h3><p>{avg_word_len:.2f}</p><h3 style="font-size:11px;margin:0">Word Len</h3></div>', unsafe_allow_html=True)
        
        with col2:
            unique_words = len(set(w.lower() for w in words))
            st.markdown(f'<div class="metric-advanced"><h3>🔤</h3><p>{unique_words}</p><h3 style="font-size:11px;margin:0">Unique</h3></div>', unsafe_allow_html=True)
        
        with col3:
            diversity = unique_words / len(words) if words else 0
            st.markdown(f'<div class="metric-advanced"><h3>📊</h3><p>{diversity:.0%}</p><h3 style="font-size:11px;margin:0">Diversity</h3></div>', unsafe_allow_html=True)
        
        # Top words chart
        st.subheader("Top 15 Keywords")
        word_freq = Counter([w.lower() for w in words if len(w) > 3])
        top_words = word_freq.most_common(15)
        
        if top_words:
            words_names, words_counts = zip(*top_words)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(words_names, words_counts, color='#1f77b4')
            st.pyplot(fig)
    else:
        st.info("👈 Analyze an article first")

# ========== TAB 3: EMOTION & BIAS ==========
with tabs[2]:
    st.subheader("🧠 Emotion & Bias Detection")
    if 'sentiment_data' in locals():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💭 Emotional Profile")
            emotion_data = sentiment_data['emotions']
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(emotion_data.keys(), emotion_data.values(), color=['#ff6b6b', '#ee5a6f', '#f1e15b', '#a8e6cf', '#ffd3b6'])
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 🎭 Bias Indicators")
            if biases:
                for bias_type, score in biases.items():
                    st.markdown(f'<div class="warning-box"><b>{bias_type}</b><p>{score} mentions detected</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box"><p>✅ No significant bias detected</p></div>', unsafe_allow_html=True)
    else:
        st.info("👈 Analyze an article first")

# ========== TAB 4: EXPLAINABILITY ==========
with tabs[3]:
    st.subheader("🔍 Model Explainability")
    if 'user_text' in locals() and user_text:
        # Top influential words
        cleaned = re.sub(r'[^\w\s]', '', user_text.lower())
        vectorized = vectorizer.transform([cleaned])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = vectorized.toarray()[0]
        top_indices = np.argsort(tfidf_scores)[-12:][::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = [(feature_names[i], tfidf_scores[i]) for i in top_indices if tfidf_scores[i] > 0]
        if top_features:
            names, scores = zip(*top_features)
            ax.barh(names, scores, color='#ff7f0e')
            st.pyplot(fig)
    else:
        st.info("👈 Analyze an article first")

# ========== TAB 5: DASHBOARD ==========
with tabs[4]:
    st.subheader("📈 Analytics Dashboard")
    try:
        conn = sqlite3.connect("advanced_analysis.db")
        df = pd.read_sql_query("SELECT prediction, credibility FROM analysis", conn)
        
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                pred_counts = df['prediction'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', colors=['#dc3545', '#28a745'])
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(df['credibility'], bins=20, color='#1f77b4', edgecolor='black')
                ax.set_xlabel("Credibility Score")
                st.pyplot(fig)
        conn.close()
    except:
        st.info("No data available")

# ========== TAB 6: BATCH PROCESS ==========
with tabs[5]:
    st.subheader("📂 Batch Processing")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded and st.button("Process All"):
        df_upload = pd.read_csv(uploaded)
        text_col = st.selectbox("Text Column:", df_upload.columns)
        
        results = []
        progress = st.progress(0)
        
        for idx, row in df_upload.iterrows():
            text = str(row[text_col])
            cleaned = re.sub(r'[^\w\s]', '', text.lower())
            vectorized = vectorizer.transform([cleaned])
            pred = model.predict(vectorized)[0]
            confidence = abs((pred - 0.5) * 200)
            result = "REAL" if round(pred) == 1 else "FAKE"
            
            results.append({"Text": text[:50], "Verdict": result, "Confidence": f"{confidence:.1f}%"})
            progress.progress((idx + 1) / len(df_upload))
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        
        csv = results_df.to_csv(index=False)
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")

st.markdown("---")
