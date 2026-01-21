import streamlit as st
import pandas as pd
from data.preprocess_data import process_data
from Recommendation_models.rating_based_recommendation import get_top_rated_items
from Recommendation_models.content_based_filtering import content_based_recommendation
from Recommendation_models.collaborative_based_filtering import collaborative_filtering_recommendations

st.set_page_config(
    page_title="AI-Enabled Recommendation Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_type' not in st.session_state:
    st.session_state.user_type = None

@st.cache_data
def load_data():
    return pd.read_csv("data/clean_data.csv")
data = process_data(load_data())

st.markdown("""
<style>
    /* Global Dark Theme */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0.3);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f1419 100%);
    }
    /* Text Colors */
    p, span, label, div, h1, h2, h3 {
        color: #ffffff !important;
    }
    h2 {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 30px !important;
        margin-bottom: 20px !important;
        font-weight: 700 !important;
    }
    .login-title {
        font-size: 36px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
    }
    .login-subtitle {
        text-align: center;
        color: #9ca3af !important;
        margin-bottom: 10px;
        font-size: 20px;
    }
    /* Divider */
    .divider {
        display: flex;
        align-items: center;
        text-align: center;
        margin: 20px 0;
    }
    .divider::before,
    .divider::after {
        content: '';
        flex: 1;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }
    .divider span {
        padding: 0 15px;
        color: #9ca3af !important;
        font-size: 14px;
        font-weight: 500;
    }
    /* Login Page Specific Styles */
    .login-page [data-testid="stNumberInput"] {
        margin-bottom: 20px;
    }
    .login-page [data-testid="stNumberInput"] input {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 2px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        padding: 16px 20px !important;
        font-size: 18px !important;
        font-weight: 500 !important;
        text-align: center !important;
        width: 100% !important;
    }
    .login-page [data-testid="stNumberInput"] input:focus {
        border: 2px solid #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2) !important;
        background: rgba(255, 255, 255, 0.12) !important;
    }
    .login-page button[kind="primary"] {
        width: 100% !important;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 16px 24px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .login-page button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }
    .login-page button[kind="secondary"] {
        width: 100% !important;
        background: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        padding: 16px 24px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    .login-page button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    .login-label {
        font-size: 14px;
        font-weight: 600;
        color: #ffffff !important;
        margin-bottom: 8px;
        text-align: left;
        display: block;
    }
    /* Product Card */
    .product-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 10px;
        margin-bottom: 20px;
        height: 100%;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border 0.2s ease;
    }
    .product-card:hover {
        transform: translateY(-4px);
        border: 1px solid rgba(99, 102, 241, 0.5);
    }
    .image-container {
        width: 100%;
        height: 200px;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 12px;
        background: rgba(255, 255, 255);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .image-container img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    .product-title {
        font-size: 15px;
        font-weight: 600;
        color: #ffffff !important;
        margin-bottom: 8px;
        line-height: 1.4;
        height: 42px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .product-brand {
        font-size: 13px;
        color: #ffffff !important;
        margin-bottom: 10px;
        font-weight: 500;
    }
    .product-stats {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    /* Rating Colors */
    .rating-excellent {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff !important;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    .rating-good {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff !important;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    .rating-average {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: #000000 !important;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(251, 191, 36, 0.3);
    }
    .rating-below-average {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: #ffffff !important;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(249, 115, 22, 0.3);
    }
    .rating-poor {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #ffffff !important;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
    }
    .reviews {
        color: #ffffff !important;
        font-size: 12px;
    }
    /* Main App Input Fields */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        color: #000000 !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
    }
    [data-testid="stTextInput"] input::placeholder {
        color: #6b7280 !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus {
        border: 2px solid #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }
    /* Buttons */
    button[kind="secondary"] {
        background-color: rgba(99, 102, 241, 0.2) !important;
        color: #ffffff !important;
        border: 1px solid rgba(99, 102, 241, 0.5) !important;
    }
    button[kind="secondary"]:hover {
        background-color: rgba(99, 102, 241, 0.3) !important;
        border: 1px solid rgba(99, 102, 241, 0.8) !important;
    }
    /* Sidebar Button - Logout */
    [data-testid="stSidebar"] button {
        background-color: rgba(239, 68, 68, 0.2) !important;
        color: #ffffff !important;
        border: 1px solid rgba(239, 68, 68, 0.5) !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: rgba(239, 68, 68, 0.4) !important;
        border: 1px solid rgba(239, 68, 68, 0.8) !important;
    }
    /* Messages */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 12px !important;
    }
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 12px !important;
    }
    .stWarning, .stInfo {
        background: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

def get_rating_class(rating):
    rating = float(rating)
    if rating >= 4.5:
        return "rating-excellent"
    elif rating >= 4.0:
        return "rating-good"
    elif rating >= 3.0:
        return "rating-average"
    elif rating >= 2.0:
        return "rating-below-average"
    else:
        return "rating-poor"

def display_products(df):
    """Display products in a properly aligned grid"""
    if df.empty:
        st.warning("No recommendations available.")
        return
    df = df.reset_index(drop=True)
    cols = st.columns(4)
    for idx, (_, row) in enumerate(df.iterrows()):
        col_idx = idx % 4
        with cols[col_idx]:
            rating_class = get_rating_class(row['Rating'])
            st.markdown(f"""
                <div class='product-card'>
                    <div class='image-container'>
                        <img src='{row["ImageURL"]}' alt='{row["Name"]}'>
                    </div>
                    <div class='product-title'>{row['Name']}</div>
                    <div class='product-brand'>Brand: {row['Brand']}</div>
                    <div class='product-stats'>
                        <span class='{rating_class}'>⭐ {row['Rating']}</span>
                        <span class='reviews'>📝 {row['ReviewCount']} Review</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

def partial_product_search(data, query, top_n=10):
    """Search for products by partial name match"""
    if not query.strip():
        return pd.DataFrame()
    query_lower = query.lower().strip()
    matching_products = data[data['Name'].str.lower().str.contains(query_lower, na=False)]
    if not matching_products.empty:
        matching_products = matching_products.sort_values(
            by=['Rating', 'ReviewCount'], 
            ascending=[False, False]
        )
        return matching_products.drop_duplicates(subset='Name').head(top_n).reset_index(drop=True)
    try:
        result = content_based_recommendation(data=data, item_name=query, top_n=top_n)
        return result.reset_index(drop=True) if not result.empty else pd.DataFrame()
    except:
        return pd.DataFrame()

# LOGIN PAGE
if not st.session_state.logged_in:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="login-page">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("""
            <div class='login-glass-container'>
                <div class='login-title'>🛍️ Welcome!</div>
                <div class='login-subtitle'>AI-Enabled Recommendation Engine</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<span class='login-label'>🔐 Existing User Login</span>", unsafe_allow_html=True)
        input_user_id = st.number_input(
            "User ID",
            min_value=1,
            value=1,
            label_visibility="collapsed",
            key="user_id_input"
        )
        if st.button("🔓 Login", use_container_width=True, type="primary", key="login_btn"):
            if input_user_id in data['ID'].values:
                st.session_state.logged_in = True
                st.session_state.user_id = input_user_id
                st.session_state.user_type = "Existing"
                st.rerun()
            else:
                st.error(f"❌ User ID {input_user_id} doesn't exist in the system.")
        st.markdown("""<div class='divider'><span>OR</span></div>""", unsafe_allow_html=True)
        if st.button("Continue as New User", use_container_width=True, type="secondary", key="guest_btn"):
            st.session_state.logged_in = True
            st.session_state.user_id = 0
            st.session_state.user_type = "Guest"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# MAIN APPLICATION
st.sidebar.title("🔐 User Profile")
if st.session_state.user_type == "Guest":
    st.sidebar.success("👤 New User (Guest)")
else:
    st.sidebar.success(f"👤 User ID: {st.session_state.user_id}")
if st.sidebar.button("🚪 Logout", use_container_width=True, key="logout_btn"):
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.user_type = None
    st.rerun()

top_n = st.sidebar.slider(
    "Items to Display",
    min_value=4,
    max_value=100,
    value=12,
    step=4
)

user_id = st.session_state.user_id
recent_items = pd.DataFrame()

if user_id > 0:
    st.sidebar.markdown("### 🕘 Recently Viewed")
    recent_items = (
        data[data["ID"] == user_id]
        .drop_duplicates(subset="Name")
        .head(5)
    )
    if recent_items.empty:
        st.sidebar.info("No recent activity.")
    else:
        for _, row in recent_items.iterrows():
            st.sidebar.markdown(f"- {row['Name']}")

st.title("🛍️ AI-Enabled Recommendation Engine")
search_query = st.text_input(
    "🔍 Search for a product",
    placeholder="Type product name"
)

if search_query.strip():
    st.subheader(f"🔍 Results for '{search_query}'")
    search_results = partial_product_search(data, search_query, top_n=top_n)
    if search_results.empty:
        st.warning(f"No products found matching '{search_query}'. Try different keywords.")
    else:
        st.info(f"Found {len(search_results)} product(s) matching your search.")
        display_products(search_results)
elif user_id == 0:
    st.subheader("🔥 Top Rated Products")
    st.markdown("*Discover our highest-rated products!*")
    top_items = get_top_rated_items(data, top_n=top_n)
    top_items = top_items.reset_index(drop=True)
    display_products(top_items)
else:
    user_history = data[data["ID"] == user_id]
    if user_history.empty:
        st.warning("⚠️ No previous activity found. Showing top-rated items!")
        top_items = get_top_rated_items(data, top_n=top_n)
        top_items = top_items.reset_index(drop=True)
        display_products(top_items)
    else:
        st.subheader(f"👋 Welcome Back, User {user_id}!")
        st.markdown(f"*We found {len(user_history)} interactions in your history*")
        st.markdown("### ✨ Based on Your Interests")
        all_recommendations = pd.DataFrame()
        num_recent_items = len(recent_items)
        recs_per_item = top_n // num_recent_items
        remaining = top_n % num_recent_items
        for idx, (_, item) in enumerate(recent_items.iterrows()):
            items_to_get = recs_per_item + (1 if idx < remaining else 0)
            recs = content_based_recommendation(
                data=data,
                item_name=item["Name"],
                top_n=items_to_get
            )
            all_recommendations = pd.concat([all_recommendations, recs], ignore_index=True)
        all_recommendations = all_recommendations.drop_duplicates(subset="Name").head(top_n)
        all_recommendations = all_recommendations.reset_index(drop=True)
        display_products(all_recommendations)
        st.markdown("### 🤝 Users Like You... Also Liked")
        cf_items = collaborative_filtering_recommendations(
            data=data,
            target_user_id=user_id,
            top_n = top_n
        )
        cf_items = cf_items.reset_index(drop=True)
        display_products(cf_items)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #9ca3af; padding: 20px;'>
    <p>Powered by AI-Enabled Recommendation Engine | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)