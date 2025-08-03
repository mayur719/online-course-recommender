import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import dok_matrix

# === Load & Prepare Data ===
#@st.cache_data
def load_data():
    #  Load dataset (update path if needed)
    df = pd.read_excel("C:\\Users\\venka\\OneDrive\\Desktop\\OCR\\online_course_recommendation_v2.xlsx")

    #  Encode user and course
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df['user_idx'] = user_encoder.fit_transform(df['user_id'])
    df['item_idx'] = item_encoder.fit_transform(df['course_id'])

    df_courses = df[['user_id', 'course_id', 'rating', 'user_idx', 'item_idx']].drop_duplicates('course_id')
    df_courses = df_courses.set_index('course_id').loc[item_encoder.classes_].reset_index()

    # Build collaborative filtering matrix
    n_users = df['user_idx'].nunique()
    n_items = df['item_idx'].nunique()
    rating_matrix = dok_matrix((n_users, n_items), dtype=np.float32)
    for row in df.itertuples():
        rating_matrix[row.user_idx, row.item_idx] = row.rating

    svd = TruncatedSVD(n_components=30, random_state=42)
    user_factors = svd.fit_transform(rating_matrix)
    item_factors = svd.components_
    pred_matrix = np.dot(user_factors, item_factors)

    # Content-based similarity
    content_features = ['user_id', 'course_id', 'rating', 'user_idx', 'item_idx']
    scaler = MinMaxScaler()
    course_features_scaled = scaler.fit_transform(df_courses[content_features])
    course_sim_matrix = cosine_similarity(course_features_scaled)

    return df, df_courses, user_encoder, item_encoder, pred_matrix, course_sim_matrix

df, df_courses, user_encoder, item_encoder, pred_matrix, course_sim_matrix = load_data()

# === Hybrid Recommendation Function ===
def hybrid_recommend(user_id_raw, top_n=5, alpha=0.6, beta=0.4):
    if user_id_raw not in user_encoder.classes_:
        return pd.DataFrame(columns=['course_id', 'course_name', 'hybrid_score'])

    user_idx = user_encoder.transform([user_id_raw])[0]
    collaborative_scores = pred_matrix[user_idx]

    user_rated_items = df[df['user_idx'] == user_idx]['item_idx'].tolist()
    if len(user_rated_items) == 0:
        content_scores = np.zeros_like(collaborative_scores)
    else:
        content_scores = course_sim_matrix[user_rated_items].mean(axis=0)

    collaborative_scores = (collaborative_scores - collaborative_scores.min()) / (collaborative_scores.max() - collaborative_scores.min() + 1e-8)
    content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)

    hybrid_scores = alpha * collaborative_scores + beta * content_scores

    rated_items = set(user_rated_items)
    all_indices = np.arange(len(hybrid_scores))
    unseen_indices = [i for i in all_indices if i not in rated_items]

    top_indices = np.argsort(hybrid_scores[unseen_indices])[::-1][:top_n]
    top_course_ids = item_encoder.inverse_transform(np.array(unseen_indices)[top_indices])
    top_scores = hybrid_scores[unseen_indices][top_indices]

    result_df = pd.DataFrame({
        'course_id': top_course_ids,
        'hybrid_score': top_scores
    })

    #  Add course name to output
    result_df = result_df.merge(df[['course_id', 'course_name']], on='course_id', how='left')
    result_df = result_df.drop_duplicates('course_id')
    return result_df[['course_id', 'course_name', 'hybrid_score']]

# === Streamlit UI ===
st.set_page_config(page_title="Hybrid Course Recommender", layout="wide")
st.title(" Hybrid Course Recommendation System")

user_input = st.text_input("Enter your User ID (e.g., 123)", value="123")
top_n = st.slider("Number of courses to recommend", 1, 10, 5)

if st.button("Recommend Courses"):
    try:
        user_id = int(user_input)
        recommendations = hybrid_recommend(user_id_raw=user_id, top_n=top_n)
        if recommendations.empty:
            st.warning("No recommendations found. Either new user or invalid user ID.")
        else:
            st.success(f"Top {top_n} Course Recommendations:")
            st.dataframe(recommendations.reset_index(drop=True), use_container_width=True)
    except ValueError:
        st.error("Please enter a valid numeric User ID.")
