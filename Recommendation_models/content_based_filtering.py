import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendation(data: pd.DataFrame, item_name: str, top_n: int = 10):
    if item_name not in data['Name'].values:
        return pd.DataFrame()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(data['Tags'])
    cosine_similarity_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    item_index = data[data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarity_content[item_index]))
    similar_prod = sorted(similar_items, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_indices = [x[0] for x in similar_prod]

    return data.iloc[recommended_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']].reset_index(drop=True)

# TO test the system  -> python -m Recommendation_models.content_based_filtering
if __name__ == "__main__":
    import pandas as pd
    from data.preprocess_data import process_data

    raw_data = pd.read_csv("data/clean_data.csv")
    data = process_data(raw_data)
  
    item_name = "Pure Gold Bitter Orange Essential Oil, 100% Natural & Undiluted, 60ml"
    result = content_based_recommendation(data, item_name, top_n=5)
    print(result)