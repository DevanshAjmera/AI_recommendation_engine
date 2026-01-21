import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def collaborative_filtering_recommendations(data: pd.DataFrame, target_user_id: int, top_n: int = 10):
    user_item_matrix = data.pivot_table(
        index='ID',
        columns='ProdID',
        values='Rating',
        aggfunc='mean'
    ).fillna(0)

    if target_user_id not in user_item_matrix.index:
        return pd.DataFrame()

    user_similarity = cosine_similarity(user_item_matrix)

    target_index = user_item_matrix.index.get_loc(target_user_id)
    user_similarities = user_similarity[target_index]

    similar_users_indices = user_similarities.argsort()[::-1][1:]  # sort similar users excluding self

    recommended_items = []
    for user_idx in similar_users_indices:
        rated_by_similar_user = user_item_matrix.iloc[user_idx]
        not_rated_by_target = ((rated_by_similar_user != 0) & (user_item_matrix.iloc[target_index] == 0))
        recommended_items.extend(user_item_matrix.columns[not_rated_by_target][:top_n])

    recommended_items = list(set(recommended_items))[:top_n]

    return data[data['ProdID'].isin(recommended_items)].head(top_n)[['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']].reset_index(drop=True)

#Example usage -> python -m Recommendation_models.collaborative_based_filtering
if __name__ == "__main__":
    import pandas as pd
    from data.preprocess_data import process_data

    raw_data = pd.read_csv("data/clean_data.csv")
    data = process_data(raw_data)

    target_user_id = 4
    print(collaborative_filtering_recommendations(data, target_user_id))