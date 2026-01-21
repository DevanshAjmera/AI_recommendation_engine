import pandas as pd
from .content_based_filtering import content_based_recommendation
from .collaborative_based_filtering import collaborative_filtering_recommendations

def hybrid_recommendation(data: pd.DataFrame, item_name: str, user_id: int, top_n: int = 10):
    content_rec = content_based_recommendation(data, item_name, top_n)
    collaborative_rec = collaborative_filtering_recommendations(data, user_id, top_n)

    hybrid_df = pd.concat([content_rec, collaborative_rec]).drop_duplicates().head(2*top_n)
    return hybrid_df.reset_index(drop=True)

#To test -> python -m Recommendation_models.hybrid
if __name__ == "__main__":
    import pandas as pd
    from data.preprocess_data import process_data
    
    raw_data = pd.read_csv("data/clean_data.csv")
    data = process_data(raw_data)

    #item_name = "OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath"
    item_name = "Pure Gold Bitter Orange Essential Oil, 100% Natural & Undiluted, 60ml"
    target_user_id = 5
    top_n = 5

    hybrid_rec = hybrid_recommendation(data, item_name, target_user_id, top_n)

    print(hybrid_rec)