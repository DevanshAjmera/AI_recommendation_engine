import pandas as pd

def get_top_rated_items(data: pd.DataFrame, top_n: int = 10):
    
    avg_ratings = data.groupby(['Name','ReviewCount','Brand','ImageURL'])['Rating'].mean().reset_index()
    # Sort by Rating first, then ReviewCount
    top_rated = avg_ratings.sort_values(by=['Rating', 'ReviewCount'], ascending=[False, False])
    return top_rated.head(top_n).reset_index(drop=True)


# Get top rated items -> python -m Recommendation_models.rating_based_recommendation
if __name__ == "__main__":
    import pandas as pd
    from data.preprocess_data import process_data

    raw_data = pd.read_csv("data/clean_data.csv")
    data = process_data(raw_data)

    print(get_top_rated_items(data))
