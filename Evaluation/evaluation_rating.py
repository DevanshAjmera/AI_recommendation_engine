import pandas as pd
from data.preprocess_data import process_data
from Recommendation_models.rating_based_recommendation import get_top_rated_items

raw_data = pd.read_csv("data/clean_data.csv")
data = process_data(raw_data)


def evaluate_rating_based(data, top_n=10, min_rating=4.0, min_reviews=10):
    print(f"\n=== RATING-BASED EVALUATION (Top {top_n}) ===")
    
    # Get recommendations
    recs = get_top_rated_items(data, top_n)
    
    if recs.empty:
        print("❌ No recommendations!")
        return None
    
    # Check quality: How many have rating >= min_rating AND reviews >= min_reviews?
    high_quality = recs[(recs['Rating'] >= min_rating) & (recs['ReviewCount'] >= min_reviews)]
    
    quality_score = len(high_quality) / top_n
    avg_rating = recs['Rating'].mean()
    avg_reviews = recs['ReviewCount'].mean()
    
    print(f"\n📊 METRICS@K={top_n}:")
    print(f"   Quality Score: {quality_score:.3f}  ({len(high_quality)}/{top_n} items with {min_rating}+ stars & {min_reviews}+ reviews)")
    print(f"   Avg Rating:    {avg_rating:.3f}")
    print(f"   Avg Reviews:   {avg_reviews:.1f}")
    
    print(f"\nTop 3 recommendations:")
    for idx, row in recs.head(3).iterrows():
        print(f"  {idx+1}. {row['Name'][:50]}... ({row['Rating']}★, {row['ReviewCount']} reviews)")
    
    return {'quality_score': quality_score, 'avg_rating': avg_rating, 'avg_reviews': avg_reviews}

# To run -> python -m Evaluation.evaluation_rating 
if __name__ == "__main__":
    evaluate_rating_based(data, top_n=10)
