import pandas as pd
import numpy as np
from data.preprocess_data import process_data
from Recommendation_models.collaborative_based_filtering import collaborative_filtering_recommendations

raw_data = pd.read_csv("data/clean_data.csv")
data = process_data(raw_data)


def evaluate_collaborative(data, num_users=10, top_n=10):
    print(f"\n=== COLLABORATIVE FILTERING EVALUATION (Top {top_n}) ===")
    
    # Get users with 10+ ratings
    user_counts = data.groupby('ID').size()
    test_users = user_counts[user_counts >= 10].index[:num_users]
    
    if len(test_users) == 0:
        print("❌ No users with enough ratings!")
        return None
    
    all_precision, all_recall = [], []
    
    for user_id in test_users:
        user_data = data[data['ID'] == user_id]
        highly_rated = user_data[user_data['Rating'] >= 4.0]
        
        if len(highly_rated) < 3:
            continue
        
        # Hide 30% of highly-rated items
        test_size = max(1, int(len(highly_rated) * 0.3))
        test_items = highly_rated.sample(n=test_size, random_state=user_id)
        test_prod_ids = set(test_items['ProdID'].values)
        
        # Remove hidden items from training
        train_data = data[~((data['ID'] == user_id) & (data['ProdID'].isin(test_prod_ids)))]
        
        # Get recommendations
        recs = collaborative_filtering_recommendations(train_data, user_id, top_n)
        if recs.empty:
            continue
        
        # Match to product IDs
        rec_prod_ids = set()
        for _, row in recs.iterrows():
            match = data[data['Name'] == row['Name']]['ProdID'].values
            if len(match) > 0:
                rec_prod_ids.add(match[0])
        
        # Calculate metrics
        hits = len(rec_prod_ids & test_prod_ids)
        all_precision.append(hits / top_n)
        all_recall.append(hits / len(test_prod_ids))
    
    if not all_precision:
        print("❌ No valid evaluations!")
        return None
    
    precision = np.mean(all_precision)
    recall = np.mean(all_recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 METRICS@K={top_n} (averaged over {len(all_precision)} users):")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1-Score:  {f1:.3f}")
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Run this to test :- python -m Evaluation.evaluation_collaborative
if __name__ == "__main__":
    evaluate_collaborative(data, num_users=10, top_n=10)
