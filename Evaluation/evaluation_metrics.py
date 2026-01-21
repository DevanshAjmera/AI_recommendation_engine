import pandas as pd
import numpy as np
from data.preprocess_data import process_data
from Recommendation_models.content_based_filtering import content_based_recommendation
from Recommendation_models.collaborative_based_filtering import collaborative_filtering_recommendations
from Recommendation_models.rating_based_recommendation import get_top_rated_items


def evaluate_content_based(data, item_name=None, top_n=10):
    if item_name is None:
        item_name = data['Name'].iloc[0]
    
    item_data = data[data['Name'] == item_name].iloc[0]
    relevant = set(data[(data['Category'] == item_data['Category']) | (data['Brand'] == item_data['Brand'])]['Name']) - {item_name}
    
    recs = content_based_recommendation(data, item_name, top_n)
    if recs.empty:
        return None
    
    true_positives = len(set(recs['Name']) & relevant)
    precision = true_positives / top_n
    recall = true_positives / len(relevant) if len(relevant) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nContent-Based: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    return {'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_collaborative(data, num_users=10, top_n=10):
    user_counts = data.groupby('ID').size()
    test_users = user_counts[user_counts >= 10].index[:num_users]
    
    precisions, recalls = [], []
    
    for user_id in test_users:
        highly_rated = data[(data['ID'] == user_id) & (data['Rating'] >= 4.0)]
        if len(highly_rated) < 3:
            continue
        
        test_size = max(1, int(len(highly_rated) * 0.3))
        test_items = highly_rated.sample(n=test_size, random_state=user_id)
        test_ids = set(test_items['ProdID'])
        
        train_data = data[~((data['ID'] == user_id) & (data['ProdID'].isin(test_ids)))]
        
        try:
            recs = collaborative_filtering_recommendations(train_data, user_id, top_n)
            if recs.empty:
                continue
            
            rec_ids = set()
            for _, row in recs.iterrows():
                match = data[data['Name'] == row['Name']]['ProdID'].values
                if len(match) > 0:
                    rec_ids.add(match[0])
            
            hits = len(rec_ids & test_ids)
            precisions.append(hits / top_n)
            recalls.append(hits / len(test_ids))
        except:
            continue
    
    if not precisions:
        return None
    
    precision, recall = np.mean(precisions), np.mean(recalls)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Collaborative: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    return {'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_rating_based(data, top_n=10, min_rating=4.0, min_reviews=10):
    recs = get_top_rated_items(data, top_n)
    if recs.empty:
        return None
    
    quality = len(recs[(recs['Rating'] >= min_rating) & (recs['ReviewCount'] >= min_reviews)]) / top_n
    avg_rating = recs['Rating'].mean()
    avg_reviews = recs['ReviewCount'].mean()
    
    print(f"Rating-Based: Quality={quality:.3f}, Avg Rating={avg_rating:.3f}, Avg Reviews={avg_reviews:.1f}")
    return {'quality': quality, 'avg_rating': avg_rating, 'avg_reviews': avg_reviews}


def evaluate_all(data, top_n=10, num_users=10):
    print(f"\n{'='*60}\nEVALUATION RESULTS (Top-{top_n})\n{'='*60}")
    
    cb = evaluate_content_based(data, top_n=top_n)
    cf = evaluate_collaborative(data, num_users=num_users, top_n=top_n)
    rb = evaluate_rating_based(data, top_n=top_n)
    
    print(f"{'='*60}\n")
    return {'content': cb, 'collaborative': cf, 'rating': rb}

# To test -> python -m Evaluation.evaluation_metrics
if __name__ == "__main__":
    raw_data = pd.read_csv("data/clean_data.csv")
    data = process_data(raw_data)
    print(f"Data: {len(data)} rows, {data['ID'].nunique()} users, {data['ProdID'].nunique()} products")
    
    results = evaluate_all(data, top_n=10, num_users=10)