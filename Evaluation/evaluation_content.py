#Precision for Content-Based Filtering (Simple Approach)
#For content-based filtering, precision measures: "Of my top 10 similar items, how many actually belong to the same category/brand as the input item?"

import pandas as pd
from data.preprocess_data import process_data

raw_data = pd.read_csv("data/clean_data.csv")
data = process_data(raw_data)

from Recommendation_models.content_based_filtering import content_based_recommendation

def evaluate_content_based_metrics(data, item_name, top_n=10):
    print(f"\n=== FULL METRICS: CONTENT-BASED (Top {top_n}) ===")
    
    # 1. CHECK ITEM EXISTS
    item_matches = data[data['Name'] == item_name]
    if item_matches.empty:
        print(f"❌ Item '{item_name}' NOT FOUND!")
        return None
    
    item_data = item_matches.iloc[0]
    item_category = item_data.get('Category', None)
    item_brand = item_data.get('Brand', 'Unknown')
    
    print(f"🎯 Input: '{item_name}'")
    print(f"   Category: {item_category}, Brand: {item_brand}")
    
    # 2. GROUND TRUTH: ALL relevant items in dataset
    relevant_items = set()
    if item_category:
        relevant_items.update(data[data['Category'] == item_category]['Name'].values)
    relevant_items.update(data[data['Brand'] == item_brand]['Name'].values)
    total_relevant = len(relevant_items) - 1  # Remove self
    print(f"📚 Total relevant items in dataset: {total_relevant}")
    
    # 3. Get recommendations
    recs = content_based_recommendation(data, item_name, top_n)
    if recs.empty:
        print("❌ No recommendations!")
        return None
    
    recommended_names = set(recs['Name'].values)
    
    # 4. Calculate ALL 3 METRICS
    true_positives = len(recommended_names & relevant_items)
    
    precision = true_positives / top_n
    recall = true_positives / total_relevant if total_relevant > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 5. Results
    print(f"\n📊 METRICS@K={top_n}:")
    print(f"   Precision: {precision:.3f}  ({true_positives}/{top_n})")
    print(f"   Recall:    {recall:.3f}  ({true_positives}/{total_relevant})")
    print(f"   F1-Score:  {f1:.3f}")
    
    print(f"\n✅ Matches: {list(recommended_names & relevant_items)[:3]}...")
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Run this to test :- python -m Evaluation.evaluation_content
if __name__ == "__main__":
    raw_data = pd.read_csv("data/clean_data.csv")
    data = process_data(raw_data)
    
    item_name = data['Name'].iloc[0]  # Guaranteed to exist
    evaluate_content_based_metrics(data, item_name, top_n=10)


#Your TF-IDF tags work PERFECTLY for similarity (Precision=1.0)
#But there are 123 total relevant items → You're only finding 8.1% of them

#Real-world use: "Show me 10 similar items"
#Users care about: Top 10 being perfect (✅ YES!)
#Users don't care about: Missing 113/123 other items