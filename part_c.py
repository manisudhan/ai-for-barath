import os
import re
import pandas as pd
import numpy as np
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Silence warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.filterwarnings("ignore")

print("=========================================================")
print(" PART C: GOOGLE-STYLE SEMANTIC SEARCH ENGINE             ")
print("=========================================================")

# 1. Load the Master Database
DB_FILE = "db_final_master_registry.csv"
try:
    print(f"[System] Loading Database: {DB_FILE}...")
    df = pd.read_csv(DB_FILE).fillna("")
    df['pincode'] = df['pincode'].astype(str).str.replace(".0", "", regex=False)
except FileNotFoundError:
    print(f"[Error] {DB_FILE} not found. Please run Part B first.")
    exit()

# 2. Build the "Search Index" for each row
# We combine all data into a single readable paragraph so the AI understands the context of the row.
print("[System] Building Omni-Search Index...")
df['search_context'] = (
    "UBID is " + df['ubid'] + ". " +
    "Business Name is " + df['business_name'] + ". " +
    "Address is " + df['address'] + " in pincode " + df['pincode'] + ". " +
    "GSTIN or PAN is " + df['gstin_or_pan'] + ". " +
    "Status is " + df['activity_status'] + " because " + df['status_reason']
)

# 3. Load the Local NLP Model (Reusing the one from Part A!)
print("[System] Loading Semantic Search AI (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("[System] Vectorizing database (creating embeddings)...")
db_embeddings = model.encode(df['search_context'].tolist())

def execute_google_style_search(user_query):
    """Combines Exact Token Matching with Semantic Vector Search."""
    
    query_lower = user_query.lower()
    scores = np.zeros(len(df))
    
    # --- A. Exact Match Boost (For IDs like KAR-1000 or Pincodes) ---
    # If the user types a specific UBID or Pincode, heavily prioritize it.
    extracted_ubids = re.findall(r'kar-\d+', query_lower)
    extracted_pincodes = re.findall(r'\b\d{6}\b', query_lower)
    
    for idx, row in df.iterrows():
        # Boost if UBID matches
        if any(u in row['ubid'].lower() for u in extracted_ubids):
            scores[idx] += 2.0 
        # Boost if Pincode matches
        if any(p in str(row['pincode']) for p in extracted_pincodes):
            scores[idx] += 1.0
        # Boost if specific Status requested
        if "active" in query_lower and row['activity_status'].lower() == "active":
            scores[idx] += 0.5
        elif "closed" in query_lower and row['activity_status'].lower() == "closed":
            scores[idx] += 0.5
        elif "dormant" in query_lower and row['activity_status'].lower() == "dormant":
            scores[idx] += 0.5

    # --- B. Semantic Similarity (Meaning Computation) ---
    # This understands that "factory" is similar to "manufacturing", etc.
    query_embedding = model.encode([user_query])
    semantic_scores = cosine_similarity(query_embedding, db_embeddings)[0]
    
    # Combine exact match boosts with semantic meaning
    final_scores = scores + semantic_scores
    
    # --- C. Rank and Return Results ---
    df['search_score'] = final_scores
    # Filter out highly irrelevant results (score threshold)
    results = df[df['search_score'] > 0.3].sort_values(by='search_score', ascending=False)
    
    return results

# 4. Interactive CLI Loop
print("\n[System] Omni-Search Engine Ready. Type 'exit' to quit.")
print("-" * 60)

while True:
    query = input("\n🔍 Search (e.g., 'gstin of UBID KAR-1000' or 'active shops in 560001'):\n> ")
    
    if query.lower() in ['exit', 'quit']:
        print("Shutting down engine...")
        break
        
    if not query.strip():
        continue

    # Execute Search
    final_data = execute_google_style_search(query)
    
    # Display Results
    print("=" * 100)
    if final_data.empty:
        print("❌ No relevant businesses found.")
    else:
        top_result = final_data.iloc[0]
        
        # Act like Google: Show a "Featured Snippet" direct answer for the top result
        print(f"✨ Top Result Match [{top_result['ubid']}]:")
        print(f"   Business : {top_result['business_name']}")
        print(f"   PAN/GSTIN: {top_result['gstin_or_pan'] if top_result['gstin_or_pan'] else 'Not Registered'}")
        print(f"   Status   : {top_result['activity_status'].upper()} ({top_result['status_reason']})")
        print("-" * 100)
        
        print(f"✅ Found {len(final_data)} total matches. (Showing Top 5):\n")
        display_cols = ['ubid', 'business_name', 'pincode', 'gstin_or_pan', 'activity_status']
        print(final_data[display_cols].head(5).to_string(index=False))
        
    print("=" * 100)