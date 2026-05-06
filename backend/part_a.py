import os
import warnings
import pandas as pd
import networkx as nx
import itertools
from thefuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Silence tokenization warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.filterwarnings("ignore", category=FutureWarning)

print("=========================================================")
print(" PART A: POSTGRESQL IDENTITY RESOLUTION ENGINE           ")
print("=========================================================")

load_dotenv()
DATABASE_URI = os.getenv("DATABASE_URL")

if not DATABASE_URI:
    print("[Error] DATABASE_URL not found in .env file.")
    exit()

print("[System] Connecting to NeonDB...")
engine = create_engine(DATABASE_URI)

print("\n[System] Loading NLP Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. BULLETPROOF COLUMN MAPPER
# This guarantees that varying database column names are perfectly aligned for the graph
SCHEMA_MAP = {
    'shop_establishment': {
        'record_id': 'record_id', 'business_name': 'business_name', 'phone': 'phone', 
        'pincode': 'pincode', 'address': 'address', 'pan': 'pan', 'gstin': 'gstin'
    },
    'bbmp_trade_license': {
        'license_identifier': 'record_id', 'trade_name': 'business_name', 'mobile_number': 'phone', 
        'postal_code': 'pincode', 'location': 'address', 'tax_id': 'pan', 'gst_number': 'gstin'
    },
    'kspcb_pollution': {
        'consent_no': 'record_id', 'industry_name': 'business_name', 'contact_number': 'phone', 
        'zip_code': 'pincode', 'premise_area': 'address', 'gst_number': 'gstin', 'pan': 'pan'
    },
    'bescom_electricity': {
        'account_no': 'record_id', 'customer_name': 'business_name', 'telephone': 'phone', 
        'pin_code': 'pincode', 'street_ward': 'address', 'permanent_account_number': 'pan', 'gst_number': 'gstin'
    }
}

master_columns = ["record_id", "business_name", "phone", "pincode", "address", "pan", "gstin", "source_db"]

def calculate_hybrid_score(name1, name2, nlp_model):
    fuzzy_score = fuzz.token_set_ratio(name1, name2)
    if fuzzy_score >= 85 or fuzzy_score < 50:
        return fuzzy_score
    set1, set2 = set(name1.split()), set(name2.split())
    diff1, diff2 = " ".join(set1 - set2), " ".join(set2 - set1)
    if diff1 and diff2:
        embeddings = nlp_model.encode([diff1, diff2])
        semantic_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        semantic_score = max(0, semantic_sim * 100)
        return fuzzy_score + ((100 - fuzzy_score) * (semantic_score / 100))
    return fuzzy_score

def clean_val(val):
    if pd.isna(val) or val is None: return ""
    cleaned = str(val).split('.')[0].strip()
    if cleaned.upper() in ['NAN', 'NULL', 'NONE', 'NA', '']: return ""
    return cleaned

# --- 2. DATA INGESTION & RENAMING ---
tables_to_load = {
    'Shop_Establishment': 'shop_establishment',
    'BBMP_Trade': 'bbmp_trade_license',
    'KSPCB_Pollution': 'kspcb_pollution',
    'BESCOM_Power': 'bescom_electricity'
}

standardized_dfs = []
for db_name, table_name in tables_to_load.items():
    try:
        df = pd.read_sql_table(table_name, engine)
        
        # Rename columns using the hardcoded dictionary map
        if table_name in SCHEMA_MAP:
            df = df.rename(columns=SCHEMA_MAP[table_name])
            
        for col in master_columns:
            if col not in df.columns: 
                df[col] = ""
                
        df['source_db'] = db_name
        standardized_dfs.append(df[master_columns])
    except Exception as e:
        print(f"  [Warning] Skipping {table_name}: {e}")

master_df = pd.concat(standardized_dfs, ignore_index=True)
master_df['pincode'] = master_df['pincode'].apply(clean_val)
master_df['phone'] = master_df['phone'].apply(clean_val)
master_df['pan'] = master_df['pan'].apply(lambda x: clean_val(x).upper())
master_df['gstin'] = master_df['gstin'].apply(lambda x: clean_val(x).upper())

# --- 3. IDENTITY RESOLUTION ---
print("\n[System] Building Graphs and Resolving Connected Identities...\n")
auto_complete_registry = []
human_review_queue = []
ubid_counter = 1000

grouped_by_pincode = master_df.groupby('pincode')

for pincode, group in grouped_by_pincode:
    if not str(pincode).strip(): continue 
    
    G = nx.Graph()
    
    # Add nodes and anchor edges
    for _, row in group.iterrows():
        rec_id = row['record_id']
        if not rec_id: continue
        G.add_node(rec_id, **row.to_dict())
        
        if row['phone']: G.add_edge(rec_id, f"PHONE_{row['phone']}")
        if row['pan']:   G.add_edge(rec_id, f"PAN_{row['pan']}")
        if row['gstin']: G.add_edge(rec_id, f"GSTIN_{row['gstin']}")

    # Extract Connected Clusters
    clusters = list(nx.connected_components(G))
    for cluster in clusters:
        record_ids = [n for n in cluster if not str(n).startswith(("PHONE_", "PAN_", "GSTIN_"))]
        
        # ==========================================================
        # RULE: IGNORE ISOLATED RECORDS (Only keep connected graphs)
        # ==========================================================
        if len(record_ids) <= 1:
            continue
            
        c_pan, c_gstin, c_phone = "", "", ""
        for rid in record_ids:
            node_data = G.nodes.get(rid, {})
            if not c_pan and node_data.get('pan'): c_pan = node_data.get('pan')
            if not c_gstin and node_data.get('gstin'): c_gstin = node_data.get('gstin')
            if not c_phone and node_data.get('phone'): c_phone = node_data.get('phone')
        
        master_name = max([str(G.nodes[r].get('business_name', '')) for r in record_ids], key=len, default="")
        master_address = max([str(G.nodes[r].get('address', '')) for r in record_ids], key=len, default="")
        gstin_or_pan = c_pan if c_pan else c_gstin

        # Calculate NLP Score if there is no PAN/GSTIN
        avg_score = 100.0
        if not gstin_or_pan:
            scores = [calculate_hybrid_score(str(G.nodes[i1].get('business_name', '')).lower(), 
                                             str(G.nodes[i2].get('business_name', '')).lower(), model) 
                      for i1, i2 in itertools.combinations(record_ids, 2)]
            avg_score = sum(scores) / len(scores) if scores else 0

        payload = {
            'ubid': f"KAR-{ubid_counter}",
            'gstin_or_pan': gstin_or_pan,
            'business_name': master_name,
            'phone': c_phone,
            'pincode': pincode,
            'address': master_address,
            'merged_records': " | ".join(record_ids),
            'confidence_score': round(avg_score, 2),
            'databases_linked': len(record_ids) # <-- NEW CONNECTION FLAG
        }

        print("-" * 65)
        print(f"🔗 CONNECTED GRAPH: {len(record_ids)} databases linked -> {payload['merged_records']}")
        
        # --- ROUTING LOGIC ---
        if c_pan or c_gstin:
            payload['linkage_evidence'] = f"Direct Connection via Tax ID ({gstin_or_pan})"
            auto_complete_registry.append(payload)
            print(f"✅ AUTO-APPROVED: Shared Tax ID")
        elif avg_score >= 85:
            payload['linkage_evidence'] = f"Shared Phone ({c_phone}) + High NLP Match ({round(avg_score, 1)}%)"
            auto_complete_registry.append(payload)
            print(f"✅ AUTO-APPROVED: High NLP Match ({round(avg_score, 1)}%)")
        else:
            payload['ubid'] = f"PENDING-KAR-{ubid_counter}"
            payload['linkage_evidence'] = f"Shared Phone ({c_phone}) + Low NLP Match ({round(avg_score, 1)}%)"
            human_review_queue.append(payload)
            print(f"⚠️ HUMAN REVIEW: Low NLP Match ({round(avg_score, 1)}%)")
            
        ubid_counter += 1

print("\n" + "="*65)
print("[System] Saving Merged Results to NeonDB...")

auto_df = pd.DataFrame(auto_complete_registry)
human_df = pd.DataFrame(human_review_queue)

if not auto_df.empty:
    auto_df.to_sql('db_auto_complete_ubids', engine, if_exists='replace', index=False)
else:
    pd.DataFrame(columns=master_columns).to_sql('db_auto_complete_ubids', engine, if_exists='replace', index=False)

if not human_df.empty:
    human_df.to_sql('db_human_review_queue', engine, if_exists='replace', index=False)
else:
    pd.DataFrame(columns=master_columns).to_sql('db_human_review_queue', engine, if_exists='replace', index=False)

print("-> Part A Complete!")
print(f"   - Saved {len(auto_df)} Connected Records to Auto-Approved")
print(f"   - Saved {len(human_df)} Connected Records to Human Review Queue")