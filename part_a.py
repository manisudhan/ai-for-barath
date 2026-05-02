import os
import warnings
import json
import pandas as pd
import networkx as nx
import itertools
from thefuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Silence tokenization warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.filterwarnings("ignore", category=FutureWarning)

print("=========================================================")
print(" PART A: IDENTITY RESOLUTION & SCHEMA MAPPING ENGINE     ")
print("=========================================================")

print("\n[System] Loading NLP Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

master_definitions = {
    "record_id": "unique record database ID application number rr number license identifier",
    "business_name": "business company trade industry establishment name",
    "owner_name": "owner applicant consumer customer person name",
    "phone": "phone mobile contact telephone number",
    "pincode": "pincode pin code postal zip code",
    "address": "address location street premise area ward",
    "pan": "pan permanent account number tax id",
    "gstin": "gstin gst goods and services tax number"
}

master_columns = list(master_definitions.keys())
master_embeddings = model.encode(list(master_definitions.values()))

def map_schema_intelligently(df, source_db_name):
    """Maps arbitrary dataframes to the master schema using NLP and Heuristics."""
    incoming_columns = df.columns.tolist()
    print(f"  -> Ingesting & Mapping: {source_db_name}")
    
    clean_incoming = [str(col).replace('_', ' ') for col in incoming_columns]
    incoming_embeddings = model.encode(clean_incoming)
    similarity_matrix = cosine_similarity(incoming_embeddings, master_embeddings)
    
    rename_mapping = {}
    assigned_master_cols = set() 
    
    matches = []
    for i, incoming_col in enumerate(incoming_columns):
        best_match_idx = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i][best_match_idx]
        best_master_col = master_columns[best_match_idx]
        matches.append((best_score, incoming_col, best_master_col))
        
    matches.sort(reverse=True, key=lambda x: x[0])
    
    for score, incoming_col, master_col in matches:
        if score > 0.45 and master_col not in assigned_master_cols:
            rename_mapping[incoming_col] = master_col
            assigned_master_cols.add(master_col)
            
    for col in incoming_columns:
        if col not in rename_mapping:
            valid_data = df[col].dropna()
            if not valid_data.empty:
                sample_val = str(valid_data.iloc[0]).strip().split('.')[0]
                
                if re.fullmatch(r'\d{10}', sample_val) and 'phone' not in assigned_master_cols:
                    rename_mapping[col] = 'phone'
                    assigned_master_cols.add('phone')
                elif re.fullmatch(r'\d{6}', sample_val) and 'pincode' not in assigned_master_cols:
                    rename_mapping[col] = 'pincode'
                    assigned_master_cols.add('pincode')
                elif re.search(r'[A-Za-z]+-\d+', sample_val) and 'record_id' not in assigned_master_cols:
                    rename_mapping[col] = 'record_id'
                    assigned_master_cols.add('record_id')

    standardized_df = df.rename(columns=rename_mapping)
    standardized_df = standardized_df.loc[:, ~standardized_df.columns.duplicated()]
    
    for col in master_columns:
        if col not in standardized_df.columns:
            standardized_df[col] = ""
            
    final_columns = master_columns + ['source_db']
    standardized_df['source_db'] = source_db_name 
    return standardized_df[final_columns]

def calculate_hybrid_score(name1, name2, nlp_model):
    """Combines fuzzy matching with semantic vectorization for mismatched tokens."""
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
    return str(val).split('.')[0].strip() if val else ""

# --- RUN LOGIC ---
files_to_load = {
    'Shop_Establishment': 'shop_establishment.csv',
    'BBMP_Trade': 'bbmp_trade_license.csv',
    'BESCOM_Power': 'bescom_electricity.csv',
    'BWSSB_Water': 'bwssb_water.csv',
    'FSSAI_Food': 'fssai_food_safety.csv',
    'KSPCB_Pollution': 'kspcb_pollution.csv'
}

standardized_dfs = []
for db_name, filename in files_to_load.items():
    try:
        raw_df = pd.read_csv(filename).fillna("")
        standardized_dfs.append(map_schema_intelligently(raw_df, db_name))
    except FileNotFoundError:
        pass

if not standardized_dfs:
    print("[Error] No master CSV files found. Exiting.")
    exit()

master_df = pd.concat(standardized_dfs, ignore_index=True).fillna("")

print("\n[System] Building Graphs and Resolving Identities...")
auto_merged_registry = []
human_review_queue = []
ubid_counter = 1000

grouped_by_pincode = master_df.groupby('pincode')

for pincode, group in grouped_by_pincode:
    if not pincode: continue 
    
    G = nx.Graph()
    for _, row in group.iterrows():
        rec_id = row['record_id']
        if not rec_id: continue
        G.add_node(rec_id, **row.to_dict())
        
        if clean_val(row['phone']): G.add_edge(rec_id, f"PHONE_{clean_val(row['phone'])}")
        if clean_val(row['pan']):   G.add_edge(rec_id, f"PAN_{clean_val(row['pan'])}")
        if clean_val(row['gstin']): G.add_edge(rec_id, f"GSTIN_{clean_val(row['gstin'])}")

    records_list = group.to_dict('records')
    for i, row1 in enumerate(records_list):
        id1 = row1['record_id']
        for j in range(i + 1, len(records_list)):
            id2 = records_list[j]['record_id']
            owner1, owner2 = clean_val(row1.get('owner_name', '')).lower(), clean_val(records_list[j].get('owner_name', '')).lower()
            if owner1 and owner2 and owner1 == owner2 and len(owner1) > 3:
                G.add_edge(id1, id2)
                continue
            
            add1, add2 = clean_val(row1.get('address', '')).lower(), clean_val(records_list[j].get('address', '')).lower()
            if add1 and add2 and len(add1) > 10 and len(add2) > 10:
                if fuzz.token_set_ratio(add1, add2) >= 90:
                    G.add_edge(id1, id2)

    clusters = list(nx.connected_components(G))
    for cluster in clusters:
        record_ids = [n for n in cluster if not str(n).startswith(("PHONE_", "PAN_", "GSTIN_"))]
        if not record_ids: continue
            
        # --- NEW: Extract PAN and GSTIN to serve as central anchors ---
        cluster_pan = ""
        cluster_gstin = ""
        for rid in record_ids:
            node_data = G.nodes.get(rid, {})
            if not cluster_pan and node_data.get('pan'):
                cluster_pan = clean_val(node_data.get('pan'))
            if not cluster_gstin and node_data.get('gstin'):
                cluster_gstin = clean_val(node_data.get('gstin'))
        # --------------------------------------------------------------

        if len(record_ids) == 1:
            auto_merged_registry.append({
                'ubid': f"KAR-{ubid_counter}",
                'master_name': G.nodes[record_ids[0]].get('business_name', ''),
                'pan_anchor': cluster_pan,       # <--- ADDED
                'gstin_anchor': cluster_gstin,   # <--- ADDED
                'linked_records': json.dumps(record_ids),
                'confidence_score': 100.0,
                'status': 'Auto-Approved (Isolated)'
            })
            ubid_counter += 1
            continue
            
        scores = [calculate_hybrid_score(str(G.nodes[id1].get('business_name', '')).lower(), 
                                         str(G.nodes[id2].get('business_name', '')).lower(), model) 
                  for id1, id2 in itertools.combinations(record_ids, 2)]
        
        avg_score = sum(scores) / len(scores) if scores else 0
        master_name = max([str(G.nodes[r].get('business_name', '')) for r in record_ids], key=len)
        
        if avg_score >= 85:
            auto_merged_registry.append({
                'ubid': f"KAR-{ubid_counter}", 
                'master_name': master_name,
                'pan_anchor': cluster_pan,       # <--- ADDED
                'gstin_anchor': cluster_gstin,   # <--- ADDED
                'linked_records': json.dumps(record_ids), 
                'confidence_score': round(avg_score, 2), 
                'status': 'Auto-Merged'
            })
        else:
            human_review_queue.append({
                'proposed_ubid': f"PENDING-KAR-{ubid_counter}", 
                'master_name': master_name,
                'pan_anchor': cluster_pan,       # <--- ADDED
                'gstin_anchor': cluster_gstin,   # <--- ADDED
                'linked_records': json.dumps(record_ids), 
                'confidence_score': round(avg_score, 2), 
                'status': 'Needs Human Review'
            })
        ubid_counter += 1
        continue
            
        scores = [calculate_hybrid_score(str(G.nodes[id1].get('business_name', '')).lower(), 
                                         str(G.nodes[id2].get('business_name', '')).lower(), model) 
                  for id1, id2 in itertools.combinations(record_ids, 2)]
        
        avg_score = sum(scores) / len(scores) if scores else 0
        master_name = max([str(G.nodes[r].get('business_name', '')) for r in record_ids], key=len)
        
        if avg_score >= 85:
            auto_merged_registry.append({
                'ubid': f"KAR-{ubid_counter}", 'master_name': master_name,
                'linked_records': json.dumps(record_ids), 'confidence_score': round(avg_score, 2), 'status': 'Auto-Merged'
            })
        else:
            human_review_queue.append({
                'proposed_ubid': f"PENDING-KAR-{ubid_counter}", 'master_name': master_name,
                'linked_records': json.dumps(record_ids), 'confidence_score': round(avg_score, 2), 'status': 'Needs Human Review'
            })
        ubid_counter += 1

print("\n[System] Exporting Output Databases...")
pd.DataFrame(auto_merged_registry).to_csv("db_auto_approved_ubids.csv", index=False)
pd.DataFrame(human_review_queue).to_csv("db_human_review_queue.csv", index=False)
print("-> Part A Complete! (Generated: db_auto_approved_ubids.csv)")