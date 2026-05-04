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

# Silence tokenization warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.filterwarnings("ignore", category=FutureWarning)

print("=========================================================")
print(" PART A: IDENTITY RESOLUTION & SCHEMA MAPPING ENGINE     ")
print("=========================================================")

print("\n[System] Loading NLP Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Enhanced definitions to replace the need for Regex fallbacks
master_definitions = {
    "record_id": "unique record database ID application number rr number license identifier account no consent no se number",
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
    """Maps arbitrary dataframes to the master schema using purely NLP."""
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
        if score > 0.40 and master_col not in assigned_master_cols:
            rename_mapping[incoming_col] = master_col
            assigned_master_cols.add(master_col)

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
    'KSPCB_Pollution': 'kspcb_pollution.csv',
    'BESCOM_Power': 'bescom_electricity.csv'
}

standardized_dfs = []
for db_name, filename in files_to_load.items():
    if os.path.exists(filename):
        try:
            # FIX FOR PARSER ERROR: usecols=range(7) forces Pandas to grab exactly 7 columns
            # and ignore any accidental trailing commas on empty fields.
            raw_df = pd.read_csv(filename, usecols=range(7), on_bad_lines='skip', engine='python').fillna("")
            standardized_dfs.append(map_schema_intelligently(raw_df, db_name))
        except Exception as e:
            print(f"  [Error] reading {filename}: {e}")
    else:
        print(f"  [Warning] {filename} not found in directory. Skipping.")

if not standardized_dfs:
    print("[Error] No master CSV files found. Exiting.")
    exit()

master_df = pd.concat(standardized_dfs, ignore_index=True).fillna("")

print("\n[System] Building Graphs and Resolving Identities...")
auto_complete_registry = []
human_review_queue = []
ubid_counter = 1000

grouped_by_pincode = master_df.groupby('pincode')

for pincode, group in grouped_by_pincode:
    if not pincode: continue 
    
    G = nx.Graph()
    # 1. Add nodes and explicit anchor edges
    for _, row in group.iterrows():
        rec_id = row['record_id']
        if not rec_id: continue
        G.add_node(rec_id, **row.to_dict())
        
        if clean_val(row['phone']): G.add_edge(rec_id, f"PHONE_{clean_val(row['phone'])}")
        if clean_val(row['pan']):   G.add_edge(rec_id, f"PAN_{clean_val(row['pan'])}")
        if clean_val(row['gstin']): G.add_edge(rec_id, f"GSTIN_{clean_val(row['gstin'])}")

    # 2. Add fuzzy edges for Business Name and Address
    records_list = group.to_dict('records')
    for i, row1 in enumerate(records_list):
        id1 = row1['record_id']
        name1 = clean_val(row1.get('business_name', '')).lower()
        add1 = clean_val(row1.get('address', '')).lower()
        
        for j in range(i + 1, len(records_list)):
            id2 = records_list[j]['record_id']
            name2 = clean_val(records_list[j].get('business_name', '')).lower()
            add2 = clean_val(records_list[j].get('address', '')).lower()
            
            # Fuzzy match on business name and address if anchors are missing
            if name1 and name2:
                if fuzz.token_set_ratio(name1, name2) >= 85:
                    if add1 and add2 and fuzz.token_set_ratio(add1, add2) >= 70:
                        G.add_edge(id1, id2)

    # 3. Process the connected clusters
    clusters = list(nx.connected_components(G))
    for cluster in clusters:
        record_ids = [n for n in cluster if not str(n).startswith(("PHONE_", "PAN_", "GSTIN_"))]
        if not record_ids: continue
            
        c_pan, c_gstin, c_phone, c_owner = "", "", "", ""
        
        # Aggregate data from the nodes in this cluster
        for rid in record_ids:
            node_data = G.nodes.get(rid, {})
            if not c_pan and node_data.get('pan'): c_pan = clean_val(node_data.get('pan'))
            if not c_gstin and node_data.get('gstin'): c_gstin = clean_val(node_data.get('gstin'))
            if not c_phone and node_data.get('phone'): c_phone = clean_val(node_data.get('phone'))
            if not c_owner and node_data.get('owner_name'): c_owner = clean_val(node_data.get('owner_name'))
        
        master_name = max([str(G.nodes[r].get('business_name', '')) for r in record_ids], key=len, default="")
        master_address = max([str(G.nodes[r].get('address', '')) for r in record_ids], key=len, default="")

        # Establish combined PAN/GSTIN column
        gstin_or_pan = c_pan if c_pan else c_gstin

        # Calculate Confidence Score based on business name similarity within the cluster
        avg_score = 100.0
        if len(record_ids) > 1:
            scores = [calculate_hybrid_score(str(G.nodes[i1].get('business_name', '')).lower(), 
                                             str(G.nodes[i2].get('business_name', '')).lower(), model) 
                      for i1, i2 in itertools.combinations(record_ids, 2)]
            avg_score = sum(scores) / len(scores) if scores else 0

        # Generate Explainable Linkage Signal
        if len(record_ids) == 1:
            evidence = "Isolated Record. No anchors or fuzzy matches found in other databases."
        elif c_pan or c_gstin:
            evidence = f"Anchored via Central ID ({gstin_or_pan}) across {len(record_ids)} systems."
        elif c_phone:
            evidence = f"Matched via Shared Phone ({c_phone}) and Name Similarity ({round(avg_score, 1)}%)."
        else:
            evidence = f"Matched via Fuzzy NLP Name/Address match. Score: {round(avg_score, 1)}%."

        # Format exact requested payload with requested columns
        payload = {
            'ubid': f"KAR-{ubid_counter}",
            'gstin_or_pan': gstin_or_pan,
            'business_name': master_name,
            'owner_name': c_owner,
            'phone': c_phone,
            'pincode': pincode,
            'address': master_address,
            'merged_records': " | ".join(record_ids),
            'confidence_score': round(avg_score, 2),
            'linkage_evidence': evidence
        }

        # Routing Logic based on Problem Statement
        if avg_score >= 85 or c_pan or c_gstin:
            auto_complete_registry.append(payload)
        else:
            # Low confidence / Ambiguous -> Route to Human Review
            payload['ubid'] = f"PENDING-KAR-{ubid_counter}"
            human_review_queue.append(payload)
            
        ubid_counter += 1

print("\n[System] Exporting Output Databases...")
pd.DataFrame(auto_complete_registry).to_csv("db_auto_complete_ubids.csv", index=False)
pd.DataFrame(human_review_queue).to_csv("db_human_review_queue.csv", index=False)
print("-> Part A Complete!")
print(f"   - Auto Complete Records: {len(auto_complete_registry)} (Saved to db_auto_complete_ubids.csv)")
print(f"   - Human Review Records: {len(human_review_queue)} (Saved to db_human_review_queue.csv)")