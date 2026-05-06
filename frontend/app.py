import os
import re
import pandas as pd
import numpy as np
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) # Explicitly allow all origins

# Silence warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ==========================================
# SECURE DATABASE CONFIGURATION (NeonDB)
# ==========================================
load_dotenv()
DATABASE_URI = os.getenv("DATABASE_URL")

if not DATABASE_URI:
    print("[Error] DATABASE_URL not found in .env file.")
    exit()

engine = create_engine(DATABASE_URI)

# ==========================================
# SEMANTIC SEARCH AI INITIALIZATION
# ==========================================
print("[System] Loading Semantic Search AI (all-MiniLM-L6-v2)...")
nlp_model = SentenceTransformer('all-MiniLM-L6-v2')

MASTER_DF = None
DB_EMBEDDINGS = None

def refresh_search_index():
    global MASTER_DF, DB_EMBEDDINGS
    try:
        print("[System] Refreshing Master Registry from NeonDB...")
        df = pd.read_sql_table('db_final_master_registry', engine).fillna("")
        df['pincode'] = df['pincode'].astype(str).str.replace(".0", "", regex=False)
        
        df['search_context'] = (
            "UBID is " + df['ubid'] + ". " +
            "Business Name is " + df['business_name'] + ". " +
            "Address is " + df['address'] + " in pincode " + df['pincode'] + ". " +
            "GSTIN or PAN is " + df['gstin_or_pan'] + ". " +
            "Status is " + df['activity_status'] + " because " + df['status_reason']
        )
        
        print("[System] Vectorizing database (creating embeddings)...")
        DB_EMBEDDINGS = nlp_model.encode(df['search_context'].tolist())
        MASTER_DF = df
        print("[System] Search Index Ready!")
        
    except ValueError:
        print("[Error] Table 'db_final_master_registry' not found in NeonDB. Cannot build search index.")

refresh_search_index()

# ==========================================
# API ENDPOINTS
# ==========================================

@app.route('/upload_to_postgres', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']
        dataset_name = request.form.get('dataset_name', 'unknown_dataset')
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        df = pd.read_csv(file)
        df['source_upload_name'] = dataset_name
        table_name = 'activity_events' if 'events' in dataset_name.lower() else 'raw_department_uploads'
        df.to_sql(table_name, engine, if_exists='append', index=False)
        return jsonify({'message': 'File successfully saved.', 'rows_inserted': len(df)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['GET'])
def search_database():
    user_query = request.args.get('q', '').strip()
    if MASTER_DF is None or DB_EMBEDDINGS is None:
        return jsonify({'error': 'Search index not initialized.'}), 500
    if not user_query:
        return jsonify(MASTER_DF.head(50).to_dict('records')), 200

    try:
        query_lower = user_query.lower()
        scores = np.zeros(len(MASTER_DF))
        
        extracted_ubids = re.findall(r'kar-\d+', query_lower)
        extracted_pincodes = re.findall(r'\b\d{6}\b', query_lower)
        
        for idx, row in MASTER_DF.iterrows():
            if any(u in row['ubid'].lower() for u in extracted_ubids): scores[idx] += 2.0 
            if any(p in str(row['pincode']) for p in extracted_pincodes): scores[idx] += 1.0
            if "active" in query_lower and row['activity_status'].lower() == "active": scores[idx] += 0.5
            elif "closed" in query_lower and row['activity_status'].lower() == "closed": scores[idx] += 0.5
            elif "dormant" in query_lower and row['activity_status'].lower() == "dormant": scores[idx] += 0.5

        query_embedding = nlp_model.encode([user_query])
        semantic_scores = cosine_similarity(query_embedding, DB_EMBEDDINGS)[0]
        final_scores = scores + semantic_scores
        
        df_results = MASTER_DF.copy()
        df_results['search_score'] = final_scores
        best_matches = df_results[df_results['search_score'] > 0.3].sort_values(by='search_score', ascending=False)
        return jsonify(best_matches.head(50).to_dict('records')), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---> UPDATED ROUTES <---
@app.route('/review_queue', methods=['GET'])
def get_review_queue():
    try:
        # FIX 1: Limit the payload to 100 records at a time.
        # This stops the frontend from trying to render thousands of rows and lagging out.
        query = "SELECT * FROM db_human_review_queue LIMIT 100"
        df = pd.read_sql(query, engine).fillna("")
        return jsonify(df.to_dict('records')), 200
    except ValueError:
        return jsonify([]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_review', methods=['POST'])
def process_review():
    data = request.json
    target_ubid = data.get('ubid')
    action = data.get('action') 

    if not target_ubid or action not in ['accept', 'reject']:
        return jsonify({'error': 'Invalid request parameters'}), 400

    try:
        with engine.begin() as conn:
            if action == 'accept':
                query = text("SELECT * FROM db_human_review_queue WHERE ubid = :ubid")
                result = conn.execute(query, {"ubid": target_ubid}).fetchone()
                
                if not result:
                    return jsonify({'error': 'Record not found in queue'}), 404

                row_dict = dict(result._mapping)
                new_ubid = row_dict['ubid'].replace("PENDING-", "")
                row_dict['ubid'] = new_ubid
                df = pd.DataFrame([row_dict])
                
                # FIX 2a: Write to the staging table (original logic)
                df.to_sql('db_auto_complete_ubids', conn, if_exists='append', index=False)
                
                # FIX 2b: Immediately write to the final master registry so it exists globally
                df.to_sql('db_final_master_registry', conn, if_exists='append', index=False)

            delete_query = text("DELETE FROM db_human_review_queue WHERE ubid = :ubid")
            conn.execute(delete_query, {"ubid": target_ubid})

        # FIX 3: Automatically refresh the in-memory Pandas search index 
        # so the Home Page updates with the new data instantly.
        if action == 'accept':
            refresh_search_index()

        return jsonify({'message': f'Record successfully {action}ed.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/refresh_index', methods=['POST'])
def trigger_refresh():
    refresh_search_index()
    return jsonify({"message": "Search Index Refreshed Successfully."}), 200

if __name__ == '__main__':
    print("🚀 Starting Backend Server on http://localhost:5000")
    app.run(port=5000, debug=True)