import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv

print("=========================================================")
print(" PART B: POSTGRESQL ACTIVITY & STATE MACHINE             ")
print("=========================================================")

# ==========================================
# SECURE DATABASE CONFIGURATION (NeonDB)
# ==========================================
# Load environment variables from .env file
load_dotenv()

DATABASE_URI = os.getenv("DATABASE_URL")

if not DATABASE_URI:
    print("[Error] DATABASE_URL not found in .env file. Please check your configuration.")
    exit()

print("[System] Connecting to NeonDB...")
engine = create_engine(DATABASE_URI)

# 1. Load the UBID Linkage Map from PostgreSQL (Output of Part A)
try:
    print("[System] Loading UBID Registry from NeonDB...")
    ubid_df = pd.read_sql_table('db_auto_complete_ubids', engine)
except ValueError:
    print("[Error] Table 'db_auto_complete_ubids' not found. You must run Part A first!")
    exit()

# 2. Load the Central Event Stream from PostgreSQL
try:
    print("[System] Loading Activity Events from NeonDB...")
    # NOTE: If your app.py saved this as 'raw_department_uploads', change the table name here to match.
    events_df = pd.read_sql_table('activity_events', engine) 
except ValueError:
    print("[Error] Table 'activity_events' not found in database!")
    exit()

print("\n[System] Loading Event Bridge...")
record_to_ubid_info = {}

# Parse the "merged_records" column which contains IDs separated by " | "
for _, row in ubid_df.iterrows():
    ubid = row['ubid']
    pincode = row['pincode']
    
    # Protect against empty rows
    if pd.notna(row['merged_records']):
        linked_ids = str(row['merged_records']).split(" | ")
        for rid in linked_ids:
            rid = rid.strip()
            if rid:
                # Store the mapping so we know which UBID and Pincode this Record ID belongs to
                record_to_ubid_info[rid] = {'ubid': ubid, 'pincode': pincode}

print(f"  -> Ready: Tracking {len(record_to_ubid_info)} historical departmental IDs.")

# 3. Bridge Events to UBIDs
print("\n[System] Bridging transactional events to UBID Registry...")
matched_events = []
unmatched_events = []

for _, row in events_df.iterrows():
    rec_id = str(row.get('record_id', '')).strip()
    
    target_info = record_to_ubid_info.get(rec_id)
    event_data = row.to_dict()
    
    if target_info:
        # Event successfully joined to an existing UBID
        event_data['ubid'] = target_info['ubid']
        event_data['pincode'] = target_info['pincode']
        matched_events.append(event_data)
    else:
        # Orphan Event: Cannot be confidently joined to a UBID, routed to review
        event_data['reason'] = f"Record ID '{rec_id}' not found in Part A UBID Registry"
        unmatched_events.append(event_data)

print(f"  -> Successfully matched {len(matched_events)} events.")
print(f"  -> Routed {len(unmatched_events)} orphan events to review queue.")

# 4. State Machine (Active/Dormant/Closed) - OPTIMIZED BY PINCODE
print("\n[System] Running Activity Classification State Machine by Pincode...")

matched_events_df = pd.DataFrame(matched_events)

# Simulated Date for testing
CURRENT_DATE = pd.to_datetime("2026-05-02")
final_status_registry = []

if not matched_events_df.empty:
    matched_events_df['event_date'] = pd.to_datetime(matched_events_df['event_date'])
    matched_events_df = matched_events_df.sort_values(by='event_date', ascending=False)
    
    # PERFORMANCE OPTIMIZATION: Process events grouped by pincode to reduce memory load
    grouped_by_pincode = matched_events_df.groupby('pincode')
    
    for pincode, pincode_events in grouped_by_pincode:
        
        # Within this specific pincode, group by the business UBID
        grouped_events = pincode_events.groupby('ubid')
        
        for ubid, group in grouped_events:
            latest_event = group.iloc[0]
            days_inactive = (CURRENT_DATE - latest_event['event_date']).days
            
            # Check for explicitly closed signals across the timeline
            closure_signals = ['closure', 'disconnection', 'cancellation', 'cancelled']
            has_closure = group['event_type'].str.lower().isin(closure_signals).any()
            
            # Explainable Classification Logic
            if has_closure:
                status, reason = "Closed", "Explicit closure/disconnection event detected."
            elif days_inactive <= 180:
                status, reason = "Active", f"Activity logged {days_inactive} days ago ({latest_event['source_db']})."
            elif 180 < days_inactive <= 730:
                status, reason = "Dormant", f"No activity for {days_inactive} days."
            else:
                status, reason = "Closed", f"Inactive for >2 years ({days_inactive} days)."
                
            final_status_registry.append({
                'ubid': ubid,
                'activity_status': status,
                'status_reason': reason,
                'last_event_date': latest_event['event_date'].strftime('%Y-%m-%d'),
                'total_events_logged': len(group)
            })

status_df = pd.DataFrame(final_status_registry)

# 5. Merge the Status back onto the original Part A Output
if not status_df.empty:
    final_report = pd.merge(ubid_df, status_df, on='ubid', how='left')
else:
    final_report = ubid_df.copy()

# Fill NaNs for valid businesses that just happened to have NO activity events
final_report['activity_status'] = final_report['activity_status'].fillna("Unknown")
final_report['status_reason'] = final_report['status_reason'].fillna("No transactional events found across any department.")
final_report['last_event_date'] = final_report['last_event_date'].fillna("N/A")
final_report['total_events_logged'] = final_report['total_events_logged'].fillna(0).astype(int)

print("\n--- FINAL MASTER REGISTRY PREVIEW ---")
print(final_report[['ubid', 'business_name', 'activity_status', 'status_reason']].head(10).to_string(index=False))

# 6. Export Final Data to PostgreSQL instead of CSV
print("\n[System] Saving Results directly to NeonDB...")

final_report.to_sql('db_final_master_registry', engine, if_exists='replace', index=False)

if unmatched_events:
    pd.DataFrame(unmatched_events).to_sql('db_orphan_events_queue', engine, if_exists='replace', index=False)

print("\n-> Part B Complete!")
print(f"   - Saved Main Registry to NeonDB table: db_final_master_registry")
if unmatched_events:
    print(f"   - Saved Human Review Queue (Orphans) to NeonDB table: db_orphan_events_queue")