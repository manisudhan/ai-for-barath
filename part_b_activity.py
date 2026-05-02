import os
import json
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

print("=========================================================")
print(" PART B: ACTIVITY INTELLIGENCE & STATE MACHINE           ")
print("=========================================================")

# 1. Load the UBID Linkage Map from Part A
try:
    ubid_df = pd.read_csv("db_auto_approved_ubids.csv")
except FileNotFoundError:
    print("[Error] db_auto_approved_ubids.csv not found. You must run part_a_ubid.py first!")
    exit()

print("\n[System] Loading Event Bridge...")
record_to_ubid = {}
for _, row in ubid_df.iterrows():
    try:
        linked_ids = json.loads(row['linked_records'])
        for rid in linked_ids:
            record_to_ubid[rid] = row['ubid']
    except Exception as e:
        pass

print(f"  -> Ready: Tracking {len(record_to_ubid)} historical departmental IDs.")

# 2. Load the Event Files
event_files = {
    'Shop_Establishment': 'events_shop_establishment.csv',
    'BBMP_Trade': 'events_bbmp.csv',
    'BESCOM_Power': 'events_bescom.csv',
    'BWSSB_Water': 'events_bwssb.csv',
    'FSSAI_Food': 'events_fssai.csv',
    'KSPCB_Pollution': 'events_kspcb.csv'
}

all_events = []
for source, filename in event_files.items():
    try:
        ev_df = pd.read_csv(filename)
        all_events.append(ev_df)
    except FileNotFoundError:
        pass

if not all_events:
    print("[Error] No event CSV files found. Exiting.")
    exit()

events_df = pd.concat(all_events, ignore_index=True)

# 3. Bridge Events to UBIDs
print("\n[System] Bridging transactional events to UBID Registry...")
matched_events = []
unmatched_events = []

# Map the source database to its specific ID column name (Fixing the Bug)
id_column_map = {
    'Shop_Establishment': 'SE_Number',
    'BBMP_Trade': 'License_No',
    'BESCOM_Power': 'RR_Number',
    'BWSSB_Water': 'Account_No',
    'FSSAI_Food': 'FSSAI_License',
    'KSPCB_Pollution': 'Consent_No'
}

for _, row in events_df.iterrows():
    source_db = row.get('source_db', '')
    id_col_name = id_column_map.get(source_db, 'record_id') 
    rec_id = str(row.get(id_col_name, '')).strip()
    
    target_ubid = record_to_ubid.get(rec_id)
    event_data = row.to_dict()
    
    if target_ubid:
        event_data['ubid'] = target_ubid
        matched_events.append(event_data)
    else:
        event_data['reason'] = f"Record ID '{rec_id}' not found in UBID Registry"
        unmatched_events.append(event_data)
        
print(f"  -> Successfully matched {len(matched_events)} events.")
print(f"  -> Routed {len(unmatched_events)} orphan events to review queue.")

# 4. State Machine (Active/Dormant/Closed)
print("\n[System] Running Activity Classification State Machine...")

matched_events_df = pd.DataFrame(matched_events)
matched_events_df['event_date'] = pd.to_datetime(matched_events_df['event_date'])
matched_events_df = matched_events_df.sort_values(by='event_date', ascending=False)

# Simulated Hackathon Date
CURRENT_DATE = pd.to_datetime("2026-05-02")
final_status_registry = []

grouped_events = matched_events_df.groupby('ubid')

for ubid, group in grouped_events:
    latest_event = group.iloc[0]
    months_inactive = relativedelta(CURRENT_DATE, latest_event['event_date']).months + \
                      (relativedelta(CURRENT_DATE, latest_event['event_date']).years * 12)
                      
    closure_signals = ['closure', 'disconnection', 'cancellation', 'cancelled']
    has_closure = group['event_type'].str.lower().isin(closure_signals).any()
    
    if has_closure:
        status, reason = "Closed", "Explicit closure/disconnection event detected."
    elif months_inactive <= 6:
        status, reason = "Active", f"Activity logged {months_inactive} months ago ({latest_event['source_db']})."
    elif 6 < months_inactive <= 24:
        status, reason = "Dormant", f"No activity for {months_inactive} months."
    else:
        status, reason = "Closed", f"Inactive for >24 months ({months_inactive} months)."
        
    final_status_registry.append({
        'ubid': ubid,
        'status': status,
        'months_inactive': months_inactive,
        'reason': reason,
        'last_event_date': latest_event['event_date'].strftime('%Y-%m-%d'),
        'total_events_logged': len(group)
    })
    
status_df = pd.DataFrame(final_status_registry)

# Merge names for a clean report
names_df = ubid_df[['ubid', 'master_name']]
final_report = pd.merge(status_df, names_df, on='ubid', how='left')

print("\n--- FINAL ACTIVITY CLASSIFICATIONS (Preview) ---")
print(final_report[['ubid', 'master_name', 'status', 'reason']].head(10).to_string(index=False))

# Export Data
final_report.to_csv("db_final_activity_status.csv", index=False)
pd.DataFrame(unmatched_events).to_csv("db_orphan_events_queue.csv", index=False)

print("\n-> Part B Complete! (Generated: db_final_activity_status.csv)")