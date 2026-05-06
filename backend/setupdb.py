import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

print("=========================================")
print(" INJECTING ACTIVITY EVENTS DATA          ")
print("=========================================")

load_dotenv()
DATABASE_URI = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URI)

# The perfectly mapped transactional events dataset
events_data = [
    # 1. Matches "Global Tech India"
    # Event date is 2026-04-15. This is < 180 days from your code's current date of 2026-05-02.
    # Expected Result: ACTIVE
    {"record_id": "SE-01", "event_date": "2026-04-15", "event_type": "renewal", "source_db": "Shop_Establishment"},
    {"record_id": "TL-01", "event_date": "2025-05-01", "event_type": "tax_payment", "source_db": "BBMP_Trade"},

    # 2. Matches "Apex Manufacturing" 
    # Contains the keyword 'closure'. Your state machine will detect this and override the date.
    # Expected Result: CLOSED
    {"record_id": "SE-03", "event_date": "2025-11-20", "event_type": "inspection", "source_db": "Shop_Establishment"},
    {"record_id": "KP-01", "event_date": "2026-02-10", "event_type": "closure", "source_db": "KSPCB_Pollution"},

    # 3. Matches NOTHING (The Failsafe Test)
    # This record ID does not exist in your Part A auto-approved table.
    # Expected Result: ROUTED TO ORPHAN QUEUE
    {"record_id": "FAKE-ID-99", "event_date": "2026-05-01", "event_type": "bill_payment", "source_db": "BESCOM_Power"}
]

df = pd.DataFrame(events_data)
df.to_sql('activity_events', engine, if_exists='replace', index=False)

print("✅ Successfully created 'activity_events' table!")
print(f" -> Inserted {len(df)} transactional events.")
print(" -> You can now run python part_b.py!")