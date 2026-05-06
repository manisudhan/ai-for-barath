import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

print("=========================================================")
print(" DATABASE SEEDER: UPLOADING CSV DATA TO NEONDB           ")
print("=========================================================")

# 1. Load environment variables
load_dotenv()
DATABASE_URI = os.getenv("DATABASE_URL")

if not DATABASE_URI:
    print("[Error] DATABASE_URL not found in .env file.")
    exit()

print("[System] Connecting to NeonDB...")
try:
    engine = create_engine(DATABASE_URI)
    # Test connection
    with engine.connect() as conn:
        print("  -> Connection successful!")
except Exception as e:
    print(f"[Error] Failed to connect to database: {e}")
    exit()

# 2. Map your local CSV files to their new Postgres Table names
# Format: {'filename.csv': 'postgres_table_name'}
files_to_upload = {
    'shop_establishment.csv': 'shop_establishment',
    'bbmp_trade_license.csv': 'bbmp_trade_license',
    'kspcb_pollution.csv': 'kspcb_pollution',
    'bescom_electricity.csv': 'bescom_electricity',
    'activity_events.csv': 'activity_events'
}

print("\n[System] Beginning data ingestion...")

for csv_file, table_name in files_to_upload.items():
    if os.path.exists(csv_file):
        print(f"  -> Uploading '{csv_file}' to table '{table_name}'...")
        try:
            # For the master DBs, force Pandas to read only the first 7 columns to avoid the trailing comma error
            if csv_file != 'activity_events.csv':
                df = pd.read_csv(csv_file, usecols=range(7), on_bad_lines='skip', engine='python').fillna("")
            else:
                # For activity events, read normally
                df = pd.read_csv(csv_file).fillna("")
                
            # Push to NeonDB
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"     ✅ Successfully uploaded {len(df)} rows.")
            
        except Exception as e:
            print(f"     ❌ Error uploading {csv_file}: {e}")
    else:
        print(f"  [Warning] File '{csv_file}' not found in current directory. Skipping.")

print("\n=========================================================")
print(" SEED COMPLETE! Your NeonDB is now populated.            ")
print(" You can now run part_a.py and part_b.py.                ")
print("=========================================================")