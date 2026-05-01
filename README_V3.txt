CastIQ Pro Durban Boat Intelligence V3

Folder:
C:\Users\Admin\Desktop\Durban Fishing

Run:
cd "C:\Users\Admin\Desktop\Durban Fishing"
pip install -r requirements.txt
streamlit run app.py

Optional secrets:
Create .streamlit\secrets.toml
AISHUB_USERNAME = "your_aishub_username"
MARINETRAFFIC_API_KEY = "your_marinetraffic_key"

Manual/backup AIS file:
data\vessel_tracks.csv

Columns:
vessel_name,mmsi,timestamp,lat,lon,speed_knots,course_deg,source
