import math
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="CastIQ Pro – Durban Boat Intelligence V5", page_icon="🎣", layout="wide")

ROOT = Path.cwd()
SPOTS_CSV = ROOT / "durban_boat_fishing_spots.csv"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
TRACKS_CSV = DATA_DIR / "vessel_tracks.csv"
CACHE_CSV = DATA_DIR / "live_ais_cache.csv"
DURBAN_LAUNCH = (-29.8689, 31.0617)
DURBAN_BBOX = {"min_lat": -29.98, "max_lat": -29.55, "min_lon": 30.95, "max_lon": 31.28}

# ---------------------------
# Helpers
# ---------------------------
def get_secret(name, default=""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def destination_point(lat, lon, bearing_deg, distance_km):
    R = 6371.0
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    d = distance_km / R
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d) * math.cos(lat1), math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def google_maps_url(lat, lon):
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"

def navionics_url(lat, lon):
    return f"https://webapp.navionics.com/?lat={lat}&lng={lon}&zoom=13"

def normalise_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {
        "latitude": "lat", "lat_dd": "lat", "y": "lat",
        "longitude": "lon", "lng": "lon", "lon_dd": "lon", "x": "lon",
        "speed": "speed_knots", "sog": "speed_knots", "speed_over_ground": "speed_knots",
        "course": "course_deg", "cog": "course_deg", "heading": "course_deg",
        "name": "vessel_name", "ship_name": "vessel_name", "boat_name": "vessel_name", "vessel": "vessel_name",
        "time": "timestamp", "datetime": "timestamp", "date_time": "timestamp", "last_position": "timestamp",
        "mmsi_number": "mmsi", "ship_id": "shipid",
    }
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})
    for c in ["vessel_name", "mmsi", "timestamp", "lat", "lon", "speed_knots", "course_deg", "source"]:
        if c not in df.columns:
            df[c] = None
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["speed_knots"] = pd.to_numeric(df["speed_knots"], errors="coerce")
    df["course_deg"] = pd.to_numeric(df["course_deg"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["vessel_name"] = df["vessel_name"].fillna("Unknown vessel").astype(str)
    df["source"] = df["source"].fillna("manual/upload").astype(str)
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"].between(-31, -28)) & (df["lon"].between(29, 33))]
    return df

# ---------------------------
# Data loading
# ---------------------------
@st.cache_data(ttl=1800)
def load_spots():
    if not SPOTS_CSV.exists():
        st.error("Missing durban_boat_fishing_spots.csv. Put it in C:\\Users\\Admin\\Desktop\\Durban Fishing next to app.py")
        st.stop()
    df = pd.read_csv(SPOTS_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

@st.cache_data(ttl=900)
def get_open_meteo(lat=-29.86, lon=31.06):
    weather_url = "https://api.open-meteo.com/v1/forecast"
    marine_url = "https://marine-api.open-meteo.com/v1/marine"
    weather_params = {
        "latitude": lat, "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation",
        "current": "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "timezone": "Africa/Johannesburg", "forecast_days": 1,
    }
    marine_params = {
        "latitude": lat, "longitude": lon,
        "hourly": "wave_height,wave_direction,wave_period",
        "current": "wave_height,wave_direction,wave_period",
        "timezone": "Africa/Johannesburg", "forecast_days": 1,
    }
    try:
        w = requests.get(weather_url, params=weather_params, timeout=12).json()
        m = requests.get(marine_url, params=marine_params, timeout=12).json()
        return w, m
    except Exception:
        return {}, {}

# ---------------------------
# Live AIS connectors
# ---------------------------
def fetch_aishub_live(username, bbox):
    """AISHub webservice. Requires an AISHub username/API access. Limited to once per minute by provider."""
    if not username:
        return pd.DataFrame()
    url = "https://data.aishub.net/ws.php"
    params = {
        "username": username,
        "format": 1,
        "output": "json",
        "compress": 0,
        "latmin": bbox["min_lat"], "latmax": bbox["max_lat"],
        "lonmin": bbox["min_lon"], "lonmax": bbox["max_lon"],
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = data.get("response", data if isinstance(data, list) else [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    rename = {"LATITUDE": "lat", "LONGITUDE": "lon", "SOG": "speed_knots", "COG": "course_deg", "NAME": "vessel_name", "MMSI": "mmsi", "TIME": "timestamp"}
    df = df.rename(columns=rename)
    df["source"] = "AISHub live"
    return normalise_columns(df)

def fetch_marinetraffic_placeholder(api_key, bbox):
    """Placeholder: MarineTraffic endpoint shape depends on subscribed AIS service/package.
    Use the app upload/manual mode until you have the exact service URL from your MT account.
    """
    return pd.DataFrame()

def load_cached_and_manual_tracks(uploaded):
    frames = []
    if TRACKS_CSV.exists():
        try:
            frames.append(normalise_columns(pd.read_csv(TRACKS_CSV)))
        except Exception:
            pass
    if CACHE_CSV.exists():
        try:
            frames.append(normalise_columns(pd.read_csv(CACHE_CSV)))
        except Exception:
            pass
    if uploaded is not None:
        try:
            frames.append(normalise_columns(pd.read_csv(uploaded)))
        except Exception as e:
            st.warning(f"Could not read uploaded file: {e}")
    if not frames:
        return pd.DataFrame(columns=["vessel_name","mmsi","timestamp","lat","lon","speed_knots","course_deg","source"])
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["mmsi", "vessel_name", "timestamp", "lat", "lon"], keep="last")
    return out

def save_live_cache(df):
    if df.empty:
        return
    existing = pd.read_csv(CACHE_CSV) if CACHE_CSV.exists() else pd.DataFrame()
    combined = pd.concat([existing, df], ignore_index=True)
    combined = normalise_columns(combined).drop_duplicates(subset=["mmsi", "vessel_name", "timestamp", "lat", "lon"], keep="last")
    combined.to_csv(CACHE_CSV, index=False)

# ---------------------------
# Fishing behaviour + drift
# ---------------------------
def detect_fishing_stops(ais_df, max_speed=3.0, min_points=2, radius_m=550):
    if ais_df.empty:
        return pd.DataFrame()
    df = ais_df.copy()
    df["is_slow"] = df["speed_knots"].fillna(99) <= max_speed
    slow = df[df["is_slow"]].copy()
    if slow.empty:
        return pd.DataFrame()
    cell = radius_m / 111000
    slow["lat_cell"] = (slow["lat"] / cell).round() * cell
    slow["lon_cell"] = (slow["lon"] / cell).round() * cell
    grouped = slow.groupby(["lat_cell", "lon_cell"], as_index=False).agg(
        stop_points=("lat", "count"),
        vessels=("vessel_name", lambda x: ", ".join(sorted(set(map(str, x)))[:8])),
        avg_speed_knots=("speed_knots", "mean"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        sources=("source", lambda x: ", ".join(sorted(set(map(str, x)))[:5])),
    )
    grouped = grouped[grouped["stop_points"] >= min_points].rename(columns={"lat_cell": "lat", "lon_cell": "lon"})
    if grouped.empty:
        return grouped
    grouped["heat_score"] = (grouped["stop_points"] * 12).clip(upper=100).astype(int)
    return grouped.sort_values(["heat_score", "stop_points"], ascending=False)

def compute_drift_lines(points_df, wind_dir, wind_speed, wave_dir, wave_h, hours=1.0):
    if points_df.empty:
        return []
    # In Durban, nearshore current often dominates. This is a planning estimate: current direction blended with wind and swell.
    # Wind direction from API is direction wind comes FROM, so drift is roughly +180.
    wind_to = (wind_dir + 180) % 360 if wind_dir is not None else 45
    swell_to = wave_dir if wave_dir is not None else wind_to
    drift_bearing = (0.55 * swell_to + 0.45 * wind_to) % 360
    speed_kmh = max(0.4, min(3.2, 0.08 * (wind_speed or 10) + 0.55 * (wave_h or 1.0)))
    distance_km = speed_kmh * hours
    lines = []
    for _, r in points_df.head(25).iterrows():
        end_lat, end_lon = destination_point(r["lat"], r["lon"], drift_bearing, distance_km)
        lines.append({"start": (r["lat"], r["lon"]), "end": (end_lat, end_lon), "bearing": drift_bearing, "distance_km": distance_km})
    return lines

def enrich_spots(spots, stops):
    spots = spots.copy()
    if stops.empty:
        spots["boat_stop_count_nearby"] = 0
        spots["nearest_boat_stop_km"] = None
        spots["boat_intel_bonus"] = 0
        return spots
    counts, nearests = [], []
    for _, r in spots.iterrows():
        d = stops.apply(lambda s: haversine_km(r["lat"], r["lon"], s["lat"], s["lon"]), axis=1)
        near = stops[d <= 2.0]
        counts.append(int(near["stop_points"].sum()) if not near.empty else 0)
        nearests.append(round(float(d.min()), 2) if len(d) else None)
    spots["boat_stop_count_nearby"] = counts
    spots["nearest_boat_stop_km"] = nearests
    spots["boat_intel_bonus"] = spots["boat_stop_count_nearby"].apply(lambda x: min(20, int(x * 2)))
    return spots

def score_spot(row, wind_speed, gust, wave_h, wave_period, launch_lat, launch_lon, target):
    distance = haversine_km(launch_lat, launch_lon, row["lat"], row["lon"])
    safety = 100
    if wind_speed > 28: safety -= 35
    elif wind_speed > 20: safety -= 20
    elif wind_speed > 14: safety -= 10
    if gust > 35: safety -= 25
    elif gust > 28: safety -= 15
    if wave_h > 2.0: safety -= 35
    elif wave_h > 1.5: safety -= 20
    elif wave_h > 1.1: safety -= 10
    if wave_period and wave_period < 8: safety -= 10
    if distance > 28: safety -= 15
    elif distance > 15: safety -= 8
    type_txt = str(row.get("type", "")).lower()
    species_txt = (str(row.get("target_species", "")) + " " + str(row.get("strategy_note", ""))).lower()
    bait_bonus = 20 if "bait" in type_txt or "bait" in species_txt else 8
    target_bonus = 18 if target.lower() in species_txt else 7
    structure_bonus = 14 if any(x in type_txt for x in ["reef", "wreck", "drop-off", "artificial"]) else 5
    crowd_penalty = 8 if any(x in str(row.get("name", "")).lower() for x in ["fontao", "no.1", "barge"]) else 0
    boat_bonus = int(row.get("boat_intel_bonus", 0) or 0)
    total = max(0, min(100, int((safety * 0.38) + bait_bonus + target_bonus + structure_bonus + boat_bonus - crowd_penalty)))
    return total, distance


# ---------------------------
# Species bait / trace intelligence
# ---------------------------
SPECIES_GUIDE = {
    "kob": {
        "bait": "Chokka + sardine combo, mackerel fillet, live mullet if available.",
        "trace": "Sliding sinker or running trace; 5/0–8/0 circle/J-hook; 0.60–0.80 mm leader.",
        "setup": "Fish near sand/reef edges, gutters and dirty-water edges. Slow drift or anchor only if safe.",
        "bait_img": "images/bait_kob.png", "trace_img": "images/trace_kob.png", "setup_img": "images/setup_kob.png",
    },
    "couta": {
        "bait": "Live mackerel, dead mackerel, walla-walla, redeye sardine; add light wire.",
        "trace": "Couta trace with wire, treble/stinger hook and small duster/skirt if needed.",
        "setup": "Slow troll/live bait around bait reefs and wreck edges; work up-current side first.",
        "bait_img": "images/bait_couta.png", "trace_img": "images/trace_couta.png", "setup_img": "images/setup_couta.png",
    },
    "tuna": {
        "bait": "Live mackerel, sardine chum, small feathers/spoons when birds or surface activity shows.",
        "trace": "Fluorocarbon leader, strong swivel, 4/0–7/0 hook; keep hardware light and clean.",
        "setup": "Watch birds, bait balls and current colour lines; drift baits behind the boat.",
        "bait_img": "images/bait_tuna.png", "trace_img": "images/trace_tuna.png", "setup_img": "images/setup_tuna.png",
    },
    "snoek": {
        "bait": "Small spoon, fillet strip, redeye/sardine, small live bait when snoek are feeding.",
        "trace": "Light wire bite trace or heavy fluorocarbon; small sharp hooks/spoons.",
        "setup": "Work early morning current lines, backline edges and bait pockets north of Durban.",
        "bait_img": "images/bait_snoek.png", "trace_img": "images/trace_snoek.png", "setup_img": "images/setup_snoek.png",
    },
    "dorade": {
        "bait": "Small live bait, sardine strip, squid strip; bright lures around floating debris.",
        "trace": "Fluorocarbon leader, 3/0–6/0 hook; avoid heavy visible hardware.",
        "setup": "Run current lines, floating debris and warm blue-water edges; keep a pitch bait ready.",
        "bait_img": "images/bait_dorade.png", "trace_img": "images/trace_dorade.png", "setup_img": "images/setup_dorade.png",
    },
    "bait": {
        "bait": "Sabiki/jigs for mackerel, mozzies and shad; small squid strips if needed.",
        "trace": "Sabiki rig with sinker; keep spare rigs ready because tangles happen fast.",
        "setup": "Start at Containers, Barge, Caissons or Fontao; mark bait balls on sonar.",
        "bait_img": "images/bait_bait.png", "trace_img": "images/trace_bait.png", "setup_img": "images/setup_bait.png",
    },
    "general": {
        "bait": "Mackerel, sardine, squid/chokka and live bait if you can get it.",
        "trace": "Carry: couta wire trace, running trace, fluorocarbon leader and sabiki bait rig.",
        "setup": "Bait first, then fish structure edges and current lines. Move if no bait shows.",
        "bait_img": "images/bait_general.png", "trace_img": "images/trace_general.png", "setup_img": "images/setup_general.png",
    },
}

def image_or_note(path_str, caption):
    path = ROOT / path_str
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Add image: {path_str}")

def render_species_guide(target):
    guide = SPECIES_GUIDE.get(str(target).lower(), SPECIES_GUIDE["general"])
    st.subheader("🎣 Bait, setup and trace for selected target")
    st.markdown(f"""
<div class="best-card">
<h3>{str(target).upper()} setup</h3>
<div><b>Bait:</b> {guide['bait']}</div>
<div><b>Trace:</b> {guide['trace']}</div>
<div><b>Boat setup:</b> {guide['setup']}</div>
</div>
""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        image_or_note(guide["bait_img"], "Bait suggestion")
    with c2:
        image_or_note(guide["trace_img"], "Trace")
    with c3:
        image_or_note(guide["setup_img"], "Setup")

# ---------------------------
# Map
# ---------------------------
def render_map(spots_view, stops, drift_lines, ais_df, launch_lat, launch_lon, selected_spot=None):
    try:
        import folium
        from folium.plugins import HeatMap, MarkerCluster
        from streamlit_folium import st_folium
    except Exception:
        st.info("Install maps: pip install folium streamlit-folium")
        st.map(spots_view[["lat", "lon"]])
        return
    m = folium.Map(location=[launch_lat, launch_lon], zoom_start=11, tiles="OpenStreetMap")
    folium.Marker([launch_lat, launch_lon], tooltip="Launch", popup="Durban launch reference", icon=folium.Icon(color="green", icon="flag")).add_to(m)

    if selected_spot is not None:
        sel_lat = float(selected_spot["lat"])
        sel_lon = float(selected_spot["lon"])
        sel_name = str(selected_spot["name"])
        sel_dist = haversine_km(launch_lat, launch_lon, sel_lat, sel_lon)
        folium.Marker([sel_lat, sel_lon], tooltip=f"SELECTED: {sel_name}", popup=f"<b>Selected destination</b><br>{sel_name}<br>{sel_dist:.1f} km from launch", icon=folium.Icon(color="red", icon="star")).add_to(m)
        folium.PolyLine([[launch_lat, launch_lon], [sel_lat, sel_lon]], tooltip=f"Route to {sel_name}: {sel_dist:.1f} km", weight=4, opacity=0.85).add_to(m)

    cluster = MarkerCluster(name="Fishing marks").add_to(m)
    for _, r in spots_view.iterrows():
        popup = f"<b>{r['name']}</b><br>Score: {r['score']}/100<br>Depth: {r.get('depth','Unknown')}<br>Distance: {r['distance_km']} km<br>Boat stops nearby: {r.get('boat_stop_count_nearby',0)}<br><a href='{google_maps_url(r['lat'], r['lon'])}' target='_blank'>Google Maps</a> | <a href='{navionics_url(r['lat'], r['lon'])}' target='_blank'>Navionics</a><br>{r.get('strategy_note','')}"
        folium.Marker([r["lat"], r["lon"]], tooltip=f"{r['name']} | {r['score']}", popup=popup).add_to(cluster)

    if not stops.empty:
        HeatMap([[r["lat"], r["lon"], max(1, r["stop_points"])] for _, r in stops.iterrows()], name="Auto boat heatmap", radius=28, blur=18, min_opacity=0.35).add_to(m)
        for _, r in stops.head(30).iterrows():
            popup = f"<b>Detected fishing/slow zone</b><br>Stop points: {r['stop_points']}<br>Vessels: {r['vessels']}<br>Sources: {r['sources']}<br>Avg speed: {r['avg_speed_knots']:.1f} kn"
            folium.CircleMarker([r["lat"], r["lon"]], radius=7, fill=True, popup=popup, tooltip="Boat stop zone").add_to(m)

    if not ais_df.empty:
        latest = ais_df.sort_values("timestamp").groupby(["mmsi", "vessel_name"], dropna=False).tail(1).head(60)
        for _, r in latest.iterrows():
            folium.CircleMarker([r["lat"], r["lon"]], radius=3, fill=True, tooltip=f"{r['vessel_name']} {r.get('speed_knots','')} kn", popup=f"{r['vessel_name']}<br>{r['source']}<br>{r.get('timestamp','')}").add_to(m)

    for line in drift_lines:
        folium.PolyLine([line["start"], line["end"]], tooltip=f"Estimated drift {line['bearing']:.0f}° / {line['distance_km']:.1f} km", weight=3).add_to(m)
        folium.Marker(line["end"], icon=folium.Icon(color="blue", icon="arrow-down"), tooltip="Estimated drift end").add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, width=None, height=650)

# ---------------------------
# ---------------------------
# UI
# ---------------------------
st.title("🎣 CastIQ Pro Durban")
st.caption("Durban boat fishing intelligence: recommendations first, map below, mobile-ready.")

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1100px;}
[data-testid="stSidebar"] {display: none;}
.big-card {border: 1px solid rgba(49,51,63,.18); border-radius: 18px; padding: 16px; margin-bottom: 12px;}
.best-card {border: 2px solid rgba(255,75,75,.55); border-radius: 22px; padding: 18px; margin-bottom: 14px; background: rgba(255,75,75,.06);}
.small-muted {opacity: .78; font-size: .92rem;}
div.stButton > button {width: 100%; min-height: 48px; border-radius: 14px; font-weight: 700;}
div[data-testid="stMetric"] {border: 1px solid rgba(49,51,63,.14); border-radius: 16px; padding: 10px;}
@media (max-width: 768px) {
  .block-container {padding-left: .75rem; padding-right: .75rem;}
  h1 {font-size: 1.65rem !important;}
  h2, h3 {font-size: 1.25rem !important;}
}
</style>
""", unsafe_allow_html=True)

if "selected_destination_name" not in st.session_state:
    st.session_state.selected_destination_name = ""

with st.expander("⚙️ Trip setup", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        launch_lat = st.number_input("Launch latitude", value=DURBAN_LAUNCH[0], format="%.6f")
    with c2:
        launch_lon = st.number_input("Launch longitude", value=DURBAN_LAUNCH[1], format="%.6f")
    c3, c4 = st.columns(2)
    with c3:
        target = st.selectbox("Target species", ["couta", "tuna", "snoek", "dorade", "kob", "general", "bait"], index=5)
    with c4:
        max_spots = st.slider("Show recommendations", 3, 20, 8)

with st.expander("🛰️ Boat intelligence / AIS upload", expanded=False):
    provider = st.selectbox("Live provider", ["None / CSV upload", "AISHub live", "MarineTraffic placeholder"])
    uploaded = st.file_uploader("Upload AIS/GPS CSV", type=["csv"])
    c5, c6, c7 = st.columns(3)
    with c5:
        max_speed = st.slider("Slow speed threshold", 1.0, 6.0, 3.0, 0.5)
    with c6:
        min_points = st.slider("Min repeated stop points", 1, 8, 2)
    with c7:
        radius_m = st.slider("Cluster radius metres", 250, 1500, 550, 50)
    drift_hours = st.slider("Drift projection hours", 0.5, 4.0, 1.5, 0.5)

spots = load_spots()
w, m = get_open_meteo(launch_lat, launch_lon)
cur_w = w.get("current", {}) if isinstance(w, dict) else {}
cur_m = m.get("current", {}) if isinstance(m, dict) else {}
wind_speed = float(cur_w.get("wind_speed_10m", 0) or 0)
wind_dir = float(cur_w.get("wind_direction_10m", 0) or 0)
gust = float(cur_w.get("wind_gusts_10m", 0) or 0)
wave_h = float(cur_m.get("wave_height", 0) or 0)
wave_dir = float(cur_m.get("wave_direction", wind_dir) or wind_dir)
wave_period = float(cur_m.get("wave_period", 0) or 0)

live_df = pd.DataFrame()
if provider == "AISHub live":
    username = get_secret("AISHUB_USERNAME")
    if username:
        try:
            live_df = fetch_aishub_live(username, DURBAN_BBOX)
            save_live_cache(live_df)
            st.success(f"AISHub live pull complete: {len(live_df)} records cached.")
        except Exception as e:
            st.warning(f"AISHub live pull failed. Using cached/upload/manual data. Detail: {e}")
    else:
        st.warning("Add AISHUB_USERNAME to Streamlit secrets or environment variables.")
elif provider == "MarineTraffic placeholder":
    if get_secret("MARINETRAFFIC_API_KEY"):
        st.info("MarineTraffic key found. Add your subscribed endpoint URL shape before live pull can be enabled.")
    else:
        st.warning("Add MARINETRAFFIC_API_KEY after you subscribe to the correct MarineTraffic AIS service.")

ais_df = load_cached_and_manual_tracks(uploaded)
if not live_df.empty:
    ais_df = pd.concat([ais_df, live_df], ignore_index=True)
    ais_df = normalise_columns(ais_df)

st.subheader("🌊 Conditions now")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Wind", f"{wind_speed:.0f} km/h", f"gust {gust:.0f}")
c2.metric("Direction", f"{wind_dir:.0f}°")
c3.metric("Wave", f"{wave_h:.1f} m")
c4.metric("Period", f"{wave_period:.0f} sec")

stops = detect_fishing_stops(ais_df, max_speed=max_speed, min_points=min_points, radius_m=radius_m)
spots = enrich_spots(spots, stops)

scores, distances = [], []
for _, r in spots.iterrows():
    s, d = score_spot(r, wind_speed, gust, wave_h, wave_period, launch_lat, launch_lon, target)
    scores.append(s)
    distances.append(round(d, 1))
spots["score"] = scores
spots["distance_km"] = distances
spots_view = spots.sort_values("score", ascending=False).head(max_spots).copy()

render_species_guide(target)

available_names = [str(x) for x in spots_view["name"].tolist()]
if available_names and (not st.session_state.selected_destination_name or st.session_state.selected_destination_name not in available_names):
    st.session_state.selected_destination_name = available_names[0]

selected_spot = None
selected_rows = spots_view[spots_view["name"].astype(str) == st.session_state.selected_destination_name]
if not selected_rows.empty:
    selected_spot = selected_rows.iloc[0].to_dict()

st.subheader("🎯 Recommendations")
if selected_spot is not None:
    st.markdown(f"""
<div class="best-card">
<h3>🔥 Active destination: {selected_spot['name']}</h3>
<div class="small-muted">Score {int(selected_spot['score'])}/100 · Depth {selected_spot.get('depth','Unknown')} · {float(selected_spot['distance_km']):.1f} km from launch · Coordinates {float(selected_spot['lat']):.6f}, {float(selected_spot['lon']):.6f}</div>
</div>
""", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.link_button("🧭 Open in Google Maps", google_maps_url(selected_spot["lat"], selected_spot["lon"]), use_container_width=True)
    with b:
        st.link_button("🌊 Open in Navionics", navionics_url(selected_spot["lat"], selected_spot["lon"]), use_container_width=True)

for i, (_, r) in enumerate(spots_view.iterrows(), start=1):
    is_selected = selected_spot is not None and str(r["name"]) == str(selected_spot["name"])
    with st.container(border=True):
        st.markdown(f"""
### {'✅' if is_selected else '📍'} #{i} {r['name']} — {int(r['score'])}/100
**Coordinates:** `{float(r['lat']):.6f}, {float(r['lon']):.6f}`  
**Depth:** `{r.get('depth','Unknown')}` · **Distance:** `{float(r['distance_km']):.1f} km` · **Boat stops nearby:** `{r.get('boat_stop_count_nearby', 0)}`  
{r.get('strategy_note','')}
""")
        if is_selected:
            st.success("Selected. The map route below is aligned to this spot.")
        else:
            if st.button("🎯 Use this spot", key=f"use_{i}_{str(r['name']).replace(' ', '_').replace('/', '_')}"):
                st.session_state.selected_destination_name = str(r["name"])
                st.rerun()

drift_base = stops if not stops.empty else spots_view.rename(columns={"name": "vessels"}).assign(stop_points=1)
drift_lines = compute_drift_lines(drift_base, wind_dir, wind_speed, wave_dir, wave_h, hours=drift_hours)

st.subheader("🗺️ Map, route, heatmap and drift")
st.caption("Map is below the recommendations for mobile use. Selected destination is red; route is drawn automatically.")
render_map(spots_view, stops, drift_lines, ais_df, launch_lat, launch_lon, selected_spot=selected_spot)

st.divider()
t1, t2, t3 = st.tabs(["Detected boat zones", "AIS/GPS records", "Setup notes"])
with t1:
    if stops.empty:
        st.info("No repeated slow/stop zones detected yet. Upload vessel_tracks.csv or enable AISHub live.")
    else:
        st.dataframe(stops, use_container_width=True)
        st.download_button("Download detected_hot_zones.csv", stops.to_csv(index=False), "detected_hot_zones.csv", "text/csv")
with t2:
    st.write(f"Records loaded: {len(ais_df)}")
    if not ais_df.empty:
        st.dataframe(ais_df.sort_values("timestamp", ascending=False).head(500), use_container_width=True)
        st.download_button("Download combined_vessel_tracks.csv", ais_df.to_csv(index=False), "combined_vessel_tracks.csv", "text/csv")
with t3:
    st.markdown(r"""
### Local folder
`C:\Users\Admin\Desktop\Durban Fishing`

### Cloud deployment files
Keep these at the root of your GitHub repo:
- `app.py`
- `requirements.txt`
- `durban_boat_fishing_spots.csv`
- `data/vessel_tracks.csv`

### Manual CSV format
```csv
vessel_name,mmsi,timestamp,lat,lon,speed_knots,course_deg,source
Example Boat,123456789,2026-05-01 07:15,-29.812,31.095,2.1,44,manual sighting
```

Planning tool only. Verify weather, surf-launch risk, skipper limits, fuel, comms, charts and legal boundaries before going offshore.
""")
