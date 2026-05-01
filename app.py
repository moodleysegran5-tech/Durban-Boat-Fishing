
import math
import os
from pathlib import Path
from datetime import datetime, date, time, timedelta

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="CastIQ Pro – Durban Boat Intelligence V8",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path.cwd()
SPOTS_CSV = ROOT / "durban_boat_fishing_spots.csv"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
TRACKS_CSV = DATA_DIR / "vessel_tracks.csv"
CACHE_CSV = DATA_DIR / "live_ais_cache.csv"

DURBAN_LAUNCH = (-29.8689, 31.0617)
DURBAN_BBOX = {"min_lat": -29.98, "max_lat": -29.55, "min_lon": 30.95, "max_lon": 31.28}

PAYGATE_STANDARD_URL = os.getenv("PAYGATE_STANDARD_URL", "https://secure.paygate.co.za/payweb3/process.trans?REFERENCE=CASTIQ_STD_1000")
PAYGATE_PREMIUM_URL = os.getenv("PAYGATE_PREMIUM_URL", "https://secure.paygate.co.za/payweb3/process.trans?REFERENCE=CASTIQ_PREM_20000")


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
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def destination_point(lat, lon, bearing_deg, distance_km):
    R = 6371.0
    brng = math.radians(float(bearing_deg))
    lat1 = math.radians(float(lat))
    lon1 = math.radians(float(lon))
    d = float(distance_km) / R
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d) * math.cos(lat1), math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def google_maps_url(lat, lon):
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"


def navionics_url(lat, lon):
    return f"https://webapp.navionics.com/?lat={lat}&lng={lon}&zoom=13"


def clean_key(txt):
    return str(txt).replace(" ", "_").replace("/", "_").replace("'", "").replace('"', "").replace(".", "_")


def safe_float(v, default=0.0):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


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
        st.error("Missing durban_boat_fishing_spots.csv. Keep it in the same GitHub root as app.py.")
        st.stop()

    df = pd.read_csv(SPOTS_CSV)
    df.columns = [str(c).strip().lower() for c in df.columns]

    for col in ["name", "lat", "lon"]:
        if col not in df.columns:
            st.error(f"Your CSV must include column: {col}")
            st.stop()

    if "depth" not in df.columns:
        df["depth"] = "Unknown"
    if "type" not in df.columns:
        df["type"] = "spot"
    if "target_species" not in df.columns:
        df["target_species"] = "general"
    if "strategy_note" not in df.columns:
        df["strategy_note"] = ""

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["lat", "lon"])


@st.cache_data(ttl=900)
def get_open_meteo(lat=-29.86, lon=31.06):
    weather_url = "https://api.open-meteo.com/v1/forecast"
    marine_url = "https://marine-api.open-meteo.com/v1/marine"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation",
        "current": "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "timezone": "Africa/Johannesburg",
        "forecast_days": 2,
    }
    marine_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wave_height,wave_direction,wave_period",
        "current": "wave_height,wave_direction,wave_period",
        "timezone": "Africa/Johannesburg",
        "forecast_days": 2,
    }
    try:
        w = requests.get(weather_url, params=weather_params, timeout=12).json()
        m = requests.get(marine_url, params=marine_params, timeout=12).json()
        return w, m
    except Exception:
        return {}, {}


# ---------------------------
# NOAA Ocean intelligence
# ---------------------------
@st.cache_data(ttl=21600)
def fetch_noaa_sst(lat, lon):
    """
    NOAA ERDDAP MUR SST lookup.
    Returns approximate Celsius SST near the selected coordinate.
    """
    base = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.csv"
    # broad fallback: latest time near point. ERDDAP supports last() but some mirrors are strict.
    queries = [
        f"{base}?analysed_sst%5B(last)%5D%5B({lat})%5D%5B({lon})%5D",
        f"{base}?analysed_sst[(last)][({lat})][({lon})]",
    ]
    for url in queries:
        try:
            r = requests.get(url, timeout=18)
            if r.status_code != 200:
                continue
            lines = [x for x in r.text.splitlines() if x.strip()]
            if len(lines) >= 3:
                val = pd.to_numeric(lines[-1].split(",")[-1], errors="coerce")
                if pd.notna(val):
                    # MUR is usually Kelvin.
                    c = float(val) - 273.15 if float(val) > 100 else float(val)
                    return round(c, 2), "live"
        except Exception:
            pass
    return None, "unavailable"


@st.cache_data(ttl=21600)
def fetch_noaa_chlorophyll(lat, lon):
    """
    NOAA CoastWatch ERDDAP chlorophyll lookup.
    Uses CoastWatch monthly/daily chlorophyll dataset when available.
    """
    possible = [
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chlamday.csv?chlorophyll%5B(last)%5D%5B({lat})%5D%5B({lon})%5D",
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chlamday.csv?chlor_a%5B(last)%5D%5B({lat})%5D%5B({lon})%5D",
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chlamday.csv?chlorophyll[(last)][({lat})][({lon})]",
    ]
    for template in possible:
        url = template.format(lat=lat, lon=lon)
        try:
            r = requests.get(url, timeout=18)
            if r.status_code != 200:
                continue
            lines = [x for x in r.text.splitlines() if x.strip()]
            if len(lines) >= 3:
                val = pd.to_numeric(lines[-1].split(",")[-1], errors="coerce")
                if pd.notna(val):
                    return round(float(val), 4), "live"
        except Exception:
            pass
    return None, "unavailable"


def ocean_signal_score(sst_c, chl):
    score = 50
    notes = []

    if sst_c is None:
        notes.append("SST unavailable")
    else:
        if 22 <= sst_c <= 27:
            score += 20
            notes.append(f"SST {sst_c:.1f}°C favourable")
        elif 19 <= sst_c < 22 or 27 < sst_c <= 29:
            score += 8
            notes.append(f"SST {sst_c:.1f}°C usable")
        else:
            score -= 8
            notes.append(f"SST {sst_c:.1f}°C less ideal")

    if chl is None:
        notes.append("chlorophyll unavailable")
    else:
        if 0.05 <= chl <= 1.5:
            score += 20
            notes.append(f"chlorophyll {chl:.3f} suggests bait productivity")
        elif 1.5 < chl <= 4:
            score += 6
            notes.append(f"chlorophyll {chl:.3f} productive but possibly greener water")
        else:
            score -= 5
            notes.append(f"chlorophyll {chl:.3f} weak/too high signal")

    return max(0, min(100, int(score))), " · ".join(notes)


def enrich_ocean_for_spots(spots_df, enabled=True):
    df = spots_df.copy()
    ssts, chls, ocean_scores, ocean_notes = [], [], [], []
    statuses = []
    for _, r in df.iterrows():
        if enabled:
            sst, sst_status = fetch_noaa_sst(float(r["lat"]), float(r["lon"]))
            chl, chl_status = fetch_noaa_chlorophyll(float(r["lat"]), float(r["lon"]))
        else:
            sst, chl, sst_status, chl_status = None, None, "off", "off"
        oscore, note = ocean_signal_score(sst, chl)
        ssts.append(sst)
        chls.append(chl)
        ocean_scores.append(oscore)
        ocean_notes.append(note)
        statuses.append(f"SST:{sst_status} / CHL:{chl_status}")
    df["sst_c"] = ssts
    df["chlorophyll"] = chls
    df["ocean_score"] = ocean_scores
    df["ocean_note"] = ocean_notes
    df["ocean_status"] = statuses
    return df


# ---------------------------
# AIS / vessel intelligence
# ---------------------------
def fetch_aishub_live(username, bbox):
    if not username:
        return pd.DataFrame()
    url = "https://data.aishub.net/ws.php"
    params = {
        "username": username,
        "format": 1,
        "output": "json",
        "compress": 0,
        "latmin": bbox["min_lat"],
        "latmax": bbox["max_lat"],
        "lonmin": bbox["min_lon"],
        "lonmax": bbox["max_lon"],
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
        return pd.DataFrame(columns=["vessel_name", "mmsi", "timestamp", "lat", "lon", "speed_knots", "course_deg", "source"])
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
    wind_to = (float(wind_dir) + 180) % 360 if wind_dir is not None else 45
    swell_to = float(wave_dir) if wave_dir is not None else wind_to
    drift_bearing = (0.55 * swell_to + 0.45 * wind_to) % 360
    speed_kmh = max(0.4, min(3.2, 0.08 * (float(wind_speed) or 10) + 0.55 * (float(wave_h) or 1.0)))
    distance_km = speed_kmh * float(hours)
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


# ---------------------------
# Scoring + smart schedule
# ---------------------------
def score_spot(row, wind_speed, gust, wave_h, wave_period, launch_lat, launch_lon, target):
    distance = haversine_km(launch_lat, launch_lon, row["lat"], row["lon"])
    safety = 100

    if wind_speed > 28:
        safety -= 35
    elif wind_speed > 20:
        safety -= 20
    elif wind_speed > 14:
        safety -= 10

    if gust > 35:
        safety -= 25
    elif gust > 28:
        safety -= 15

    if wave_h > 2.0:
        safety -= 35
    elif wave_h > 1.5:
        safety -= 20
    elif wave_h > 1.1:
        safety -= 10

    if wave_period and wave_period < 8:
        safety -= 10

    if distance > 28:
        safety -= 15
    elif distance > 15:
        safety -= 8

    type_txt = str(row.get("type", "")).lower()
    species_txt = (str(row.get("target_species", "")) + " " + str(row.get("strategy_note", ""))).lower()
    bait_bonus = 20 if "bait" in type_txt or "bait" in species_txt else 8
    target_bonus = 18 if target.lower() in species_txt else 7
    structure_bonus = 14 if any(x in type_txt for x in ["reef", "wreck", "drop-off", "artificial"]) else 5
    crowd_penalty = 8 if any(x in str(row.get("name", "")).lower() for x in ["fontao", "no.1", "barge"]) else 0
    boat_bonus = int(row.get("boat_intel_bonus", 0) or 0)
    ocean_bonus = int((safe_float(row.get("ocean_score"), 50) - 50) * 0.22)

    total = max(0, min(100, int((safety * 0.34) + bait_bonus + target_bonus + structure_bonus + boat_bonus + ocean_bonus - crowd_penalty)))
    return total, distance


def estimate_travel_minutes(distance_km, speed_kmh=32):
    if speed_kmh <= 0:
        speed_kmh = 32
    return max(8, int((distance_km / speed_kmh) * 60))


def tide_phase_score(mode):
    mode = str(mode).lower()
    if "push" in mode or "incoming" in mode:
        return 12, "incoming/pushing tide window favours bait movement"
    if "out" in mode:
        return 6, "outgoing tide usable but monitor dirty water/current"
    if "slack" in mode:
        return -4, "slack tide can reduce movement; use structure and bait"
    return 0, "manual tide not specified"


def build_smart_schedule(start_dt, end_dt, spots_view, launch_lat, launch_lon, wind_speed, wave_h, tide_mode, target):
    schedule = []
    if end_dt <= start_dt:
        return schedule

    total_minutes = int((end_dt - start_dt).total_seconds() / 60)
    if total_minutes < 60:
        return schedule

    ranked = spots_view.sort_values("score", ascending=False).head(5).copy()
    if ranked.empty:
        return schedule

    tide_bonus, tide_note = tide_phase_score(tide_mode)
    ranked["schedule_priority"] = ranked["score"] + tide_bonus + ranked["ocean_score"].fillna(50).astype(float) * 0.08
    ranked = ranked.sort_values("schedule_priority", ascending=False)

    current = start_dt

    bait_candidates = ranked[
        ranked["type"].astype(str).str.lower().str.contains("bait", na=False)
        | ranked["target_species"].astype(str).str.lower().str.contains("bait", na=False)
        | ranked["strategy_note"].astype(str).str.lower().str.contains("bait", na=False)
    ]

    # Add bait stop when target benefits from live bait and enough time exists
    if target in ["couta", "tuna", "dorade", "snoek", "general"] and total_minutes >= 180:
        bait = bait_candidates.iloc[0] if not bait_candidates.empty else ranked.iloc[0]
        bait_minutes = 45 if total_minutes < 300 else 60
        end_bait = current + timedelta(minutes=bait_minutes)
        schedule.append({
            "start": current,
            "end": end_bait,
            "spot": bait["name"],
            "phase": "Bait / first scan",
            "depth": bait.get("depth", "Unknown"),
            "lat": bait["lat"],
            "lon": bait["lon"],
            "instruction": "Collect live bait, confirm bait balls on sonar, and only stay longer if predator marks appear.",
            "why": f"Premium logic: bait-first route because {target} usually improves when live bait and current movement align. {tide_note}. Ocean: {bait.get('ocean_note','')}",
        })
        current = end_bait + timedelta(minutes=estimate_travel_minutes(float(bait.get("distance_km", 8)) * 0.15, 30))

    remaining = int((end_dt - current).total_seconds() / 60)
    if remaining <= 30:
        return schedule

    # Allocate time: best spot gets biggest block, second spot gets next, final is fallback
    usable = ranked.head(4).to_dict("records")
    weights = [0.42, 0.30, 0.18, 0.10]
    previous_lat, previous_lon = launch_lat, launch_lon

    for idx, spot in enumerate(usable):
        remaining = int((end_dt - current).total_seconds() / 60)
        if remaining < 45:
            break

        travel = estimate_travel_minutes(haversine_km(previous_lat, previous_lon, spot["lat"], spot["lon"]), 32)
        if schedule:
            current = current + timedelta(minutes=min(travel, 25))

        remaining = int((end_dt - current).total_seconds() / 60)
        if remaining < 45:
            break

        block = int(total_minutes * weights[idx])
        block = max(45, min(block, remaining))
        if idx == len(usable) - 1:
            block = remaining

        end_block = min(current + timedelta(minutes=block), end_dt)

        phase = "Primary drift" if idx == 0 else "Secondary drift" if idx == 1 else "Fallback / move if bite slows"
        instruction = "Drift the structure edge. Reset up-current if bait/fish marks show. Move if no life after 30–40 minutes."
        if idx == 0:
            instruction = "Commit to this window first. Work the reef/current edge and do not chase too early."
        elif idx == 1:
            instruction = "Move only after the primary window. Use this as the pressure-release or second feeding line."

        schedule.append({
            "start": current,
            "end": end_block,
            "spot": spot["name"],
            "phase": phase,
            "depth": spot.get("depth", "Unknown"),
            "lat": spot["lat"],
            "lon": spot["lon"],
            "instruction": instruction,
            "why": f"Selected by premium schedule engine: score {int(spot['score'])}/100, ocean {int(spot.get('ocean_score', 50))}/100, wind {wind_speed:.0f} km/h, swell {wave_h:.1f} m, {tide_note}.",
        })

        current = end_block
        previous_lat, previous_lon = spot["lat"], spot["lon"]

        if current >= end_dt:
            break

    return schedule


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
    st.subheader("🎣 Bait, setup and trace")
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
# Pages
# ---------------------------
def intro_page():
    st.title("🎣 CastIQ Pro Durban")
    st.markdown("""
### Fishing intelligence for Durban offshore and bay boat anglers

CastIQ Pro helps solve a common problem: anglers often launch with static GPS marks, scattered weather apps, uncertain tide timing and limited visibility of bait movement.

The app turns those fragments into one practical decision:
> **Where should I go, when should I fish it, and what should I use?**

It combines reef marks, live marine conditions, vessel activity, drift logic, species bait guidance, ocean intelligence and time-based planning into a structured fishing plan.
""")

    c1, c2, c3 = st.columns(3)
    c1.metric("Decision engine", "Spot + time")
    c2.metric("Premium signals", "SST + CHL + AIS")
    c3.metric("Output", "Fishing schedule")

    st.info("Use the Trip Planner tab to select species, fishing time, and build your recommended schedule.")


def packages_page():
    st.title("💼 Packages")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
<div class="big-card">
<h2>Standard</h2>
<h3>R1,000 / month</h3>
<p>Built for serious recreational anglers who want better planning and fewer wasted trips.</p>
<ul>
<li>Smart fishing spot recommendations</li>
<li>Time-based fishing schedule</li>
<li>Species bait, setup and trace guidance</li>
<li>Weather and marine condition checks</li>
<li>Basic boat heatmap / upload mode</li>
<li>Mobile access</li>
</ul>
</div>
""", unsafe_allow_html=True)
        st.link_button("💳 Subscribe Standard", PAYGATE_STANDARD_URL, use_container_width=True)

    with c2:
        st.markdown("""
<div class="best-card">
<h2>Premium</h2>
<h3>R20,000 once-off</h3>
<p>Professional-grade fishing intelligence for advanced anglers, charter operators and offshore teams.</p>
<ul>
<li>Advanced AIS vessel intelligence</li>
<li>NOAA SST and chlorophyll ocean signals</li>
<li>Smart schedule using wind, swell, tide and ocean data</li>
<li>Drift prediction engine</li>
<li>Private catch-history learning layer</li>
<li>Premium “why this spot now” explanations</li>
<li>Priority feature updates</li>
</ul>
</div>
""", unsafe_allow_html=True)
        st.link_button("🚀 Unlock Premium", PAYGATE_PREMIUM_URL, use_container_width=True)


def regulations_page():
    st.title("📜 Regulations & Safety")
    st.warning("Planning guide only. Always verify latest South African marine recreational fishing rules, permits, MPAs, weather and skipper safety requirements before launch.")

    st.markdown("""
### Core reminders
- Carry the correct recreational fishing permit.
- Recreational catch may not be sold.
- Respect species-specific size limits, bag limits and closed seasons.
- Check Marine Protected Areas and restricted zones.
- Confirm boat safety equipment, comms, fuel, surf launch risk and weather window.
""")

    regs = pd.DataFrame([
        {"Topic": "Shad / Elf", "Reminder": "Closed season generally applies 1 September to 30 November. Verify latest rule before keeping fish."},
        {"Topic": "Kob", "Reminder": "Size and bag limits apply. Release undersize fish safely."},
        {"Topic": "Couta", "Reminder": "Bag limits apply. Confirm latest recreational limit."},
        {"Topic": "Tuna", "Reminder": "Species-specific rules may apply depending on tuna type."},
        {"Topic": "Reef fish / Rockcod", "Reminder": "Many reef species have strict size/bag restrictions. Identify species before keeping."},
        {"Topic": "Overall bag", "Reminder": "Overall cumulative daily limits may apply in addition to species-specific limits."},
    ])
    st.dataframe(regs, use_container_width=True, hide_index=True)

    st.subheader("Pre-launch checklist")
    st.checkbox("I checked the latest permit and recreational fishing regulations")
    st.checkbox("I checked size, bag and closed-season rules for my target species")
    st.checkbox("I checked MPAs / restricted areas")
    st.checkbox("I checked weather, swell, wind, fuel and safety equipment")


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
        popup = f"""
        <b>{r['name']}</b><br>
        Score: {r['score']}/100<br>
        Depth: {r.get('depth','Unknown')}<br>
        Distance: {r['distance_km']} km<br>
        Ocean: {r.get('ocean_score','N/A')}/100<br>
        SST: {r.get('sst_c','N/A')}°C<br>
        Chlorophyll: {r.get('chlorophyll','N/A')}<br>
        Boat stops nearby: {r.get('boat_stop_count_nearby',0)}<br>
        <a href='{google_maps_url(r['lat'], r['lon'])}' target='_blank'>Google Maps</a> |
        <a href='{navionics_url(r['lat'], r['lon'])}' target='_blank'>Navionics</a><br>
        {r.get('strategy_note','')}
        """
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


def trip_planner_page():
    st.title("🎣 Trip Planner")
    st.caption("Recommendations first, map at the bottom, mobile-friendly.")

    if "selected_destination_name" not in st.session_state:
        st.session_state.selected_destination_name = ""

    with st.expander("⚙️ Trip setup", expanded=True):
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

        c5, c6, c7 = st.columns(3)
        with c5:
            start_t = st.time_input("Start fishing time", value=time(6, 0))
        with c6:
            end_t = st.time_input("End fishing time", value=time(11, 0))
        with c7:
            tide_mode = st.selectbox("Tide signal", ["Incoming / pushing", "Outgoing", "Slack / unknown"], index=0)

        use_ocean = st.toggle("Use live NOAA SST + chlorophyll scoring", value=True)

    with st.expander("🛰️ Boat intelligence / AIS upload", expanded=False):
        provider = st.selectbox("Live provider", ["None / CSV upload", "AISHub live", "MarineTraffic placeholder"])
        uploaded = st.file_uploader("Upload AIS/GPS CSV", type=["csv"])
        c8, c9, c10 = st.columns(3)
        with c8:
            max_speed = st.slider("Slow speed threshold", 1.0, 6.0, 3.0, 0.5)
        with c9:
            min_points = st.slider("Min repeated stop points", 1, 8, 2)
        with c10:
            radius_m = st.slider("Cluster radius metres", 250, 1500, 550, 50)
        drift_hours = st.slider("Drift projection hours", 0.5, 4.0, 1.5, 0.5)

    spots = load_spots()
    w, m = get_open_meteo(launch_lat, launch_lon)
    cur_w = w.get("current", {}) if isinstance(w, dict) else {}
    cur_m = m.get("current", {}) if isinstance(m, dict) else {}

    wind_speed = safe_float(cur_w.get("wind_speed_10m"), 0)
    wind_dir = safe_float(cur_w.get("wind_direction_10m"), 0)
    gust = safe_float(cur_w.get("wind_gusts_10m"), 0)
    wave_h = safe_float(cur_m.get("wave_height"), 0)
    wave_dir = safe_float(cur_m.get("wave_direction"), wind_dir)
    wave_period = safe_float(cur_m.get("wave_period"), 0)

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
        st.info("MarineTraffic access depends on your subscribed endpoint. Use CSV/manual upload until endpoint is confirmed.")

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

    # Enrich ocean only on candidate shortlist for performance
    base_scores, distances = [], []
    for _, r in spots.iterrows():
        s, d = score_spot(r.assign(ocean_score=50) if hasattr(r, "assign") else r, wind_speed, gust, wave_h, wave_period, launch_lat, launch_lon, target)
        base_scores.append(s)
        distances.append(round(d, 1))
    spots["base_score"] = base_scores
    spots["distance_km"] = distances

    candidates = spots.sort_values("base_score", ascending=False).head(max_spots).copy()
    candidates = enrich_ocean_for_spots(candidates, enabled=use_ocean)

    scores = []
    for _, r in candidates.iterrows():
        s, _ = score_spot(r, wind_speed, gust, wave_h, wave_period, launch_lat, launch_lon, target)
        scores.append(s)
    candidates["score"] = scores
    spots_view = candidates.sort_values("score", ascending=False).head(max_spots).copy()

    render_species_guide(target)

    start_dt = datetime.combine(date.today(), start_t)
    end_dt = datetime.combine(date.today(), end_t)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)

    schedule = build_smart_schedule(start_dt, end_dt, spots_view, launch_lat, launch_lon, wind_speed, wave_h, tide_mode, target)

    st.subheader("🧭 Smart fishing schedule")
    st.markdown("""
<div class="best-card">
<b>Premium schedule engine:</b> This plan uses wind, swell, SST, chlorophyll, tide signal, travel time, species logic and reef/boat intelligence to decide where to spend your fishing window.
</div>
""", unsafe_allow_html=True)

    if not schedule:
        st.warning("Not enough time or data to build a schedule. Extend your start/end window.")
    else:
        for idx, step in enumerate(schedule, start=1):
            with st.container(border=True):
                st.markdown(f"""
### {idx}. {step['phase']}: {step['spot']}
**Time:** `{step['start'].strftime('%H:%M')}` – `{step['end'].strftime('%H:%M')}`  
**Depth:** `{step['depth']}`  
**Coordinates:** `{float(step['lat']):.6f}, {float(step['lon']):.6f}`  
**Instruction:** {step['instruction']}  
**Why this slot:** {step['why']}
""")

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
<div class="small-muted">Score {int(selected_spot['score'])}/100 · Depth {selected_spot.get('depth','Unknown')} · {float(selected_spot['distance_km']):.1f} km from launch · Ocean {int(selected_spot.get('ocean_score', 50))}/100 · Coordinates {float(selected_spot['lat']):.6f}, {float(selected_spot['lon']):.6f}</div>
<div class="small-muted">{selected_spot.get('ocean_note','')}</div>
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
**Depth:** `{r.get('depth','Unknown')}` · **Distance:** `{float(r['distance_km']):.1f} km` · **Ocean:** `{int(r.get('ocean_score', 50))}/100` · **Boat stops nearby:** `{r.get('boat_stop_count_nearby', 0)}`  
**Ocean signal:** {r.get('ocean_note','')}  
{r.get('strategy_note','')}
""")
            if is_selected:
                st.success("Selected. The map route below is aligned to this spot.")
            else:
                if st.button("🎯 Use this spot", key=f"use_{i}_{clean_key(r['name'])}"):
                    st.session_state.selected_destination_name = str(r["name"])
                    st.rerun()

    drift_base = stops if not stops.empty else spots_view.rename(columns={"name": "vessels"}).assign(stop_points=1)
    drift_lines = compute_drift_lines(drift_base, wind_dir, wind_speed, wave_dir, wave_h, hours=drift_hours)

    st.subheader("🗺️ Map, route, heatmap and drift")
    st.caption("Map is below the recommendations for mobile use. Selected destination is red; route is drawn automatically.")
    render_map(spots_view, stops, drift_lines, ais_df, launch_lat, launch_lon, selected_spot=selected_spot)

    st.divider()
    t1, t2, t3, t4 = st.tabs(["Ocean intelligence", "Detected boat zones", "AIS/GPS records", "Setup notes"])
    with t1:
        st.write("NOAA ocean signals for the recommendation shortlist.")
        st.dataframe(spots_view[["name", "lat", "lon", "depth", "sst_c", "chlorophyll", "ocean_score", "ocean_status", "ocean_note"]], use_container_width=True)
        st.download_button("Download ocean_intelligence.csv", spots_view.to_csv(index=False), "ocean_intelligence.csv", "text/csv")
    with t2:
        if stops.empty:
            st.info("No repeated slow/stop zones detected yet. Upload vessel_tracks.csv or enable AISHub live.")
        else:
            st.dataframe(stops, use_container_width=True)
            st.download_button("Download detected_hot_zones.csv", stops.to_csv(index=False), "detected_hot_zones.csv", "text/csv")
    with t3:
        st.write(f"Records loaded: {len(ais_df)}")
        if not ais_df.empty:
            st.dataframe(ais_df.sort_values("timestamp", ascending=False).head(500), use_container_width=True)
            st.download_button("Download combined_vessel_tracks.csv", ais_df.to_csv(index=False), "combined_vessel_tracks.csv", "text/csv")
    with t4:
        st.markdown(r"""
### Cloud deployment files
Keep these at the root of your GitHub repo:
- `app.py`
- `requirements.txt`
- `durban_boat_fishing_spots.csv`
- `data/vessel_tracks.csv`
- `images/` folder

### Manual CSV format
```csv
vessel_name,mmsi,timestamp,lat,lon,speed_knots,course_deg,source
Example Boat,123456789,2026-05-01 07:15,-29.812,31.095,2.1,44,manual sighting
```

Planning tool only. Verify weather, surf-launch risk, skipper limits, fuel, comms, charts and legal boundaries before going offshore.
""")


# ---------------------------
# UI Styling + Mobile Bottom Navigation
# ---------------------------
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 6.2rem; max-width: 1100px;}
[data-testid="stSidebar"] {display: none;}
.big-card {border: 1px solid rgba(49,51,63,.18); border-radius: 18px; padding: 16px; margin-bottom: 12px;}
.best-card {border: 2px solid rgba(255,75,75,.55); border-radius: 22px; padding: 18px; margin-bottom: 14px; background: rgba(255,75,75,.06);}
.small-muted {opacity: .78; font-size: .92rem;}
div.stButton > button {width: 100%; min-height: 48px; border-radius: 14px; font-weight: 700;}
div[data-testid="stMetric"] {border: 1px solid rgba(49,51,63,.14); border-radius: 16px; padding: 10px;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.castiq-bottom-nav {position: fixed; left: 0; right: 0; bottom: 0; z-index: 999999; background: rgba(255,255,255,0.96); backdrop-filter: blur(14px); border-top: 1px solid rgba(49,51,63,.16); box-shadow: 0 -8px 28px rgba(0,0,0,.12); padding: 8px 8px calc(8px + env(safe-area-inset-bottom));}
.castiq-nav-inner {max-width: 760px; margin: 0 auto; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 7px;}
.castiq-nav-item {text-decoration: none !important; color: #30333f !important; text-align: center; border-radius: 18px; padding: 9px 4px 8px; font-size: 0.78rem; line-height: 1.1rem; font-weight: 750; border: 1px solid rgba(49,51,63,.12); background: rgba(248,249,252,.92); min-height: 52px; display: flex; align-items: center; justify-content: center; flex-direction: column;}
.castiq-nav-item span {display: block; font-size: 1.12rem; line-height: 1.25rem;}
.castiq-nav-item.active {color: white !important; background: linear-gradient(135deg, #0f766e, #0ea5e9); border-color: rgba(14,165,233,.45); box-shadow: 0 6px 14px rgba(14,165,233,.28);}
@media (min-width: 900px) {.castiq-bottom-nav {left: 50%; right: auto; transform: translateX(-50%); width: min(760px, calc(100% - 24px)); bottom: 14px; border: 1px solid rgba(49,51,63,.14); border-radius: 28px; padding: 8px;}}
@media (max-width: 768px) {.block-container {padding-left: .75rem; padding-right: .75rem; padding-bottom: 6.8rem;} h1 {font-size: 1.65rem !important;} h2, h3 {font-size: 1.25rem !important;} .castiq-nav-item {font-size: .72rem; min-height: 54px; border-radius: 16px;}}
</style>
""", unsafe_allow_html=True)


def get_current_page():
    try:
        page = st.query_params.get("page", "home")
    except Exception:
        page = "home"
    if isinstance(page, list):
        page = page[0] if page else "home"
    page = str(page).lower().strip()
    return page if page in {"home", "trip", "rules", "packages"} else "home"


def render_bottom_nav(current_page):
    def cls(page):
        return "castiq-nav-item active" if current_page == page else "castiq-nav-item"
    st.markdown(f"""
<div class="castiq-bottom-nav">
  <div class="castiq-nav-inner">
    <a class="{cls('home')}" href="?page=home" target="_self"><span>🏠</span>Home</a>
    <a class="{cls('trip')}" href="?page=trip" target="_self"><span>🎣</span>Plan</a>
    <a class="{cls('rules')}" href="?page=rules" target="_self"><span>📜</span>Rules</a>
    <a class="{cls('packages')}" href="?page=packages" target="_self"><span>💼</span>Packages</a>
  </div>
</div>
""", unsafe_allow_html=True)


current_page = get_current_page()
render_bottom_nav(current_page)

if current_page == "home":
    intro_page()
elif current_page == "trip":
    trip_planner_page()
elif current_page == "rules":
    regulations_page()
elif current_page == "packages":
    packages_page()
