import urllib.request
import urllib.parse
import json
import re
import os
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import difflib

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
ESPN_BASE = "http://site.api.espn.com/apis/site/v2/sports/mma/ufc"
EVENTS_RECORD_PATH = os.path.join(os.path.dirname(__file__), "data", "events_record.json")


def refresh_events_record():
    """Fetch all events (6 months back → 6 months forward) and save to disk."""
    start = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
    end   = (datetime.now() + timedelta(days=180)).strftime("%Y%m%d")
    data  = _get_json(f"{ESPN_BASE}/scoreboard?dates={start}-{end}")

    records = []
    for ev in data.get("events", []):
        comp      = ev.get("competitions", [{}])[0]
        venue     = comp.get("venue", {})
        completed = comp.get("status", {}).get("type", {}).get("completed", False)
        status    = comp.get("status", {}).get("type", {}).get("name", "")
        city      = venue.get("address", {}).get("city", "")
        country   = venue.get("address", {}).get("country", "")

        bouts = []
        for bout in ev.get("competitions", []):
            fighters = bout.get("competitors", [])
            if len(fighters) < 2:
                continue
            winner = ""
            for f in fighters:
                if f.get("winner"):
                    winner = f["athlete"]["displayName"]
            r_ath = fighters[0]["athlete"]
            b_ath = fighters[1]["athlete"]
            r_id  = fighters[0].get("id", "")
            b_id  = fighters[1].get("id", "")
            bouts.append({
                "r_fighter":    r_ath.get("displayName", ""),
                "b_fighter":    b_ath.get("displayName", ""),
                "r_record":     fighters[0].get("records", [{}])[0].get("summary", ""),
                "b_record":     fighters[1].get("records", [{}])[0].get("summary", ""),
                "r_flag":       r_ath.get("flag", {}).get("href", ""),
                "b_flag":       b_ath.get("flag", {}).get("href", ""),
                "r_img":        f"https://a.espncdn.com/i/headshots/mma/players/full/{r_id}.png" if r_id else "",
                "b_img":        f"https://a.espncdn.com/i/headshots/mma/players/full/{b_id}.png" if b_id else "",
                "weight_class": bout.get("type", {}).get("abbreviation", ""),
                "is_title":     "title" in bout.get("type", {}).get("abbreviation", "").lower(),
                "winner":       winner,
                "method":       bout.get("status", {}).get("type", {}).get("shortDetail", ""),
                "r_url": "", "b_url": "",
            })

        records.append({
            "id":        ev.get("id", ""),
            "name":      ev.get("name", ""),
            "date":      ev.get("date", "")[:10],
            "location":  f"{city}, {country}".strip(", "),
            "status":    status,
            "completed": completed,
            "bouts":     bouts,
        })

    os.makedirs(os.path.dirname(EVENTS_RECORD_PATH), exist_ok=True)
    with open(EVENTS_RECORD_PATH, "w") as f:
        json.dump(records, f, indent=2)
    return records


def _load_events_record():
    """Load from disk, refreshing if file is older than 6 hours."""
    try:
        age = (datetime.now().timestamp() - os.path.getmtime(EVENTS_RECORD_PATH)) / 3600
        if age < 6:
            with open(EVENTS_RECORD_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return refresh_events_record()


def _get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    return urllib.request.urlopen(req, timeout=10).read().decode("utf-8")


def _get_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def _parse_espn_event(ev):
    """Parse an ESPN event dict into our standard format."""
    comp = ev.get("competitions", [{}])[0]
    venue = comp.get("venue", {})
    city    = venue.get("address", {}).get("city", "")
    country = venue.get("address", {}).get("country", "")
    location = f"{city}, {country}".strip(", ")

    date_str = ev.get("date", "")
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        date_fmt = dt.strftime("%B %d, %Y")
    except Exception:
        date_fmt = date_str[:10]

    return {
        "name":     ev.get("name", ""),
        "date":     date_fmt,
        "location": location,
        "url":      ev.get("id", ""),   # ESPN event ID used as key
        "_espn":    ev,                 # full data cached for get_event_fights
    }


def _parse_espn_bouts(ev):
    """Extract list of fights from an ESPN event dict."""
    fights = []
    for bout in ev.get("competitions", []):
        fighters = bout.get("competitors", [])
        if len(fighters) < 2:
            continue
        r = fighters[0]["athlete"]
        b = fighters[1]["athlete"]
        wc  = bout.get("type", {}).get("abbreviation", "")
        is_title = "title" in wc.lower() or "championship" in wc.lower()

        # winner (for completed bouts)
        winner = ""
        for f in fighters:
            if f.get("winner"):
                winner = f["athlete"]["displayName"]

        # method/round from status notes if available
        method = bout.get("status", {}).get("type", {}).get("shortDetail", "")
        round_num = ""

        fights.append({
            "r_fighter":   r.get("displayName", ""),
            "b_fighter":   b.get("displayName", ""),
            "weight_class": wc,
            "is_title":    is_title,
            "winner":      winner,
            "method":      method,
            "round":       round_num,
            "r_url":       "",
            "b_url":       "",
        })
    return fights


def _record_to_event(r):
    """Convert a stored record entry to the app's event dict format."""
    try:
        dt = datetime.strptime(r["date"], "%Y-%m-%d")
        date_fmt = dt.strftime("%B %d, %Y")
    except Exception:
        date_fmt = r["date"]
    return {
        "name":      r["name"],
        "date":      date_fmt,
        "location":  r["location"],
        "url":       r["id"],
        "completed": r.get("completed", False),
        "bouts":     r.get("bouts", []),
    }


def get_all_events():
    """Return all events (past 6 months + future 6 months) from record."""
    records = _load_events_record()
    return [_record_to_event(r) for r in records]


def get_upcoming_events():
    """Return upcoming (not yet completed) events."""
    return [e for e in get_all_events() if not e["completed"]]


def get_recent_completed_events(n=3):
    """Return last n completed events, newest first."""
    completed = [e for e in get_all_events() if e["completed"]]
    completed.sort(key=lambda e: e["date"], reverse=True)
    return completed[:n]


def get_event_fights(event_id_or_ev):
    """Return fights for an event (accepts event dict or ESPN event ID)."""
    if isinstance(event_id_or_ev, dict):
        return event_id_or_ev.get("bouts", [])
    # Fallback: look up by ID in record
    records = _load_events_record()
    for r in records:
        if str(r["id"]) == str(event_id_or_ev):
            return r.get("bouts", [])
    return []


def get_completed_event_results(event_id_or_ev):
    """Return results for a completed event."""
    return get_event_fights(event_id_or_ev)


# ── Fighter stats (ufcstats scraping kept as fallback) ────────────────────────

def _parse_height_cm(ht_str):
    m = re.match(r"(\d+)'\s*(\d+)", ht_str)
    if m:
        return int(m.group(1)) * 30.48 + int(m.group(2)) * 2.54
    return None


def _parse_reach_cm(reach_str):
    m = re.match(r"([\d.]+)", reach_str)
    if m:
        return float(m.group(1)) * 2.54
    return None


def _parse_weight_kg(wt_str):
    m = re.match(r"([\d.]+)", wt_str)
    if m:
        return float(m.group(1)) * 0.453592
    return None


def _parse_age(dob_str):
    try:
        dob   = datetime.strptime(dob_str.strip(), "%b %d, %Y")
        today = datetime.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception:
        return None


def scrape_fighter_stats(fighter_url):
    """Scrape live stats from a fighter's ufcstats page (fallback only)."""
    if not fighter_url:
        return {}
    try:
        html  = _get(fighter_url)
        soup  = BeautifulSoup(html, "lxml")
        stats = {}

        for item in soup.select("li.b-list__box-list-item"):
            text  = item.get_text(separator=":", strip=True)
            parts = [p.strip() for p in text.split(":") if p.strip()]
            if len(parts) >= 2:
                key, val = parts[0].lower(), parts[1]
                if "height"   in key: stats["height"]      = _parse_height_cm(val)
                elif "weight" in key: stats["weight"]      = _parse_weight_kg(val)
                elif "reach"  in key: stats["reach"]       = _parse_reach_cm(val)
                elif "stance" in key: stats["stance"]      = val
                elif "dob"    in key: stats["age"]         = _parse_age(val)
                elif "slpm"   in key: stats["SLpM"]        = float(val) if val.replace(".", "").isdigit() else None
                elif "str. acc" in key: stats["sig_str_acc"] = float(val.strip("%")) / 100 if "%" in val else None
                elif "sapm"   in key: stats["SApM"]        = float(val) if val.replace(".", "").isdigit() else None
                elif "str. def" in key: stats["str_def"]   = float(val.strip("%")) / 100 if "%" in val else None
                elif "td avg" in key: stats["td_avg"]      = float(val) if val.replace(".", "").isdigit() else None
                elif "td acc" in key: stats["td_acc"]      = float(val.strip("%")) / 100 if "%" in val else None
                elif "td def" in key: stats["td_def"]      = float(val.strip("%")) / 100 if "%" in val else None
                elif "sub. avg" in key: stats["sub_avg"]   = float(val) if val.replace(".", "").isdigit() else None

        record_el = soup.select_one("span.b-content__title-record")
        if record_el:
            m = re.search(r"(\d+)-(\d+)", record_el.get_text(strip=True))
            if m:
                stats["wins"]   = int(m.group(1))
                stats["losses"] = int(m.group(2))

        return stats
    except Exception:
        return {}


def find_fighter_url(fighter_name):
    """Search ufcstats for a fighter URL (fallback only)."""
    last_name = fighter_name.strip().split()[-1]
    char = last_name[0].lower()
    try:
        html  = _get(f"http://ufcstats.com/statistics/fighters?char={char}&page=all")
        soup  = BeautifulSoup(html, "lxml")
        rows  = soup.select("tr.b-statistics__table-row")
        candidates = []
        for row in rows:
            cols = row.select("td")
            if len(cols) < 2:
                continue
            first = cols[0].get_text(strip=True)
            last  = cols[1].get_text(strip=True)
            full  = f"{first} {last}".strip()
            link  = row.select_one("a.b-link")
            if link:
                candidates.append((full, link.get("href", "")))
        names   = [c[0] for c in candidates]
        matches = difflib.get_close_matches(fighter_name, names, n=1, cutoff=0.6)
        if matches:
            for full, url in candidates:
                if full == matches[0]:
                    return url
    except Exception:
        pass
    return ""
