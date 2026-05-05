import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import re
from datetime import datetime
import difflib

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

def _get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    return urllib.request.urlopen(req, timeout=10).read().decode("utf-8")


def get_upcoming_events():
    """Return list of upcoming UFC events: [{name, date, location, url}]"""
    html = _get("http://ufcstats.com/statistics/events/upcoming")
    soup = BeautifulSoup(html, "lxml")
    events = []
    rows = soup.select("tr.b-statistics__table-row")
    for row in rows:
        link = row.select_one("a.b-link")
        if not link:
            continue
        cols = row.select("td")
        name = link.get_text(strip=True)
        url = link.get("href", "")
        date = cols[1].get_text(strip=True) if len(cols) > 1 else ""
        loc = cols[2].get_text(strip=True) if len(cols) > 2 else ""
        if name and url:
            events.append({"name": name, "date": date, "location": loc, "url": url})
    return events


def get_recent_completed_events(n=3):
    """Return last n completed events."""
    html = _get("http://ufcstats.com/statistics/events/completed")
    soup = BeautifulSoup(html, "lxml")
    events = []
    rows = soup.select("tr.b-statistics__table-row")
    for row in rows:
        link = row.select_one("a.b-link")
        if not link:
            continue
        cols = row.select("td")
        name = link.get_text(strip=True)
        url = link.get("href", "")
        date = cols[1].get_text(strip=True) if len(cols) > 1 else ""
        loc = cols[2].get_text(strip=True) if len(cols) > 2 else ""
        if name and url:
            events.append({"name": name, "date": date, "location": loc, "url": url})
        if len(events) >= n:
            break
    return events


def get_event_fights(event_url):
    """Return list of fights for an event: [{r_fighter, b_fighter, weight_class, is_title, r_url, b_url}]"""
    html = _get(event_url)
    soup = BeautifulSoup(html, "lxml")
    fights = []
    rows = soup.select("tr.b-fight-details__table-row")
    for row in rows:
        fighter_links = row.select("a.b-link")
        if len(fighter_links) < 2:
            continue
        r_name = fighter_links[0].get_text(strip=True)
        b_name = fighter_links[1].get_text(strip=True)
        r_url = fighter_links[0].get("href", "")
        b_url = fighter_links[1].get("href", "")
        cols = row.select("td")
        weight_class = cols[6].get_text(strip=True) if len(cols) > 6 else ""
        is_title = "title" in weight_class.lower() or "championship" in weight_class.lower()
        if r_name and b_name:
            fights.append({
                "r_fighter": r_name,
                "b_fighter": b_name,
                "weight_class": weight_class,
                "is_title": is_title,
                "r_url": r_url,
                "b_url": b_url,
            })
    return fights


def get_completed_event_results(event_url):
    """Return fight results for a completed event."""
    html = _get(event_url)
    soup = BeautifulSoup(html, "lxml")
    fights = []
    rows = soup.select("tr.b-fight-details__table-row")
    for row in rows:
        fighter_links = row.select("a.b-link")
        if len(fighter_links) < 2:
            continue
        r_name = fighter_links[0].get_text(strip=True)
        b_name = fighter_links[1].get_text(strip=True)
        cols = row.select("td")
        winner_col = cols[0].get_text(strip=True) if cols else ""
        method = cols[7].get_text(strip=True) if len(cols) > 7 else ""
        round_num = cols[8].get_text(strip=True) if len(cols) > 8 else ""
        weight_class = cols[6].get_text(strip=True) if len(cols) > 6 else ""

        # Determine winner from win indicator
        win_spans = row.select("i.b-flag__text")
        winner = ""
        if win_spans:
            txt = win_spans[0].get_text(strip=True).lower()
            if "win" in txt:
                winner = r_name

        if r_name and b_name:
            fights.append({
                "r_fighter": r_name,
                "b_fighter": b_name,
                "winner": winner,
                "method": method,
                "round": round_num,
                "weight_class": weight_class,
            })
    return fights


def _parse_height_cm(ht_str):
    """Convert '6' 2\"' to cm float."""
    m = re.match(r"(\d+)'\s*(\d+)", ht_str)
    if m:
        return int(m.group(1)) * 30.48 + int(m.group(2)) * 2.54
    return None


def _parse_reach_cm(reach_str):
    """Convert '75\"' to cm float."""
    m = re.match(r"([\d.]+)", reach_str)
    if m:
        return float(m.group(1)) * 2.54
    return None


def _parse_weight_kg(wt_str):
    """Convert '185 lbs.' to kg float."""
    m = re.match(r"([\d.]+)", wt_str)
    if m:
        return float(m.group(1)) * 0.453592
    return None


def _parse_age(dob_str):
    """Convert 'May 01, 1994' to age int."""
    try:
        dob = datetime.strptime(dob_str.strip(), "%b %d, %Y")
        today = datetime.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception:
        return None


def scrape_fighter_stats(fighter_url):
    """Scrape live stats from a fighter's ufcstats page."""
    if not fighter_url:
        return {}
    try:
        html = _get(fighter_url)
        soup = BeautifulSoup(html, "lxml")

        stats = {}
        for item in soup.select("li.b-list__box-list-item"):
            text = item.get_text(separator=":", strip=True)
            parts = [p.strip() for p in text.split(":") if p.strip()]
            if len(parts) >= 2:
                key, val = parts[0].lower(), parts[1]
                if "height" in key:
                    stats["height"] = _parse_height_cm(val)
                elif "weight" in key:
                    stats["weight"] = _parse_weight_kg(val)
                elif "reach" in key:
                    stats["reach"] = _parse_reach_cm(val)
                elif "stance" in key:
                    stats["stance"] = val
                elif "dob" in key:
                    stats["age"] = _parse_age(val)
                elif "slpm" in key:
                    stats["SLpM"] = float(val) if val.replace(".", "").isdigit() else None
                elif "str. acc" in key:
                    stats["sig_str_acc"] = float(val.strip("%")) / 100 if "%" in val else None
                elif "sapm" in key:
                    stats["SApM"] = float(val) if val.replace(".", "").isdigit() else None
                elif "str. def" in key:
                    stats["str_def"] = float(val.strip("%")) / 100 if "%" in val else None
                elif "td avg" in key:
                    stats["td_avg"] = float(val) if val.replace(".", "").isdigit() else None
                elif "td acc" in key:
                    stats["td_acc"] = float(val.strip("%")) / 100 if "%" in val else None
                elif "td def" in key:
                    stats["td_def"] = float(val.strip("%")) / 100 if "%" in val else None
                elif "sub. avg" in key:
                    stats["sub_avg"] = float(val) if val.replace(".", "").isdigit() else None

        # W-L record
        record_el = soup.select_one("span.b-content__title-record")
        if record_el:
            record_text = record_el.get_text(strip=True)
            m = re.search(r"(\d+)-(\d+)", record_text)
            if m:
                stats["wins"] = int(m.group(1))
                stats["losses"] = int(m.group(2))

        return stats
    except Exception:
        return {}


def find_fighter_url(fighter_name):
    """Search ufcstats for a fighter by name, return their URL."""
    last_name = fighter_name.strip().split()[-1]
    char = last_name[0].lower()
    try:
        html = _get(f"http://ufcstats.com/statistics/fighters?char={char}&page=all")
        soup = BeautifulSoup(html, "lxml")
        rows = soup.select("tr.b-statistics__table-row")
        candidates = []
        for row in rows:
            cols = row.select("td")
            if len(cols) < 2:
                continue
            first = cols[0].get_text(strip=True)
            last = cols[1].get_text(strip=True)
            full = f"{first} {last}".strip()
            link = row.select_one("a.b-link")
            if link:
                candidates.append((full, link.get("href", "")))

        names = [c[0] for c in candidates]
        matches = difflib.get_close_matches(fighter_name, names, n=1, cutoff=0.6)
        if matches:
            for full, url in candidates:
                if full == matches[0]:
                    return url
    except Exception:
        pass
    return ""
