
from curl_cffi import requests
import json
import pandas as pd
import numpy as np
import re
import unicodedata
import difflib
import os
import time
from scipy.optimize import minimize
from scipy.stats import poisson

# ============================================================
# CONFIG
# ============================================================
BASE_URL = "https://www.sportybet.com"
REGION = "ke"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Current-Country": REGION.upper(),
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
DECAY_LAMBDA = 0.0018
CACHE_FILE = "football_cache.json"
CACHE_MAX_AGE_DAYS = 7
GAMES_TO_REMOVE = 10

LEAGUES = {
    "EPL": "en.1", "Championship": "en.2", "La Liga": "es.1", "Segunda": "es.2",
    "Bundesliga": "de.1", "2. Bundesliga": "de.2", "Serie A": "it.1", "Serie B": "it.2",
    "Ligue 1": "fr.1", "Ligue 2": "fr.2", "Eredivisie": "nl.1", "Primeira Liga": "pt.1",
    "Belgian Pro League": "be.1", "Süper Lig": "tr.1", "Russian Premier League": "ru.1",
    "Austrian Bundesliga": "at.1", "Swiss Super League": "ch.1", "Super League Greece": "gr.1",
    "Scottish Premiership": "scotland", "Ukrainian Premier League": "ua.1", "Liga I": "ro.1",
    "Ekstraklasa": "pl.1", "Danish Superliga": "dk.1", "Allsvenskan": "se.1", "Eliteserien": "no.1",
    "Czech First League": "cz.1", "HNL": "hr.1", "SuperLiga": "rs.1", "Serie A (Brazil)": "br.1",
    "Primera División (Argentina)": "ar.1", "Liga MX": "mx.1", "MLS": "us.1", "J1 League": "jp.1",
    "Chinese Super League": "cn.1", "A-League": "au.1",
}
SEASONS_TO_TRY = ["2024-25", "2023-24", "2022-23"]

# ============================================================
# TEAM RESOLVER
# ============================================================
NOISE_WORDS = {"fc","afc","cf","sc","ac","as","ss","ssc","sv","vfl","vfb","bsc","tsg",
    "fsv","bk","sk","fk","kf","nk","hnk","cd","sd","ud","rc","rcd","cs","us","club",
    "de","futbol","football","sport","sportif","calcio","soccer","the",
    "1899","1900","1901","1904","1907","1909","1911","1913","04","05","07","09","96","98"}

TEAM_ALIASES = {
    "man city":"manchester city","man utd":"manchester united","man united":"manchester united",
    "newcastle":"newcastle united","west ham":"west ham united","brighton":"brighton & hove albion",
    "tottenham":"tottenham hotspur","spurs":"tottenham hotspur","wolves":"wolverhampton wanderers",
    "nottm forest":"nottingham forest","nott'm forest":"nottingham forest","nottingham":"nottingham forest",
    "sheffield utd":"sheffield united","sheffield weds":"sheffield wednesday","leicester":"leicester city",
    "leeds":"leeds united","norwich":"norwich city","stoke":"stoke city","swansea":"swansea city",
    "hull":"hull city","cardiff":"cardiff city","birmingham":"birmingham city","barca":"barcelona",
    "atletico":"atletico madrid","real":"real madrid","betis":"real betis","sociedad":"real sociedad",
    "athletic club":"athletic bilbao","celta":"celta vigo","rayo":"rayo vallecano","bayern":"bayern munich",
    "dortmund":"borussia dortmund","bvb":"borussia dortmund","leverkusen":"bayer leverkusen",
    "leipzig":"rb leipzig","frankfurt":"eintracht frankfurt","wolfsburg":"vfl wolfsburg",
    "gladbach":"borussia monchengladbach","freiburg":"sc freiburg","hoffenheim":"tsg hoffenheim",
    "mainz":"mainz 05","augsburg":"fc augsburg","stuttgart":"vfb stuttgart","werder":"werder bremen",
    "union berlin":"union berlin","bochum":"vfl bochum","heidenheim":"1. fc heidenheim",
    "darmstadt":"sv darmstadt 98","koln":"1. fc koln","inter":"inter milan","milan":"ac milan",
    "juve":"juventus","roma":"as roma","lazio":"ss lazio","verona":"hellas verona",
    "psg":"paris saint-germain","marseille":"olympique marseille","lyon":"olympique lyonnais",
    "monaco":"as monaco","lens":"rc lens","nice":"ogc nice","rennes":"stade rennais",
    "reims":"stade de reims","strasbourg":"rc strasbourg","brest":"stade brestois",
    "metz":"fc metz","lorient":"fc lorient","clermont":"clermont foot","az":"az alkmaar",
    "twente":"fc twente","utrecht":"fc utrecht","sporting":"sporting cp","braga":"sc braga",
    "fenerbahce":"fenerbahce","besiktas":"besiktas","sao paulo":"sao paulo","gremio":"gremio",
    "vasco":"vasco da gama","boca":"boca juniors","river":"river plate","lafc":"los angeles fc",
    "america":"club america","chivas":"guadalajara","tigres":"tigres uanl",
}

def normalize(name):
    if not name: return ""
    s = name.lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^\w\s]"," ",s)
    s = re.sub(r"\s+"," ",s).strip()
    return " ".join(t for t in s.split() if t not in NOISE_WORDS).strip()

class TeamResolver:
    def __init__(self, known_teams):
        self.known = list(known_teams)
        self.known_norm = {}
        for t in self.known: self.known_norm.setdefault(normalize(t), t)
        self.alias_norm = {normalize(k): normalize(v) for k, v in TEAM_ALIASES.items()}
        
    def resolve(self, name, cutoff=0.70):
        if not name: return None
        if name.strip() in self.known: return name.strip()
        norm = normalize(name)
        if not norm: return None
        if norm in self.known_norm: return self.known_norm[norm]
        alias = self.alias_norm.get(norm)
        if alias:
            if alias in self.known_norm: return self.known_norm[alias]
            at = set(alias.split())
            for kn, kr in self.known_norm.items():
                kt = set(kn.split())
                if at and (at <= kt or kt <= at): return kr
        nt = set(norm.split()); best, best_s = None, 0.0
        for kn, kr in self.known_norm.items():
            kt = set(kn.split())
            if kt and (nt <= kt or kt <= nt):
                s = len(nt & kt)/max(len(nt | kt),1)
                if s > best_s: best_s, best = s, kr
        if best and best_s >= 0.4: return best 
        for kn, kr in self.known_norm.items():
            if len(kn) > 3 and (kn in norm or norm in kn): return kr
        m = difflib.get_close_matches(norm, list(self.known_norm.keys()), n=1, cutoff=cutoff)
        return self.known_norm[m[0]] if m else None

# ============================================================
# DIXON-COLES MODEL
# ============================================================
class DixonColes:
    def __init__(self):
        self.teams, self.idx = [], {}
        self.attack, self.defense = {}, {}
        self.home_adv, self.rho = 0.0, 0.0
        self.resolver = None

    def _tau(self, x, y, lam, mu, rho):
        if x==0 and y==0: return 1 - lam*mu*rho
        if x==0 and y==1: return 1 + lam*rho
        if x==1 and y==0: return 1 + mu*rho
        if x==1 and y==1: return 1 - rho
        return 1.0

    def _nll(self, params, h_idx, a_idx, hg, ag, w):
        n = len(self.teams)
        att = params[:n] - params[:n].mean()
        dfn = params[n:2*n]
        ha = params[2*n]; rho = params[2*n+1]
        lam = np.exp(att[h_idx] + dfn[a_idx] + ha)
        mu  = np.exp(att[a_idx] + dfn[h_idx])
        tau = np.ones_like(lam)
        tau = np.where((hg == 0) & (ag == 0), 1 - lam * mu * rho, tau)
        tau = np.where((hg == 0) & (ag == 1), 1 + lam * rho, tau)
        tau = np.where((hg == 1) & (ag == 0), 1 + mu * rho, tau)
        tau = np.where((hg == 1) & (ag == 1), 1 - rho, tau)
        log_lik = np.log(np.maximum(tau, 1e-9)) + poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
        return -np.sum(w * log_lik) + 0.01*(np.sum(att**2)+np.sum(dfn**2))

    def fit(self, df):
        self.teams = sorted(set(df["home"])|set(df["away"]))
        self.idx = {t:i for i,t in enumerate(self.teams)}
        n = len(self.teams)
        latest = df["date"].max()
        w = np.exp(-DECAY_LAMBDA*(latest-df["date"]).dt.days.values)
        h_idx = df["home"].map(self.idx).values
        a_idx = df["away"].map(self.idx).values
        hg = df["home_goals"].values
        ag = df["away_goals"].values
        x0 = np.concatenate([np.zeros(n),np.zeros(n),[0.25],[-0.05]])
        res = minimize(self._nll, x0, args=(h_idx, a_idx, hg, ag, w), 
                       method="L-BFGS-B", options={"maxiter":300,"disp":False})
        p = res.x; att = p[:n]-p[:n].mean(); dfn = p[n:2*n]
        self.attack = {t:att[self.idx[t]] for t in self.teams}
        self.defense = {t:dfn[self.idx[t]] for t in self.teams}
        self.home_adv, self.rho = p[2*n], p[2*n+1]
        self.resolver = TeamResolver(self.teams)
        return self

    def predict(self, home, away, max_goals=8):
        hr = self.resolver.resolve(home); ar = self.resolver.resolve(away)
        if hr is None or ar is None: return None
        lam = np.exp(self.attack[hr]+self.defense[ar]+self.home_adv)
        mu = np.exp(self.attack[ar]+self.defense[hr])
        m = np.zeros((max_goals+1,max_goals+1))
        for i in range(max_goals+1):
            for j in range(max_goals+1):
                m[i,j] = self._tau(i,j,lam,mu,self.rho)*poisson.pmf(i,lam)*poisson.pmf(j,mu)
        m /= m.sum(); tot = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
        return {"home":np.tril(m,-1).sum(),"draw":np.trace(m),"away":np.triu(m,1).sum(),
                "over05":m[tot>=1].sum(),"over15":m[tot>=2].sum(),"under15":m[tot<=1].sum(),
                "over25":m[tot>=3].sum(),"under25":m[tot<=2].sum(),
                "over35":m[tot>=4].sum(),"under35":m[tot<=3].sum(),
                "btts_yes":m[1:,1:].sum(),"btts_no":1-m[1:,1:].sum()}

# ============================================================
# DATA FETCHING
# ============================================================
def fetch_data():
    if os.path.exists(CACHE_FILE):
        age = (time.time()-os.path.getmtime(CACHE_FILE))/86400
        if age < CACHE_MAX_AGE_DAYS:
            with open(CACHE_FILE) as f: df = pd.DataFrame(json.load(f))
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df.dropna(subset=["date"])
    print("Downloading data...")
    rows = []
    for s in SEASONS_TO_TRY:
        for n, c in LEAGUES.items():
            url = f"https://raw.githubusercontent.com/openfootball/football.json/master/{s}/{c}.json"
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200:
                    for m in r.json().get("matches", []):
                        if m.get("score", {}).get("ft"):
                            rows.append({"date":m.get("date"),"home":m.get("team1"),"away":m.get("team2"),
                                         "home_goals":m["score"]["ft"][0],"away_goals":m["score"]["ft"][1]})
            except: pass
    df = pd.DataFrame(rows).dropna()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cdf = df.copy(); cdf["date"] = cdf["date"].dt.strftime("%Y-%m-%d")
    with open(CACHE_FILE,"w") as f: json.dump(cdf.to_dict(orient="records"), f)
    return df

# ============================================================
# SPORTYBET CLIENT (BINARY SEARCH SMART FILTER)
# ============================================================
class SportyBetClient:
    def __init__(self):
        self.session = requests.Session(); self.session.headers.update(HEADERS)
        
    def get_booking_data(self, code):
        code = code.strip()
        if "share/" in code:
            code = code.split("share/")[-1].split("?")[0]
            
        print(f"   [DEBUG] Requesting Booking Code: {code}")
        
        session = requests.Session()
        session.verify = False
        
        try:
            session.get("https://www.sportybet.com/ke/", impersonate="chrome120", timeout=10)
        except Exception:
            pass
        
        url = f"{BASE_URL}/api/{REGION}/orders/share/{code}"
        
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Current-Country": "KE",
            "Referer": f"https://www.sportybet.com/ke/share/{code}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        try:
            r = session.get(url, headers=headers, impersonate="chrome120", timeout=15)
            
            if r.status_code != 200:
                print(f"   [DEBUG] HTTP Error: {r.status_code}")
                return None
                
            data = r.json()
            
            if data.get("bizCode") != 10000:
                print(f"   [DEBUG] API Error: {data.get('message', 'Unknown Error')}")
                return None
            
            slip = data.get("data", {})
            games = slip.get("selections") or slip.get("outcomes") or slip.get("events") or []
            slip["selections"] = games 
            return slip
            
        except Exception as e:
            print(f"   [DEBUG] Connection Error: {e}")
            return None
    def _try_post(self, selections):
        """Helper function to post a payload. Returns (shareCode, True) or (None, False)"""
        payload = {"selections": []}
        for s in selections:
            raw = s["original_selection"]
            event_id = str(raw.get("eventId", "")).strip()
            market_id = raw.get("marketId")
            outcome_id = raw.get("outcomeId")
            specifier = raw.get("specifier", "")

            if "markets" in raw and isinstance(raw["markets"], list) and len(raw["markets"]) > 0:
                m = raw["markets"][0]
                if not market_id: market_id = m.get("id")
                if "outcomes" in m and isinstance(m["outcomes"], list) and len(m["outcomes"]) > 0:
                    o = m["outcomes"][0]
                    if not outcome_id: outcome_id = o.get("id")
                    if not specifier: specifier = o.get("specifier", "")

            if not event_id or not market_id or not outcome_id: continue
            
            sel_item = {
                "eventId": event_id,
                "marketId": str(market_id),
                "outcomeId": str(outcome_id)
            }
            if specifier and str(specifier).lower() not in ["none", "null", ""]:
                sel_item["specifier"] = str(specifier)

            payload["selections"].append(sel_item)
            
        if not payload["selections"]: return None, False

        try:
            r = self.session.post(f"{BASE_URL}/api/{REGION}/orders/share", json=payload, timeout=15)
            r.raise_for_status(); data = r.json()
            if data.get("bizCode") == 10000: 
                return data.get("data",{}).get("shareCode"), True
            return None, False
        except Exception:
            return None, False

    def create_booking_code(self, selections, label=""):
        """Uses binary search to find and drop ONLY the suspended games"""
        # Pre-filter obviously dead games
        now_ms = int(time.time() * 1000)
        valid_selections = []
        for s in selections:
            start_time = s["original_selection"].get("estimateStartTime")
            if start_time:
                try:
                    if int(start_time) < (now_ms + 5 * 60 * 1000): continue
                except: pass
            valid_selections.append(s)

        if not valid_selections: return None

        # Try posting all at once first
        code, success = self._try_post(valid_selections)
        if success: return code

        print(f"   🔍 API rejected the batch. Searching for the suspended game...")
        
        # Binary search recursive function
        def find_valid_games(games_list):
            if not games_list: return []
            
            # If it's just 1 game and fails, it's the bad one
            if len(games_list) == 1:
                code, success = self._try_post(games_list)
                return games_list if success else []
                
            # Try sending the whole chunk
            code, success = self._try_post(games_list)
            if success: return games_list
            
            # Split in half and test recursively
            mid = len(games_list) // 2
            left_valid = find_valid_games(games_list[:mid])
            right_valid = find_valid_games(games_list[mid:])
            return left_valid + right_valid

        # Find all good games
        good_games = find_valid_games(valid_selections)
        
        if not good_games: return None
        
        # Final attempt with ONLY the good games
        final_code, success = self._try_post(good_games)
        if success:
            print(f"   ✅ Found {len(good_games)} valid games. Generated Code.")
            return final_code
        return None

# ============================================================
# ANALYZE & SPLIT
# ============================================================
def analyze_slip(selections, model):
    evaluated = []
    
    for sel in selections:
        home = sel.get("homeTeamName") or sel.get("homeTeam") or sel.get("team1") or "Unknown"
        away = sel.get("awayTeamName") or sel.get("awayTeam") or sel.get("team2") or "Unknown"
        
        market = ""
        outcome = ""
        if "markets" in sel and len(sel["markets"]) > 0:
            market = sel["markets"][0].get("desc", "")
            if "outcomes" in sel["markets"][0] and len(sel["markets"][0]["outcomes"]) > 0:
                outcome = sel["markets"][0]["outcomes"][0].get("desc", "")
        
        odds = sel.get("odds", 1.0)
        if "markets" in sel and len(sel["markets"]) > 0 and "outcomes" in sel["markets"][0]:
            odds = sel["markets"][0]["outcomes"][0].get("odds", odds)
        odds = float(odds)
        
        implied_prob = 1.0 / odds if odds > 1.0 else 0.5
        mp = None

        if home != "Unknown" and away != "Unknown":
            p = model.predict(home, away)
            if p:
                if "1X2" in market:
                    if "Home" in outcome or outcome == "1": mp = p["home"]
                    elif "Draw" in outcome or outcome == "X": mp = p["draw"]
                    elif "Away" in outcome or outcome == "2": mp = p["away"]
                elif "Over/Under" in market:
                    if "Over 0.5" in outcome: mp = p["over05"]
                    elif "Over 1.5" in outcome: mp = p["over15"]
                    elif "Under 1.5" in outcome: mp = p["under15"]
                    elif "Over 2.5" in outcome: mp = p["over25"]
                    elif "Under 2.5" in outcome: mp = p["under25"]
                    elif "Over 3.5" in outcome: mp = p["over35"]
                    elif "Under 3.5" in outcome: mp = p["under35"]
                elif "GG/NG" in market or "BTTS" in market:
                    if "Yes" in outcome or "GG" in outcome: mp = p["btts_yes"]
                    elif "No" in outcome or "NG" in outcome: mp = p["btts_no"]

        prob = mp if mp is not None else implied_prob
        source = "AI" if mp is not None else "Bookie"

        evaluated.append({
            "original_selection": sel,
            "home": home, "away": away, "market": market, "outcome": outcome,
            "prob": prob, "source": source
        })

    evaluated.sort(key=lambda x: x["prob"])
    
    remove_count = min(GAMES_TO_REMOVE, max(0, len(evaluated) - 1))
    to_remove = evaluated[:remove_count]
    to_keep = evaluated[remove_count:]
    
    return to_keep, to_remove

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" SPORTYBET — DUAL CODE GENERATOR (BINARY SEARCH)")
    print("=" * 70)
    
    code = input("\nPaste your SportyBet booking code: ").strip()
    if not code: exit()

    print("\n[1/3] Loading model...")
    hist = fetch_data()
    model = DixonColes().fit(hist)

    print("[2/3] Reading slip...")
    client = SportyBetClient()
    slip = client.get_booking_data(code)
    if not slip or not slip.get("selections"):
        print("Could not read slip. Is it expired?")
        exit()
        
    print(f"[3/3] Analyzing {len(slip['selections'])} games...\n")
    
    to_keep, to_remove = analyze_slip(slip["selections"], model)

    print("=" * 70)
    print(f" 🛑 REMOVED {len(to_remove)} WORST GAMES")
    print("=" * 70)
    for i, p in enumerate(to_remove, 1):
        print(f"{i:2}. {p['home'][:15]} vs {p['away'][:15]:<15} | {p['outcome']} ({p['prob']:.1%})")

    print("\n" + "=" * 70)
    print(f" ✅ KEPT {len(to_keep)} BEST GAMES")
    print("=" * 70)

    print("\nGenerating SportyBet Codes (Auto-finding suspended games)...")
    
    safe_code = client.create_booking_code(to_keep, label="SAFE SLIP")
    if safe_code:
        print(f" ✅ SAFE CODE: {safe_code}")
    else:
        print(" ❌ Safe code failed entirely.")

    risky_code = client.create_booking_code(to_remove, label="RISKY SLIP")
    if risky_code:
        print(f" 🎲 RISKY CODE: {risky_code}")
    else:
        print(" ❌ Risky code failed entirely.")
        
    print("=" * 70)

