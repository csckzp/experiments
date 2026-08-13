import os
import re
from bs4 import BeautifulSoup
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAJORS_FILE = os.path.join(BASE_DIR, "majors.html")
CLUBS_FILE = os.path.join(BASE_DIR, "clubs.html")

def load_majors(file_path: str = MAJORS_FILE) -> List[Dict[str, str]]:
    """Extract list of undergraduate majors from majors.html"""
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    
    programs = []
    seen = set()
    for a in soup.find_all("a"):
        txt = " ".join(a.get_text().split())
        if any(prefix in txt for prefix in ["BS in", "BA in", "BBA in", "BS/", "BA/", "Bachelor of", "Dual-degree"]):
            clean_name = re.sub(r"\s+", " ", txt).strip()
            if clean_name and clean_name not in seen and len(clean_name) < 100:
                seen.add(clean_name)
                cat = "STEM & Tech" if any(k in clean_name for k in ["Engineering", "Physics", "Computer", "Bio", "Chem", "Cyber"]) else "Arts & Business"
                programs.append({"name": clean_name, "category": cat})
    return programs

def load_clubs(file_path: str = CLUBS_FILE) -> List[Dict[str, str]]:
    """Extract list of student clubs from clubs.html"""
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        
    clubs = []
    seen = set()
    for a in soup.find_all("a", href=lambda h: h and "/organization/" in h):
        full_text = " ".join(a.get_text().split())
        cleaned = re.sub(r"^[A-Z]\s+(?=[A-Z])", "", full_text)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            name = cleaned[:45].rsplit(" ", 1)[0] if len(cleaned) > 45 else cleaned
            desc = cleaned[len(name):].strip() or "Active Hofstra student organization."
            clubs.append({"name": name.strip(), "description": desc.strip()})
    return clubs

MAJORS_DATA = load_majors()
CLUBS_DATA = load_clubs()

def get_majors_summary(limit: int = 40) -> str:
    return "\n".join([f"- {m['name']} ({m['category']})" for m in MAJORS_DATA[:limit]])

def get_clubs_summary(limit: int = 40) -> str:
    return "\n".join([f"- {c['name']}: {c['description'][:90]}..." for c in CLUBS_DATA[:limit]])
