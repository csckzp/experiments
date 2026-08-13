import os
import re
from bs4 import BeautifulSoup
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAJORS_FILE = os.path.join(BASE_DIR, "majors.html")
CLUBS_FILE = os.path.join(BASE_DIR, "clubs.html")

DEFAULT_MAJORS = [
    {"name": "BS in Computer Science", "category": "STEM & Technology"},
    {"name": "BS in Bioengineering", "category": "STEM & Technology"},
    {"name": "BS in Cybersecurity", "category": "STEM & Technology"},
    {"name": "BS in Applied Physics", "category": "STEM & Technology"},
    {"name": "BBA in Entrepreneurship & Innovation", "category": "Business & Economics"},
    {"name": "BBA in Accounting & Business Analytics", "category": "Business & Economics"},
    {"name": "BBA in Marketing & Digital Media", "category": "Business & Economics"},
    {"name": "BA in Music & Performing Arts", "category": "Arts & Creative Media"},
    {"name": "BA in Film Studies & Production", "category": "Arts & Creative Media"},
    {"name": "BA in Fine Arts & Graphic Design", "category": "Arts & Creative Media"},
    {"name": "BS in Biochemistry & Pre-Med", "category": "Health & Life Sciences"},
    {"name": "BS in Community Health", "category": "Health & Life Sciences"},
    {"name": "BA in Psychology", "category": "Health & Life Sciences"},
    {"name": "BA in Political Science & Global Affairs", "category": "Humanities & Social Sciences"},
    {"name": "BA in Journalism & Public Relations", "category": "Humanities & Social Sciences"},
]

DEFAULT_CLUBS = [
    {"name": "180 Degrees Consulting", "description": "Branch of the world's largest student consultancy, working with socially conscious organizations."},
    {"name": "A-Capella (Keynotes)", "description": "Student vocal performance club bringing music and performance to campus."},
    {"name": "Accounting Society", "description": "Professional network for accounting and business students with weekly industry speakers."},
    {"name": "ActiveMinds at Hofstra", "description": "Student-led mental health awareness and campus wellness organization."},
    {"name": "African Student Association", "description": "Promotes African culture, fashion, dance, and social awareness across campus."},
    {"name": "AI in Medicine & Tech Interest Group", "description": "Explores the intersection of artificial intelligence, healthcare, and robotics."},
    {"name": "Alpha Kappa Psi Business Fraternity", "description": "Co-ed professional business fraternity promoting professional development and leadership."},
    {"name": "Robotics & Automation Society", "description": "Hands-on engineering team building autonomous systems and competing nationally."},
    {"name": "Environmental Action Coalition", "description": "Focuses on campus sustainability, eco-friendly initiatives, and climate action."},
    {"name": "Hofstra Culinary & Foodies Club", "description": "Celebrates diverse cuisines, hosting cooking workshops and campus dining events."}
]

def load_majors(file_path: str = MAJORS_FILE) -> List[Dict[str, str]]:
    """Extract clean list of undergraduate majors and programs from majors.html"""
    if not os.path.exists(file_path):
        return DEFAULT_MAJORS
    
    try:
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
                    
                    if any(k in clean_name for k in ["Engineering", "Physics", "Computer", "Bio", "Chemistry", "Mathematics", "Cybersecurity"]):
                        cat = "STEM & Technology"
                    elif any(k in clean_name for k in ["Accounting", "Business", "Finance", "Marketing", "Entrepreneurship", "Management"]):
                        cat = "Business & Economics"
                    elif any(k in clean_name for k in ["Art", "Music", "Drama", "Film", "Design", "Dance"]):
                        cat = "Arts & Creative Media"
                    elif any(k in clean_name for k in ["Psychology", "Health", "Nursing", "Pre-Athletic", "Speech"]):
                        cat = "Health & Life Sciences"
                    else:
                        cat = "Humanities & Social Sciences"
                    
                    programs.append({"name": clean_name, "category": cat})
                    
        return programs if len(programs) >= 5 else DEFAULT_MAJORS
    except Exception:
        return DEFAULT_MAJORS

def load_clubs(file_path: str = CLUBS_FILE) -> List[Dict[str, str]]:
    """Extract clean list of student organizations and clubs from clubs.html"""
    if not os.path.exists(file_path):
        return DEFAULT_CLUBS
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            
        clubs = []
        seen = set()
        org_links = soup.find_all("a", href=lambda h: h and "/organization/" in h)
        
        for a in org_links:
            full_text = " ".join(a.get_text().split())
            if not full_text or len(full_text) < 3:
                continue
                
            cleaned = re.sub(r"^[A-Z]\s+(?=[A-Z])", "", full_text)
            if cleaned in seen:
                continue
            seen.add(cleaned)
            
            if len(cleaned) > 45:
                name = cleaned[:45].rsplit(" ", 1)[0]
                desc = cleaned[len(name):].strip()
            else:
                name = cleaned
                desc = "Active student organization at Hofstra University."
                
            clubs.append({
                "name": name.strip(),
                "description": desc.strip() if desc else "Student organization"
            })
            
        return clubs if len(clubs) >= 5 else DEFAULT_CLUBS
    except Exception:
        return DEFAULT_CLUBS

MAJORS_DATA = load_majors()
CLUBS_DATA = load_clubs()

def get_majors_summary(limit: int = 50) -> str:
    sample = MAJORS_DATA[:limit] if len(MAJORS_DATA) > limit else MAJORS_DATA
    return "\n".join([f"- {m['name']} [{m['category']}]" for m in sample])

def get_clubs_summary(limit: int = 50) -> str:
    sample = CLUBS_DATA[:limit] if len(CLUBS_DATA) > limit else CLUBS_DATA
    return "\n".join([f"- {c['name']}: {c['description'][:100]}..." for c in sample])
