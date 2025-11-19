"""Regex-driven extraction utilities for structured attributes."""

from __future__ import annotations

import re
from typing import Dict, Literal, Optional, Tuple

from normalize import normalize_name

# Spanish universities provided by the user (used for aliasing and hard labels)
SPANISH_UNIVERSITIES = [
    "Universidad de Alcalá",
    "Universidad de Almería",
    "Universidad de Cádiz",
    "Universidad de Córdoba",
    "Universidad de Granada",
    "Universidad de Huelva",
    "Universidad de Jaén",
    "Universidad de Málaga",
    "Universidad de Sevilla",
    "Universidad Pablo de Olavide",
    "Universidad de Zaragoza",
    "Universidad de La Rioja",
    "Universidad de Navarra (UPNA – Pública)",
    "Universidad del País Vasco / Euskal Herriko Unibertsitatea (UPV/EHU)",
    "Universidad de Cantabria",
    "Universidad de Oviedo",
    "Universidade da Coruña",
    "Universidade de Santiago de Compostela",
    "Universidade de Vigo",
    "Universidad de Murcia",
    "Universidad Politécnica de Cartagena (UPCT)",
    "Universidad de Castilla-La Mancha",
    "Universidad de Extremadura",
    "Universidad de Salamanca",
    "Universidad de Valladolid",
    "Universidad de León",
    "Universidad de Burgos",
    "Universidad de Barcelona (UB)",
    "Universitat Autònoma de Barcelona (UAB)",
    "Universitat Politècnica de Catalunya (UPC)",
    "Universitat Pompeu Fabra (UPF)",
    "Universitat de Girona (UdG)",
    "Universitat de Lleida (UdL)",
    "Universitat Rovira i Virgili (URV)",
    "Universitat de les Illes Balears (UIB)",
    "Universitat de València (UV)",
    "Universitat Politècnica de València (UPV)",
    "Universidad de Alicante (UA)",
    "Universitat Jaume I (UJI)",
    "Universidad Miguel Hernández de Elche (UMH)",
    "Universidad de La Laguna (ULL)",
    "Universidad de Las Palmas de Gran Canaria (ULPGC)",
    "Universidad Autónoma de Madrid (UAM)",
    "Universidad Complutense de Madrid (UCM)",
    "Universidad Politécnica de Madrid (UPM)",
    "Universidad Rey Juan Carlos (URJC)",
    "Universidad Carlos III de Madrid (UC3M)",
    "Universidad Nacional de Educación a Distancia (UNED)",
    "Universidad Internacional de Andalucía (UNIA)",
    "Universidad Internacional Menéndez Pelayo (UIMP)",
    "Universitat Oberta de Catalunya (UOC)",
    "Universidad de Deusto",
    "Mondragon Unibertsitatea (MU)",
    "Universidad de Navarra (UNAV)",
    "Universidad Pontificia Comillas (Comillas ICAI-ICADE)",
    "Universidad Pontificia de Salamanca (UPSA)",
    "Universidad Ramon Llull (URL)",
    "Universitat Abat Oliba CEU (UAO CEU)",
    "Universidad CEU San Pablo",
    "Universidad CEU Cardenal Herrera (UCH CEU)",
    "Universitat Internacional de Catalunya (UIC Barcelona)",
    "Universidad Católica de Valencia San Vicente Mártir (UCV)",
    "Universidad Católica de Murcia (UCAM)",
    "Universidad Católica Santa Teresa de Jesús de Ávila (UCAV)",
    "Universidad Francisco de Vitoria (UFV)",
    "Universidad Antonio de Nebrija (Nebrija)",
    "Universidad Alfonso X el Sabio (UAX)",
    "Universidad Camilo José Cela (UCJC)",
    "Universidad Europea de Madrid (UEM)",
    "Universidad Europea de Valencia",
    "Universidad Europea de Canarias",
    "Universidad Europea del Atlántico (UNEATLANTICO)",
    "Universidad Europea Miguel de Cervantes (UEMC)",
    "Universidad San Jorge (USJ)",
    "Universidad Villanueva",
    "IE University",
    "Universidad a Distancia de Madrid (UDIMA)",
    "Universidad Internacional de La Rioja (UNIR)",
    "Universidad Internacional de Valencia (VIU)",
    "Universidad Isabel I",
    "TECH Universidad Tecnológica",
    "Universidad Atlántico Medio",
    "Universidad Fernando Pessoa–Canarias",
    "Universidad EUNEIZ (Vitoria-Gasteiz)",
    "Universidad Loyola Andalucía",
    "Universidad de Vic – Universidad Central de Cataluña (UVic-UCC)",
    "Universidad San Dámaso",
    "Universidad Internacional de Empresa (UNIE)",
]

# Hard exceptions always win
ORG_HARD_POSITIVE = {
    "ie business school": "university",
    "ie university": "university",
    "universidad de navarra": "university",
    "universidad complutense de madrid": "university",
}
ORG_HARD_NEGATIVE = {
    "bank of spain": "company",
    "banco de españa": "company",
}

for uni in SPANISH_UNIVERSITIES:
    ORG_HARD_POSITIVE.setdefault(uni.lower(), "university")

# Section names we use as priors
SECTION_PRIOR = {
    "academic_background": "university",
    "academic_experience": "university",
    "corporate_experience": "company",
}

# Lexical cues (soft signals)
UNIV_CUES = [
    "university",
    "universidad",
    "université",
    "università",
    "universidade",
    "college",
    "school of",
    "faculty",
    "facultad",
    "escuela",
    "polytechnic",
    "politécnica",
    "institute of technology",
    "institut of technology",
]
COMP_CUES = [
    "inc",
    "llc",
    "ltd",
    "ltda",
    "s.a.",
    "s.l.",
    "gmbh",
    "ag",
    "bv",
    "spa",
    "pty",
    "partners",
    "group",
    "consulting",
    "capital",
    "bank",
    "banco",
    "studio",
    "lab",
    "labs",
    "holding",
    "fund",
]
GOV_CUES = [
    "ministry",
    "council",
    "agency",
    "commission",
    "secretariat",
    "court",
    "bank of",
    "central bank",
    "city of",
    "state of",
    "department of",
]

SECTION_STOP_ORGS = {
    "academic experience",
    "academic exp",
    "corporate experience",
    "academic background",
    "courses taught",
    "education",
    "publications",
    "professional experience",
}

HONORS_THESIS_TOKENS = (
    "summa cum laude",
    "magna cum laude",
    "cum laude",
    "thesis",
    "dissertation",
    "with a thesis",
    "with thesis",
    "honors",
    "honours",
)

PREP_STRIP_RE = re.compile(r"^(?:by|at|in|of)\s+", re.I)


def strip_preps(s: str) -> str:
    return PREP_STRIP_RE.sub("", s or "").strip()


def looks_like_honor(s: str) -> bool:
    low = (s or "").lower()
    return any(tok in low for tok in HONORS_THESIS_TOKENS)


def plausible_org(s: str) -> bool:
    if not s:
        return False
    t = strip_preps(normalize_name(s))
    low = t.lower()
    if low in SECTION_STOP_ORGS:
        return False
    if looks_like_honor(t):
        return False
    if len(t) > 100:
        return False
    toks = [w for w in re.split(r"\W+", t) if w]
    if len(toks) < 2:
        if not toks:
            return False
        token = toks[0]
        if not (token.isupper() and len(token) >= 3):
            return False
    letters = sum(ch.isalpha() for ch in t)
    if letters / max(1, len(t)) < 0.6:
        return False
    return True

def _alias_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


ORG_ALIAS_BY_KEY: Dict[str, str] = {
    "iebusinessschool": "IE University",
    "iebbusinessschool": "IE University",
    "ieuniversity": "IE University",
    "institutodeempresa": "IE University",
    "mbaie": "IE University",
    "universityofnavarra": "Universidad de Navarra",
    "universidaddenavarra": "Universidad de Navarra",
    "udenavarra": "Universidad de Navarra",
}

for uni in SPANISH_UNIVERSITIES:
    ORG_ALIAS_BY_KEY.setdefault(_alias_key(uni), uni)

# Backwards-compatible alias map for clustering helpers (uses human-readable keys).
ORG_ALIASES: Dict[str, str] = {
    "ie": "IE University",
    "ie university": "IE University",
    "instituto de empresa": "IE University",
    "ie business school": "IE University",
    "mba ie": "IE University",
    "u. de navarra": "Universidad de Navarra",
    "u de navarra": "Universidad de Navarra",
    "u navarra": "Universidad de Navarra",
    "universidad navarra": "Universidad de Navarra",
    "university of navarra": "Universidad de Navarra",
}

for uni in SPANISH_UNIVERSITIES:
    ORG_ALIASES.setdefault(uni.lower(), uni)

LOCATION_ALIASES: Dict[str, str] = {
    "spaing": "Spain",
    "u.k.": "United Kingdom",
    "uk": "United Kingdom",
    "uae": "United Arab Emirates",
}

LOCATION_ROLE_TOKENS = {
    "analyst",
    "advisor",
    "adviser",
    "associate",
    "consultant",
    "consulting",
    "data",
    "director",
    "experience",
    "infrastructure",
    "innovation",
    "manager",
    "operations",
    "privacy",
    "risk",
    "strategy",
}


def _looks_like_location(name: str) -> bool:
    if not name:
        return False
    letters = sum(ch.isalpha() for ch in name)
    digits = sum(ch.isdigit() for ch in name)
    if letters < 2:
        return False
    if digits and digits >= letters:
        return False
    if len(name) > 60:
        return False
    words = [tok for tok in name.split() if tok]
    if len(words) > 8:
        return False
    tokens = {tok.strip(". '").lower() for tok in words}
    if tokens & LOCATION_ROLE_TOKENS:
        return False
    return True

DEGREE_MAP: Dict[str, str] = {
    "postdoctoral": "Postdoc",
    "post-doctoral": "Postdoc",
    "postdoctoral studies": "Postdoc",
    "postdoc": "Postdoc",
    "phd": "PhD",
    "ph.d.": "PhD",
    "doctor": "PhD",
    "e.m.b.a.": "MBA",
    "m.b.a.": "MBA",
    "mba": "MBA",
    "msc": "MSc",
    "m.sc.": "MSc",
    "ms": "MSc",
    "master": "Master",
    "masters": "Master",
    "master's": "Master",
    "master’s": "Master",
    "ma": "MA",
    "m.a.": "MA",
    "bachelor": "Bachelor",
    "bachelor's": "Bachelor",
    "bachelor’s": "Bachelor",
    "bsc": "BSc",
    "b.sc.": "BSc",
    "bs": "BSc",
    "b.s.": "BSc",
    "ba": "BA",
    "b.a.": "BA",
    "meng": "MEng",
    "beng": "BEng",
    "certificate": "Certificate",
    "executive certificate": "Certificate",
    "diploma": "Certificate",
}

_FIELD_IN_PAT = re.compile(r"\b(?:in|en)\s+([A-Za-z0-9 &'/-]{3,}?)(?=$|,|;)", re.IGNORECASE)
_FIELD_OF_PAT = re.compile(r"\b(?:of|de|on)\s+([A-Za-z0-9 &'/-]{3,}?)(?=$|,|;)", re.IGNORECASE)

_COUNTRY_PATTERNS = [
    (r"\b(usa|u\.s\.a\.|u\.s\.|united states)\b", "USA"),
    (r"\b(uk|u\.k\.|united kingdom|england|scotland|wales|northern ireland)\b", "United Kingdom"),
    (r"\b(uae|united arab emirates)\b", "United Arab Emirates"),
    (r"\b(spain|españa)\b", "Spain"),
]

_EXPECTED_RELATION_TYPES: Dict[str, Literal["university", "company"]] = {
    "studied_at": "university",
    "worked_at": "company",
    "teaches": "university",
}


def canon_org(name: Optional[str]) -> str:
    """Normalise organisation names and expand known aliases."""

    if not name:
        return ""
    norm = normalize_name(strip_preps(name))
    key = _alias_key(norm)
    return ORG_ALIAS_BY_KEY.get(key, norm)

def canon_location(name: str) -> str:
    """Return a coarse, country-first canonical location (e.g., 'USA', 'Spain').
    If no country pattern is found, return a cleaned version of the input.
    """
    if not name:
        return ""
    n = normalize_name(name)
    if not _looks_like_location(n):
        return ""
    low = n.lower()
    n = LOCATION_ALIASES.get(low, n)
    low = n.lower()
    for pat, canon in _COUNTRY_PATTERNS:
        if re.search(pat, low):
            return canon
    return n  # fallback: leave as cleaned full string (city/state etc.)

def classify_org(name: Optional[str]) -> Optional[str]:
    """Backwards compatible alias for the new soft label helper."""

    if not name:
        return None
    label = soft_name_label(name)
    return None if label == "unknown" else label


def soft_name_label(n: str) -> Literal["university", "company", "government", "unknown"]:
    """Heuristic classifier from the NAME only (soft)."""

    low = (n or "").lower()
    if low in ORG_HARD_POSITIVE:
        return "university"
    if low in ORG_HARD_NEGATIVE:
        return ORG_HARD_NEGATIVE[low]
    if any(c in low for c in UNIV_CUES):
        return "university"
    if any(c in low for c in GOV_CUES):
        return "government"
    if any(c in low for c in COMP_CUES):
        return "company"
    return "unknown"


def normalize_degree(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return a descriptive degree label plus extracted field if available."""

    if not text:
        return None, None
    clean = re.sub(r"\s+", " ", text).strip(" ,.;")
    if not clean:
        return None, None
    lowered = clean.lower()
    lowered_simple = re.sub(r"[^a-z0-9]+", " ", lowered)
    lowered_simple = f" {lowered_simple.strip()} "
    level = None
    for key, value in DEGREE_MAP.items():
        key_norm = re.sub(r"[^a-z0-9]+", " ", key.lower()).strip()
        if not key_norm:
            continue
        needle = f" {key_norm} "
        if needle in lowered_simple:
            level = value
            break
    field: Optional[str] = None
    match = _FIELD_IN_PAT.search(clean)
    if not match:
        match = _FIELD_OF_PAT.search(clean)
    if match:
        field = match.group(1).strip(" ,.;") or None
    label: Optional[str]
    if level and field:
        label = f"{level} in {field}"
    elif level:
        label = level
    else:
        label = clean or None
    return label, field


def year_bin(year: Optional[int], width: int = 5) -> Optional[str]:
    """Return the inclusive year bin for the provided width."""

    if year is None or year < 1900:
        return None
    start = year - ((year - 1900) % width)
    end = start + width - 1
    return f"{start}-{end}"
