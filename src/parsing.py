"""Utilities for cleaning and segmenting professor bios."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List

from bs4 import BeautifulSoup


_SECTION_MAP = {
    "corporate experience": "corporate_experience",
    "academic experience": "academic_experience",
    "academic background": "academic_background",
}


def clean_html(html: str) -> str:
    """Return plain text cleaned from the incoming HTML snippet."""

    soup = BeautifulSoup(html or "", "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(" \n", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sections(html: str) -> Dict[str, str]:
    """Split the biography by <h4> headings into predefined sections."""

    soup = BeautifulSoup(html or "", "lxml")
    sections: Dict[str, List[str]] = defaultdict(list)

    current_key = "intro"
    for element in soup.recursiveChildGenerator():
        if getattr(element, "name", None) == "h4":
            heading = element.get_text(" ", strip=True).lower()
            current_key = _SECTION_MAP.get(heading, "intro")
            continue
        if getattr(element, "string", None):
            text = element.string.strip()
            if not text:
                continue
            sections[current_key].append(text)

    if not sections:
        sections["intro"].append(clean_html(html))

    return {k: " ".join(v).strip() for k, v in sections.items() if v}


def sentences(text: str) -> List[str]:
    """Split a block of text into light-weight sentences."""

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
