import argparse
import io
import json
import re
import time
import urllib.parse
import urllib.robotparser
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, NavigableString

from data.dataset_contracts import describe_array, ensure_dir, sha256_file, utc_now, write_json


try:
    import pdfplumber
except Exception:
    pdfplumber = None


BATTERY_QUERY_SETS = {
    "cathode_degradation": (
        "capacity fade mechanisms in P2-type sodium manganese oxide cathodes",
        "temperature dependence of SEI growth in sodium ion batteries",
        "cycle stability of NASICON sodium vanadium phosphate with different electrolyte salts",
        "rate capability limits of hard carbon anodes for sodium ion cells",
        "structural degradation of O3-type sodium iron manganese oxide during extended cycling",
    ),
    "recycling": (
        "hydrometallurgical recovery of manganese from spent sodium ion battery black mass",
        "shrinking core kinetics of acid leaching for battery cathode recycling",
        "selective leaching of manganese and iron from sodium ion cathode waste",
        "crystallization kinetics during precipitation in battery recycling",
    ),
    "bms_anomaly": (
        "machine learning methods for state of health estimation in sodium ion batteries",
        "graph neural network approaches to thermal runaway precursor detection",
        "early fault detection in lithium battery packs using impedance spectroscopy",
        "temporal graph networks for anomaly detection in battery management systems",
    ),
    "india_context": (
        "thermal management challenges for electric vehicle batteries in Indian climate",
        "cost trajectory of sodium ion battery manufacturing in India",
        "economic viability of second life batteries for grid storage in India",
    ),
    "crystal_structure": (
        "DFT study of P2 layered sodium transition metal oxide crystal structures",
        "phase stability of NASICON sodium vanadium phosphate from first principles calculations",
        "ab initio investigation of sodium intercalation in hard carbon",
    ),
    "eis_degradation": (
        "impedance spectroscopy tracking of SEI growth in sodium ion batteries",
        "Randles circuit parameter evolution during battery aging",
        "charge transfer resistance as early indicator of battery capacity fade",
    ),
}

BATTERY_QUERIES = tuple(q for group in BATTERY_QUERY_SETS.values() for q in group)


@dataclass
class PaperRecord:
    work_id: str
    title: str
    doi: str
    publication_year: int
    venue: str
    source_api: str
    landing_url: str
    pdf_url: str
    is_open_access: bool
    query: str
    abstract: str = ""
    concepts: List[str] = field(default_factory=list)


@dataclass
class MeasurementCandidate:
    measurement_id: str
    work_id: str
    doi: str
    title: str
    formula: str
    property_name: str
    value: float
    unit: str
    context_hash: str
    context: str
    source_url: str
    extraction_method: str
    confidence: float
    cycle_count: Optional[float] = None
    retention_percent: Optional[float] = None
    capacity_mAh_g: Optional[float] = None
    temperature_C: Optional[float] = None
    c_rate: Optional[float] = None
    voltage_low_V: Optional[float] = None
    voltage_high_V: Optional[float] = None


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.last = 0.0

    def wait(self) -> None:
        now = time.time()
        delta = now - self.last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self.last = time.time()


class RobotsCache:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.cache: Dict[str, urllib.robotparser.RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self.cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                rp.read()
            except Exception:
                return True
            self.cache[base] = rp
        try:
            return self.cache[base].can_fetch(self.user_agent, url)
        except Exception:
            return True


class OpenAlexClient:
    def __init__(self, mailto: str = "kineticsforge@example.com"):
        self.base = "https://api.openalex.org/works"
        self.mailto = mailto
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"KineticsForge literature scraper ({mailto})"})
        self.limiter = RateLimiter(0.15)

    def search(self, query: str, max_results: int = 50) -> List[PaperRecord]:
        out: List[PaperRecord] = []
        cursor = "*"
        per_page = min(50, max_results)
        while len(out) < max_results:
            self.limiter.wait()
            params = {
                "search": query,
                "filter": "is_oa:true,type:article",
                "per-page": per_page,
                "cursor": cursor,
                "mailto": self.mailto,
                "select": "id,doi,title,publication_year,primary_location,best_oa_location,open_access,abstract_inverted_index,concepts,primary_topic",
            }
            resp = self.session.get(self.base, params=params, timeout=45)
            resp.raise_for_status()
            payload = resp.json()
            for item in payload.get("results", []):
                out.append(self.to_record(item, query))
                if len(out) >= max_results:
                    break
            next_cursor = payload.get("meta", {}).get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
        return out

    def to_record(self, item: Dict[str, Any], query: str) -> PaperRecord:
        oa = item.get("open_access") or {}
        best = item.get("best_oa_location") or {}
        primary = item.get("primary_location") or {}
        pdf_url = best.get("pdf_url") or primary.get("pdf_url") or ""
        landing_url = best.get("landing_page_url") or primary.get("landing_page_url") or item.get("doi") or item.get("id") or ""
        concepts = [c.get("display_name", "") for c in item.get("concepts", [])[:8] if c.get("display_name")]
        return PaperRecord(
            work_id=str(item.get("id", "")),
            title=str(item.get("title") or ""),
            doi=str(item.get("doi") or "").replace("https://doi.org/", ""),
            publication_year=int(item.get("publication_year") or 0),
            venue=str((primary.get("source") or {}).get("display_name") or ""),
            source_api="openalex",
            landing_url=str(landing_url),
            pdf_url=str(pdf_url),
            is_open_access=bool(oa.get("is_oa", False)),
            query=query,
            abstract=self.abstract_from_inverted_index(item.get("abstract_inverted_index") or {}),
            concepts=concepts,
        )

    @staticmethod
    def abstract_from_inverted_index(index: Dict[str, List[int]]) -> str:
        if not index:
            return ""
        positions: List[Tuple[int, str]] = []
        for word, locs in index.items():
            for loc in locs:
                positions.append((int(loc), word))
        return " ".join(word for _, word in sorted(positions))


class CrossrefClient:
    def __init__(self, mailto: str = "kineticsforge@example.com"):
        self.base = "https://api.crossref.org/works"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"KineticsForge literature scraper (mailto:{mailto})"})
        self.limiter = RateLimiter(0.25)

    def search(self, query: str, max_results: int = 30) -> List[PaperRecord]:
        self.limiter.wait()
        params = {
            "query.bibliographic": query,
            "filter": "type:journal-article,from-pub-date:2000-01-01",
            "rows": min(max_results, 100),
            "select": "DOI,title,published-print,published-online,container-title,URL,abstract,license,link",
        }
        resp = self.session.get(self.base, params=params, timeout=45)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        records: List[PaperRecord] = []
        for item in items:
            title = " ".join(item.get("title") or [])
            year = 0
            for key in ("published-print", "published-online"):
                parts = (item.get(key) or {}).get("date-parts") or []
                if parts and parts[0]:
                    year = int(parts[0][0])
                    break
            links = item.get("link") or []
            pdf_url = ""
            for link in links:
                if "pdf" in str(link.get("content-type", "")).lower() or str(link.get("URL", "")).lower().endswith(".pdf"):
                    pdf_url = str(link.get("URL", ""))
                    break
            records.append(
                PaperRecord(
                    work_id=f"crossref:{item.get('DOI', '')}",
                    title=title,
                    doi=str(item.get("DOI", "")),
                    publication_year=year,
                    venue=" ".join(item.get("container-title") or []),
                    source_api="crossref",
                    landing_url=str(item.get("URL", "")),
                    pdf_url=pdf_url,
                    is_open_access=bool(item.get("license")),
                    query=query,
                    abstract=self.clean_html(item.get("abstract", "")),
                    concepts=[],
                )
            )
        return records

    @staticmethod
    def clean_html(text: str) -> str:
        if not text:
            return ""
        return BeautifulSoup(text, "lxml").get_text(" ", strip=True)


class SemanticScholarClient:
    def __init__(self, mailto: str = "kineticsforge@example.com"):
        self.base = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"KineticsForge literature scraper ({mailto})"})
        self.limiter = RateLimiter(0.35)

    def search(self, query: str, max_results: int = 30) -> List[PaperRecord]:
        self.limiter.wait()
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": "title,year,venue,externalIds,openAccessPdf,abstract,url",
        }
        resp = self.session.get(self.base, params=params, timeout=45)
        resp.raise_for_status()
        records: List[PaperRecord] = []
        for item in resp.json().get("data", []):
            external = item.get("externalIds") or {}
            pdf = item.get("openAccessPdf") or {}
            doi = str(external.get("DOI") or "")
            records.append(
                PaperRecord(
                    work_id=f"semanticscholar:{item.get('paperId', '')}",
                    title=str(item.get("title") or ""),
                    doi=doi,
                    publication_year=int(item.get("year") or 0),
                    venue=str(item.get("venue") or ""),
                    source_api="semantic_scholar",
                    landing_url=str(item.get("url") or (f"https://doi.org/{doi}" if doi else "")),
                    pdf_url=str(pdf.get("url") or ""),
                    is_open_access=bool(pdf.get("url")),
                    query=query,
                    abstract=str(item.get("abstract") or ""),
                    concepts=[],
                )
            )
        return records


class ArxivClient:
    def __init__(self, mailto: str = "kineticsforge@example.com"):
        self.base = "https://export.arxiv.org/api/query"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"KineticsForge literature scraper ({mailto})"})
        self.limiter = RateLimiter(3.0)

    def search(self, query: str, max_results: int = 20) -> List[PaperRecord]:
        self.limiter.wait()
        params = {
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": min(max_results, 50),
            "sortBy": "relevance",
        }
        resp = self.session.get(self.base, params=params, timeout=45)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        records: List[PaperRecord] = []
        for entry in root.findall("atom:entry", ns):
            title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
            summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())
            work_id = entry.findtext("atom:id", default="", namespaces=ns) or ""
            year = 0
            published = entry.findtext("atom:published", default="", namespaces=ns) or ""
            if published[:4].isdigit():
                year = int(published[:4])
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href", "")
            records.append(
                PaperRecord(
                    work_id=f"arxiv:{work_id}",
                    title=title,
                    doi="",
                    publication_year=year,
                    venue="arXiv",
                    source_api="arxiv",
                    landing_url=work_id,
                    pdf_url=pdf_url,
                    is_open_access=True,
                    query=query,
                    abstract=summary,
                    concepts=[],
                )
            )
        return records


class UnpaywallClient:
    def __init__(self, mailto: str = "kineticsforge@example.com"):
        self.mailto = mailto
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"KineticsForge literature scraper ({mailto})"})
        self.limiter = RateLimiter(0.30)

    def enrich(self, record: PaperRecord) -> PaperRecord:
        if not record.doi or record.pdf_url:
            return record
        self.limiter.wait()
        url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(record.doi)}"
        try:
            resp = self.session.get(url, params={"email": self.mailto}, timeout=30)
            if resp.status_code != 200:
                return record
            data = resp.json()
            loc = data.get("best_oa_location") or {}
            pdf = loc.get("url_for_pdf") or ""
            landing = loc.get("url") or loc.get("url_for_landing_page") or ""
            if pdf:
                record.pdf_url = str(pdf)
                record.is_open_access = True
            if landing and not record.landing_url:
                record.landing_url = str(landing)
        except Exception:
            return record
        return record


class DocumentFetcher:
    def __init__(self, root: Path, user_agent: str, obey_robots: bool = True, max_mb: int = 32):
        self.root = ensure_dir(root)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.robots = RobotsCache(user_agent)
        self.obey_robots = obey_robots
        self.max_bytes = max_mb * 1024 * 1024
        self.limiter = RateLimiter(0.6)

    def safe_name(self, record: PaperRecord, suffix: str) -> str:
        token = re.sub(r"[^A-Za-z0-9._-]+", "_", record.doi or record.work_id.split("/")[-1] or record.title[:60]).strip("_")
        return f"{token[:120]}{suffix}"

    def fetch(self, record: PaperRecord) -> Optional[Path]:
        url = record.pdf_url or record.landing_url
        if not url or not url.startswith(("http://", "https://")):
            return None
        if self.obey_robots and not self.robots.allowed(url):
            return None
        self.limiter.wait()
        try:
            with self.session.get(url, stream=True, timeout=60, allow_redirects=True) as resp:
                if resp.status_code >= 400:
                    return None
                ctype = resp.headers.get("content-type", "").lower()
                suffix = ".pdf" if "pdf" in ctype or url.lower().split("?")[0].endswith(".pdf") else ".html"
                path = self.root / self.safe_name(record, suffix)
                if path.exists() and path.stat().st_size > 0:
                    return path
                total = 0
                tmp = path.with_suffix(path.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > self.max_bytes:
                            tmp.unlink(missing_ok=True)
                            return None
                        f.write(chunk)
                tmp.replace(path)
                return path
        except Exception:
            return None


class DocumentTextExtractor:
    def __init__(self, max_pdf_pages: int = 18):
        self.max_pdf_pages = max_pdf_pages

    def extract(self, path: Path) -> Dict[str, Any]:
        if path.suffix.lower() == ".pdf":
            return self.extract_pdf(path)
        return self.extract_html(path)

    def extract_html(self, path: Path) -> Dict[str, Any]:
        raw = path.read_bytes()
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup.find_all(["sub", "sup"]):
            tag.replace_with(NavigableString(tag.get_text("", strip=True)))
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        tables = []
        for idx, table in enumerate(soup.find_all("table")[:30]):
            rows = []
            for tr in table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if cells:
                    rows.append(cells)
            if rows:
                tables.append({"table_index": idx, "rows": rows})
        text = soup.get_text(" ", strip=True)
        return {"text": self.normalize_text(text), "tables": tables, "pages": 0, "kind": "html"}

    def extract_pdf(self, path: Path) -> Dict[str, Any]:
        if pdfplumber is None:
            return {"text": "", "tables": [], "pages": 0, "kind": "pdf_unavailable"}
        text_parts: List[str] = []
        tables: List[Dict[str, Any]] = []
        try:
            with pdfplumber.open(path) as pdf:
                pages = min(len(pdf.pages), self.max_pdf_pages)
                for page_idx in range(pages):
                    page = pdf.pages[page_idx]
                    text_parts.append(page.extract_text() or "")
                    for table_idx, table in enumerate(page.extract_tables() or []):
                        cleaned = [[str(cell or "").strip() for cell in row] for row in table if row]
                        if cleaned:
                            tables.append({"page": page_idx + 1, "table_index": table_idx, "rows": cleaned})
                return {"text": self.normalize_text("\n".join(text_parts)), "tables": tables, "pages": pages, "kind": "pdf"}
        except Exception:
            return {"text": "", "tables": [], "pages": 0, "kind": "pdf_error"}

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.replace("−", "-").replace("–", "-").replace("—", "-")
        text = text.replace("∙", " ").replace("·", " ").replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()


class BatteryEvidenceExtractor:
    capacity_re = re.compile(r"(?P<value>\d{1,3}(?:\.\d+)?)\s*(?:mA\s*h\s*g\s*-?\s*1|mAh\s*g\s*-?\s*1|mAh\s*/\s*g|mA\s*h\s*/\s*g|mAhg-1)", re.I)
    retention_re = re.compile(r"(?P<value>\d{1,3}(?:\.\d+)?)\s*%\s*(?:capacity\s*)?(?:retention|retained|remaining)", re.I)
    recovery_re = re.compile(r"(?P<value>\d{1,3}(?:\.\d+)?)\s*%\s*(?:Mn|manganese|Fe|iron|Na|sodium)?\s*(?:recovery|leaching|extraction|dissolution)", re.I)
    cycle_re = re.compile(r"(?:after|over|for|at|following)?\s*(?P<value>\d{2,5})\s*(?:cycles|cycle\b)", re.I)
    temp_re = re.compile(r"(?P<value>\d{2,3}(?:\.\d+)?)\s*(?:°\s*C|deg\s*C|degrees\s*C|C\b)", re.I)
    c_rate_re = re.compile(r"(?:(?P<direct>\d+(?:\.\d+)?)\s*C\b|C\s*/\s*(?P<divisor>\d+(?:\.\d+)?))", re.I)
    voltage_re = re.compile(r"(?P<low>\d(?:\.\d+)?)\s*(?:-|to|~)\s*(?P<high>\d(?:\.\d+)?)\s*V", re.I)
    formula_re = re.compile(
        r"\b(?:Na|Li)(?:[\d.xδ]+)?(?:(?:[A-Z][a-z]?|\([A-Za-z0-9.]+\))(?:[\d.xδ]+)?){1,12}(?:O(?:[\d.xδ]+)?)?(?:\b|[.,;:)])"
    )
    spaced_formula_re = re.compile(
        r"\b(?:Na|Li)\s*[0-9.x+-]{0,6}\s*(?:(?:Mn|Fe|Ni|Co|Al|Ti|Mg|V|Cr|P|O)\s*[0-9.x+-]{0,6}\s*){2,10}",
        re.I,
    )

    def extract(self, record: PaperRecord, text: str, tables: Sequence[Dict[str, Any]], source_url: str) -> List[MeasurementCandidate]:
        candidates = self.extract_from_text(record, text, source_url, "text")
        for table in tables:
            flattened = self.table_to_text(table)
            candidates.extend(self.extract_from_text(record, flattened, source_url, "table"))
        return self.deduplicate(candidates)

    def extract_from_text(self, record: PaperRecord, text: str, source_url: str, method: str) -> List[MeasurementCandidate]:
        if not text:
            return []
        candidates: List[MeasurementCandidate] = []
        for match in list(self.capacity_re.finditer(text)) + list(self.retention_re.finditer(text)) + list(self.recovery_re.finditer(text)):
            start = max(0, match.start() - 520)
            end = min(len(text), match.end() + 520)
            context = text[start:end]
            formula = self.nearest_formula(context) or self.nearest_formula(text[max(0, match.start() - 1200) : min(len(text), match.end() + 1200)])
            if not formula:
                formula = "unknown"
            value = float(match.group("value"))
            prop, unit = self.property_from_match(match)
            cycle = self.extract_first_float(self.cycle_re, context)
            temp = self.extract_first_float(self.temp_re, context)
            c_rate = self.extract_c_rate(context)
            voltage = self.extract_voltage(context)
            retention = value if prop == "capacity_retention" else self.extract_first_float(self.retention_re, context)
            capacity = value if prop == "specific_capacity" else self.extract_first_float(self.capacity_re, context)
            confidence = self.confidence(prop, formula, cycle, temp, c_rate, voltage, method)
            candidates.append(
                MeasurementCandidate(
                    measurement_id="",
                    work_id=record.work_id,
                    doi=record.doi,
                    title=record.title,
                    formula=self.clean_formula(formula),
                    property_name=prop,
                    value=value,
                    unit=unit,
                    context_hash=self.short_hash(context),
                    context=context[:1400],
                    source_url=source_url,
                    extraction_method=method,
                    confidence=confidence,
                    cycle_count=cycle,
                    retention_percent=retention,
                    capacity_mAh_g=capacity,
                    temperature_C=temp,
                    c_rate=c_rate,
                    voltage_low_V=voltage[0] if voltage else None,
                    voltage_high_V=voltage[1] if voltage else None,
                )
            )
        for idx, cand in enumerate(candidates):
            cand.measurement_id = f"{self.short_hash(cand.work_id + cand.context_hash + cand.property_name + str(idx))}"
        return candidates

    def property_from_match(self, match: re.Match) -> Tuple[str, str]:
        pat = match.re.pattern
        if "mAh" in pat or "mA" in pat:
            return "specific_capacity", "mAh/g"
        if "recovery" in pat or "leaching" in pat:
            return "metal_recovery", "%"
        return "capacity_retention", "%"

    def nearest_formula(self, text: str) -> str:
        formulas = []
        for m in self.formula_re.finditer(text):
            token = m.group(0).strip(".,;:)(")
            if any(el in token for el in ("Mn", "Fe", "Ni", "Co", "V", "P", "O")) and len(token) >= 5:
                formulas.append(token)
        for m in self.spaced_formula_re.finditer(text):
            token = re.sub(r"\s+", "", m.group(0)).strip(".,;:)(")
            if any(el in token for el in ("Mn", "Fe", "Ni", "Co", "V", "P", "O")) and len(token) >= 5:
                formulas.append(token)
        if not formulas:
            return ""
        scored = sorted(formulas, key=lambda x: (("Na" not in x), len(x)))
        return scored[0]

    @staticmethod
    def clean_formula(formula: str) -> str:
        formula = formula.replace(" ", "")
        formula = formula.replace("<sub>", "").replace("</sub>", "")
        formula = formula.replace("δ", "x").replace("∆", "x")
        formula = re.sub(r"[^A-Za-z0-9().x+-]", "", formula)
        return formula.strip(".,;:()[]")

    @staticmethod
    def extract_first_float(pattern: re.Pattern, text: str) -> Optional[float]:
        m = pattern.search(text)
        return float(m.group("value")) if m else None

    @staticmethod
    def extract_c_rate(text: str) -> Optional[float]:
        m = BatteryEvidenceExtractor.c_rate_re.search(text)
        if not m:
            return None
        if m.group("direct"):
            return float(m.group("direct"))
        div = float(m.group("divisor"))
        return 1.0 / div if div else None

    @staticmethod
    def extract_voltage(text: str) -> Optional[Tuple[float, float]]:
        m = BatteryEvidenceExtractor.voltage_re.search(text)
        if not m:
            return None
        low = float(m.group("low"))
        high = float(m.group("high"))
        if low > high:
            low, high = high, low
        if 0.0 < low < 6.0 and 0.0 < high < 6.0:
            return low, high
        return None

    @staticmethod
    def short_hash(text: str) -> str:
        import hashlib

        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]

    @staticmethod
    def confidence(prop: str, formula: str, cycle: Optional[float], temp: Optional[float], c_rate: Optional[float], voltage: Optional[Tuple[float, float]], method: str) -> float:
        score = 0.35
        if formula != "unknown":
            score += 0.20
        if prop == "specific_capacity":
            score += 0.10
        if cycle is not None:
            score += 0.12
        if temp is not None:
            score += 0.08
        if c_rate is not None:
            score += 0.07
        if voltage is not None:
            score += 0.05
        if method == "table":
            score += 0.10
        return float(np.clip(score, 0.05, 0.98))

    @staticmethod
    def table_to_text(table: Dict[str, Any]) -> str:
        rows = table.get("rows") or []
        return " | ".join(" ; ".join(str(cell) for cell in row) for row in rows)

    @staticmethod
    def deduplicate(candidates: Sequence[MeasurementCandidate]) -> List[MeasurementCandidate]:
        seen = set()
        out = []
        for cand in candidates:
            key = (cand.work_id, cand.formula, cand.property_name, round(cand.value, 3), cand.context_hash)
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
        return out


class LiteratureScraperPipeline:
    def __init__(
        self,
        root: Path,
        queries: Sequence[str],
        max_papers: int,
        max_documents: int,
        mailto: str,
        obey_robots: bool,
        download: bool,
    ):
        self.root = root
        self.queries = queries
        self.max_papers = max_papers
        self.max_documents = max_documents
        self.mailto = mailto
        self.download = download
        self.scraped_dir = ensure_dir(root / "real" / "scraped")
        self.raw_dir = ensure_dir(self.scraped_dir / "raw_documents")
        self.openalex = OpenAlexClient(mailto)
        self.crossref = CrossrefClient(mailto)
        self.semantic = SemanticScholarClient(mailto)
        self.arxiv = ArxivClient(mailto)
        self.unpaywall = UnpaywallClient(mailto)
        self.fetcher = DocumentFetcher(self.raw_dir, f"KineticsForge literature scraper ({mailto})", obey_robots=obey_robots)
        self.text_extractor = DocumentTextExtractor()
        self.evidence_extractor = BatteryEvidenceExtractor()

    def discover(self) -> List[PaperRecord]:
        records: List[PaperRecord] = []
        per_query = max(5, int(np.ceil(self.max_papers / max(len(self.queries), 1))))
        for query in self.queries:
            try:
                records.extend(self.openalex.search(query, max_results=per_query))
            except Exception:
                pass
            if len(records) < self.max_papers:
                try:
                    records.extend(self.crossref.search(query, max_results=max(5, per_query // 2)))
                except Exception:
                    pass
            if len(records) < self.max_papers:
                try:
                    records.extend(self.semantic.search(query, max_results=max(5, per_query // 2)))
                except Exception:
                    pass
            if len(records) < self.max_papers:
                try:
                    records.extend(self.arxiv.search(query, max_results=max(3, per_query // 3)))
                except Exception:
                    pass
        dedup: Dict[str, PaperRecord] = {}
        for record in records:
            key = record.doi.lower() if record.doi else record.work_id
            if key and key not in dedup:
                dedup[key] = self.unpaywall.enrich(record)
        selected = list(dedup.values())[: self.max_papers]
        return selected

    def extract_record(self, record: PaperRecord, document_path: Optional[Path]) -> Tuple[List[MeasurementCandidate], Dict[str, Any]]:
        text = " ".join([record.title, record.abstract])
        tables: List[Dict[str, Any]] = []
        document_info = {"path": "", "sha256": "", "kind": "metadata_only", "pages": 0}
        if document_path and document_path.exists():
            extracted = self.text_extractor.extract(document_path)
            if extracted.get("text"):
                text = " ".join([text, extracted["text"]])
            tables = extracted.get("tables") or []
            document_info = {
                "path": str(document_path),
                "sha256": sha256_file(document_path),
                "kind": extracted.get("kind", document_path.suffix.lstrip(".")),
                "pages": int(extracted.get("pages") or 0),
            }
        candidates = self.evidence_extractor.extract(record, text, tables, record.pdf_url or record.landing_url)
        return candidates, document_info

    def run(self) -> Dict[str, Any]:
        records = self.discover()
        metadata_rows: List[Dict[str, Any]] = []
        candidates: List[MeasurementCandidate] = []
        docs_seen = 0
        for record in records:
            document_path = None
            if self.download and docs_seen < self.max_documents and (record.pdf_url or record.landing_url):
                document_path = self.fetcher.fetch(record)
                if document_path:
                    docs_seen += 1
            extracted, doc_info = self.extract_record(record, document_path)
            candidates.extend(extracted)
            row = asdict(record)
            row.update({f"document_{k}": v for k, v in doc_info.items()})
            row["candidate_count"] = len(extracted)
            metadata_rows.append(row)
        metadata = pd.DataFrame(metadata_rows)
        evidence = pd.DataFrame([asdict(c) for c in candidates])
        metadata_path = self.scraped_dir / "paper_metadata.parquet"
        evidence_path = self.scraped_dir / "literature_measurements.parquet"
        jsonl_path = self.scraped_dir / "literature_evidence.jsonl"
        if metadata_path.exists() and not metadata.empty:
            existing = pd.read_parquet(metadata_path)
            metadata = pd.concat([existing, metadata], ignore_index=True)
            metadata["_dedupe_key"] = metadata["doi"].fillna("").str.lower()
            metadata.loc[metadata["_dedupe_key"] == "", "_dedupe_key"] = metadata.loc[metadata["_dedupe_key"] == "", "work_id"]
            metadata = metadata.drop_duplicates("_dedupe_key").drop(columns=["_dedupe_key"])
        if evidence_path.exists() and not evidence.empty:
            existing = pd.read_parquet(evidence_path)
            evidence = pd.concat([existing, evidence], ignore_index=True)
            evidence = evidence.drop_duplicates(["work_id", "formula", "property_name", "value", "context_hash"])
        if not metadata.empty:
            metadata.to_parquet(metadata_path, index=False)
        if not evidence.empty:
            evidence.to_parquet(evidence_path, index=False)
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for cand in candidates:
                    f.write(json.dumps(asdict(cand), ensure_ascii=False) + "\n")
        metrics = self.metrics(metadata, evidence)
        manifest = {
            "created_at": utc_now(),
            "queries": list(self.queries),
            "max_papers": self.max_papers,
            "download_enabled": self.download,
            "records": len(records),
            "documents_downloaded": docs_seen,
            "measurement_candidates": len(candidates),
            "paths": {
                "metadata": str(metadata_path),
                "measurements": str(evidence_path),
                "jsonl": str(jsonl_path),
                "raw_documents": str(self.raw_dir),
            },
            "metrics": metrics,
        }
        write_json(self.scraped_dir / "scrape_manifest.json", manifest)
        return manifest

    @staticmethod
    def metrics(metadata: pd.DataFrame, evidence: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "papers": int(len(metadata)),
            "measurement_rows": int(len(evidence)),
            "papers_with_candidates": int((metadata.get("candidate_count", pd.Series(dtype=int)) > 0).sum()) if not metadata.empty else 0,
        }
        if not evidence.empty:
            out["confidence"] = describe_array(evidence["confidence"].to_numpy())
            for prop in sorted(evidence["property_name"].dropna().unique()):
                vals = evidence.loc[evidence["property_name"] == prop, "value"].to_numpy(dtype=float)
                out[f"{prop}_values"] = describe_array(vals)
            out["formula_count"] = int(evidence["formula"].nunique())
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape open scholarly literature and extract battery measurements with provenance.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--query", action="append", default=None)
    parser.add_argument("--max-papers", type=int, default=60)
    parser.add_argument("--max-documents", type=int, default=18)
    parser.add_argument("--mailto", default="kineticsforge@example.com")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--ignore-robots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = tuple(args.query) if args.query else BATTERY_QUERIES
    pipeline = LiteratureScraperPipeline(
        root=Path(args.root).resolve(),
        queries=queries,
        max_papers=args.max_papers,
        max_documents=args.max_documents,
        mailto=args.mailto,
        obey_robots=not args.ignore_robots,
        download=not args.no_download,
    )
    manifest = pipeline.run()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
