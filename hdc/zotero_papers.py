"""
hdc/zotero_papers.py
=====================
Zotero → HDC paper index for SNNTraining.

Pulls papers from a private Zotero library (and optionally Google Scholar
profiles), encodes each paper's title + abstract as a binary hypervector
via character n-gram VSA, and stores them in three HDC memories with no
backpropagation:

    HDCGlueAssocMemory      — 11-category knowledge base (XOR + popcount)
    LifeLongSemanticLearner — concept vocabulary that grows with new papers
    CognitiveMapMemory      — similarity-indexed full-text index

Abstract enrichment cascade (for papers without abstracts):
    arXiv API               — free, no rate limit, covers most SNN/HDC papers
    Semantic Scholar API    — fallback (100 req / 5 min, auto rate-limited)
    CrossRef API            — DOI → metadata last resort

Setup
-----
    cp .env.example .env          # then fill in your credentials
    pip install requests
    python -m hdc.zotero_papers --sync   # index all papers once
    python -m hdc.zotero_papers --watch  # continuous daemon (hourly poll)
    python -m hdc.zotero_papers --query "resonator networks factorization"
    python -m hdc.zotero_papers --stats

Required env vars (see .env.example):
    ZOTERO_API_KEY       — from zotero.org/settings/keys
    ZOTERO_LIBRARY_ID    — numeric user ID (printed when you create the key)
    ZOTERO_LIBRARY_TYPE  — "user" (default) or "group"
    ZOTERO_COLLECTION_KEY — optional; leave blank to index the whole library
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import logging
import re
import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

logger = logging.getLogger(__name__)


# ── .env auto-load ─────────────────────────────────────────────────────────────

def _load_dotenv(path: str = ".env"):
    p = Path(path)
    if not p.exists():
        p = Path(__file__).parent.parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()


# ── Constants ──────────────────────────────────────────────────────────────────

ZOTERO_API_BASE       = "https://api.zotero.org"
ARXIV_API_BASE        = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
CROSSREF_BASE         = "https://api.crossref.org/works"

# HDC concept vocabulary for auto-categorisation
_CONCEPT_VOCAB: Dict[str, List[str]] = {
    "resonator_networks":    ["resonator", "factori", "phasor", "fractional power",
                              "hierarchical resonator"],
    "snn_hdc":               ["spiking", "neuromorphic", "spike", "lif neuron",
                              "sensorimotor", "event camera", "active perception"],
    "fault_tolerance":       ["error masking", "fault", "voltage scaling", "reliability",
                              "error propagation", "hype", "bit flip"],
    "memory_structures":     ["cognitive map", "associative memory", "item memory",
                              "semantic vector", "lifelong", "life-long",
                              "summed semantic", "visual place"],
    "hardware":              ["edge ai", "in-memory", "cmos", "computing-in-memory",
                              "mcu", "hdcc", "compiler", "block permute",
                              "stochastic computing", "memristive", "loihi", "rram"],
    "learning_algorithms":   ["one-shot", "online learning", "continual", "refinehd",
                              "adaptive", "incremental", "gluing", "hebbian",
                              "predictive coding", "surprise-driven", "anomaly-triggered"],
    "time_series_hdc":       ["time series", "temporal", "rocket", "minirocket",
                              "multivariate", "seizure", "eeg", "sensor stream",
                              "sensor fusion"],
    "vsa_foundations":       ["xor", "binding", "bundling", "hamming", "popcount",
                              "hyperdimensional", "hypervector", "hdc", "vsa",
                              "vector symbolic", "binary spatter", "fhrr",
                              "superposition"],
    "physical_ai":           ["world model", "physical ai", "digital twin",
                              "physics-informed", "kinematic", "multi-horizon",
                              "planning", "multimodal", "embodied", "robotics"],
    "industrial_automation": ["automation", "industrial", "plant model",
                              "finite state automata", "fault detection",
                              "distributed control"],
    "general":               [],
}
_CATEGORY_ORDER = list(_CONCEPT_VOCAB.keys())

_DIM_RE  = re.compile(r'\b[Dd](?:imension(?:ality)?|[-= ])?\s*[=:]?\s*(\d{3,5})\b')
_ACCU_RE = re.compile(r'(\d{2,3}(?:\.\d+)?)\s*%\s*(?:accuracy|correct)')


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class ZoteroConfig:
    api_key:            str = field(default_factory=lambda: os.environ.get("ZOTERO_API_KEY", ""))
    library_id:         str = field(default_factory=lambda: os.environ.get("ZOTERO_LIBRARY_ID", ""))
    library_type:       str = field(default_factory=lambda: os.environ.get("ZOTERO_LIBRARY_TYPE", "user"))
    collection_key:     str = field(default_factory=lambda: os.environ.get("ZOTERO_COLLECTION_KEY", ""))
    poll_interval_secs: int = 3600
    batch_size:         int = 100
    seen_cache:         str = "hdc_zotero_seen.json"
    state_path:         str = "hdc_learner_state.pt"
    hd_dim:             int = 4096
    device:             str = "cpu"
    max_abstract_chars: int = 4000
    scholar_profiles:   List[Dict] = field(default_factory=list)


# ── Google Scholar scraper (optional) ─────────────────────────────────────────

class _ScholarHTMLParser(HTMLParser):
    """Pull article rows from a public Google Scholar profile page."""

    def __init__(self):
        super().__init__()
        self.articles: List[Dict] = []
        self._cur: Dict = {}
        self._in_title = self._in_venue = self._in_year = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "")
        if tag == "tr" and cls == "gsc_a_tr":
            self._cur = {}
        elif tag == "a" and cls == "gsc_a_at":
            self._in_title = True
        elif tag == "div" and "gs_gray" in cls:
            if "title" in self._cur and "venue" not in self._cur:
                self._in_venue = True
        elif tag == "span" and "gsc_a_h" in cls:
            self._in_year = True

    def handle_data(self, d):
        d = d.strip()
        if not d:
            return
        if self._in_title:
            self._cur["title"] = d
            self._in_title = False
        elif self._in_venue:
            self._cur["venue"] = d
            self._in_venue = False
        elif self._in_year:
            self._cur["year"] = d
            self._in_year = False

    def handle_endtag(self, tag):
        if tag == "tr" and self._cur.get("title"):
            self.articles.append(dict(self._cur))
            self._cur = {}


class GoogleScholarSource:
    """Fetch publications from a public Google Scholar profile page."""

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self, user_id: str, author_name: str, affiliation: str = ""):
        self.user_id     = user_id
        self.author_name = author_name
        self.affiliation = affiliation
        self._url = (
            f"https://scholar.google.com/citations"
            f"?user={user_id}&hl=en&sortby=pubdate&pagesize=100"
        )

    def fetch_articles(self) -> List[Dict]:
        if not _REQUESTS_OK:
            return []
        try:
            r = requests.get(self._url, headers=self._HEADERS, timeout=20)
            if r.status_code != 200:
                logger.warning(f"Scholar [{self.author_name}]: HTTP {r.status_code}")
                return []
            parser = _ScholarHTMLParser()
            parser.feed(r.text)
            for art in parser.articles:
                art["source"]      = f"scholar:{self.user_id}"
                art["author_name"] = self.author_name
            logger.info(f"Scholar [{self.author_name}]: {len(parser.articles)} papers")
            return parser.articles
        except Exception as exc:
            logger.error(f"Scholar [{self.author_name}]: {exc}")
            return []


# ── Abstract enrichment ────────────────────────────────────────────────────────

class AbstractEnricher:
    """
    Fetches missing abstracts via:
      1. arXiv full-text search (free, no rate limit)
      2. Semantic Scholar paper search (auto rate-limited to 100 req / 5 min)
      3. CrossRef title search (free)
    """

    _S2_HEADERS = {"User-Agent": "SNNTraining/1.0 (research; github.com/Enotrium/SNNTraining)"}
    _S2_DELAY   = 1.5

    def __init__(self):
        self._s2_last_call = 0.0
        self._cache: Dict[str, str] = {}

    def _title_key(self, title: str) -> str:
        return hashlib.md5(title.lower().encode()).hexdigest()[:12]

    def fetch(self, title: str, doi: str = "", year: str = "") -> str:
        key = self._title_key(title)
        if key in self._cache:
            return self._cache[key]
        abstract = (
            self._from_arxiv(title)
            or self._from_semantic_scholar(title)
            or self._from_crossref(title, doi)
            or ""
        )
        self._cache[key] = abstract
        return abstract

    def _from_arxiv(self, title: str) -> str:
        try:
            r = requests.get(
                ARXIV_API_BASE,
                params={"search_query": f'ti:"{title[:80]}"', "max_results": 1, "sortBy": "relevance"},
                timeout=15,
            )
            if r.status_code != 200:
                return ""
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(r.text)
            entries = root.findall("atom:entry", ns)
            if not entries:
                return ""
            entry = entries[0]
            found = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            if _title_overlap(title, found) < 0.6:
                return ""
            return ((entry.findtext("atom:summary", "", ns) or "").strip())[:4000]
        except Exception:
            return ""

    def _from_semantic_scholar(self, title: str) -> str:
        elapsed = time.time() - self._s2_last_call
        if elapsed < self._S2_DELAY:
            time.sleep(self._S2_DELAY - elapsed)
        try:
            r = requests.get(
                f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
                params={"query": title[:80], "fields": "title,abstract", "limit": 1},
                headers=self._S2_HEADERS,
                timeout=15,
            )
            self._s2_last_call = time.time()
            if r.status_code in (429, 403):
                return ""
            if r.status_code != 200:
                return ""
            data = r.json().get("data", [])
            if not data:
                return ""
            if _title_overlap(title, data[0].get("title", "")) < 0.6:
                return ""
            return (data[0].get("abstract") or "")[:4000]
        except Exception:
            return ""

    def _from_crossref(self, title: str, doi: str = "") -> str:
        try:
            if doi:
                r = requests.get(f"{CROSSREF_BASE}/{doi}", timeout=12)
            else:
                r = requests.get(
                    CROSSREF_BASE,
                    params={"query.title": title[:80], "rows": 1, "select": "title,abstract"},
                    timeout=12,
                )
            if r.status_code != 200:
                return ""
            msg = r.json().get("message", {})
            items = [msg] if isinstance(msg, dict) else msg.get("items", [])
            if not items:
                return ""
            abstract = items[0].get("abstract", "")
            return re.sub(r"<[^>]+>", " ", abstract).strip()[:4000]
        except Exception:
            return ""


def _title_overlap(a: str, b: str) -> float:
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    return len(wa & wb) / len(wa) if wa else 0.0


# ── Concept extraction ─────────────────────────────────────────────────────────

def extract_category(text: str) -> Tuple[str, List[str]]:
    tl = text.lower()
    best_cat, best_hits = "general", []
    for cat, keywords in _CONCEPT_VOCAB.items():
        hits = [kw for kw in keywords if kw in tl]
        if hits and cat != "general" and len(hits) > len(best_hits):
            best_cat, best_hits = cat, hits
    return best_cat, best_hits


def category_label(cat: str) -> int:
    try:
        return _CATEGORY_ORDER.index(cat)
    except ValueError:
        return len(_CATEGORY_ORDER) - 1


# ── Paper encoder ──────────────────────────────────────────────────────────────

from hdc.hdc_glue import gen_hvs, hv_xor, hv_majority, HDCGlueAssocMemory
from hdc.vector_semantic import LifeLongSemanticLearner
from hdc.cognitive_map import CognitiveMapMemory


class PaperEncoder:
    """Character n-gram VSA encoding: text → binary (dim,) hypervector.

    Uses 2-, 3-, and 4-grams. Each gram is encoded by XOR-binding per-character
    hypervectors with positional permutations, then majority-voting the bundle.
    No external tokenizer or pretrained embeddings needed.
    """

    def __init__(self, dim: int = 4096, seed: int = 42):
        self.dim  = dim
        self.seed = seed
        self._cache: Dict[str, torch.Tensor] = {}

    def _char_hv(self, ch: str) -> torch.Tensor:
        if ch not in self._cache:
            s = self.seed + (hash(ch) & 0x7FFFFFFF)
            self._cache[ch] = gen_hvs(1, self.dim, seed=s).squeeze(0)
        return self._cache[ch]

    def encode_text(self, text: str) -> torch.Tensor:
        text = text.lower()[:4000]
        if not text.strip():
            return gen_hvs(1, self.dim, seed=0).squeeze(0)
        hvs = []
        for n in (2, 3, 4):
            for i in range(len(text) - n + 1):
                gram = text[i:i + n]
                hv = self._char_hv(gram[0])
                for k, ch in enumerate(gram[1:], 1):
                    hv = hv_xor(hv, torch.roll(self._char_hv(ch), shifts=k))
                hvs.append(hv)
        if not hvs:
            return gen_hvs(1, self.dim, seed=0).squeeze(0)
        return hv_majority(torch.stack(hvs).float().mean(dim=0))

    def encode_paper(self, title: str, abstract: str,
                     tags: Optional[List[str]] = None) -> torch.Tensor:
        t_hv   = self.encode_text(title)
        a_hv   = self.encode_text(abstract) if abstract else t_hv
        tag_hv = self.encode_text(" ".join(tags)) if tags else t_hv
        return hv_majority(torch.stack([hv_xor(t_hv, a_hv), tag_hv]).float().mean(dim=0))


@dataclass
class PaperInsights:
    title:        str
    category:     str
    concepts:     List[str]
    dim_mentions: List[int]
    acc_mentions: List[float]
    year:         Optional[str]
    source:       str


def extract_insights(title: str, abstract: str, tags: List[str],
                     year: str = "", source: str = "") -> PaperInsights:
    full = f"{title} {abstract} {' '.join(tags)}".lower()
    cat, concepts = extract_category(full)
    dims = [int(m) for m in _DIM_RE.findall(abstract)]
    accs = [float(m) for m in _ACCU_RE.findall(abstract)]
    return PaperInsights(
        title=title, category=cat, concepts=concepts,
        dim_mentions=dims, acc_mentions=accs,
        year=year or None, source=source,
    )


# ── Multi-source paper learner ─────────────────────────────────────────────────

N_CATEGORIES = len(_CATEGORY_ORDER)


class SNNPaperLearner:
    """
    Pulls papers from Zotero + optional Google Scholar profiles and builds
    a searchable HDC knowledge base — no backpropagation, no GPU required.

    Three memories are maintained:
      knowledge_base  — HDCGlueAssocMemory: 11-topic prototype store
      semantic        — LifeLongSemanticLearner: growing concept vocabulary
      cognitive_map   — CognitiveMapMemory: per-paper HV index for similarity search

    Usage::

        learner = SNNPaperLearner()
        learner.load()          # resume from disk if available
        learner.step()          # fetch & encode all new papers
        results = learner.find_similar("threshold adaptation LIF", top_k=5)
        learner.save()
    """

    def __init__(self, cfg: Optional[ZoteroConfig] = None):
        self.cfg = cfg or ZoteroConfig()
        dim = self.cfg.hd_dim

        self.encoder      = PaperEncoder(dim=dim)
        self.enricher     = AbstractEnricher()

        self.knowledge_base = HDCGlueAssocMemory(n_classes=N_CATEGORIES, dim=dim)
        self.semantic       = LifeLongSemanticLearner(input_dim=dim, dim=dim)
        self.cognitive_map  = CognitiveMapMemory(dim=dim, max_size=10000)

        self._scholar_sources = [
            GoogleScholarSource(p["user_id"], p["name"], p.get("affiliation", ""))
            for p in self.cfg.scholar_profiles
        ]

        self._seen: Dict[str, str] = {}
        self._load_seen()

        self.papers_processed: int = 0
        self.source_counts: Dict[str, int] = {}
        self.total_dim_mentions: List[int]   = []
        self.total_acc_mentions: List[float] = []

    # ── Zotero fetch ────────────────────────────────────────────────────────────

    def _zotero_url(self, start: int = 0) -> str:
        lib = f"{self.cfg.library_type}s/{self.cfg.library_id}"
        if self.cfg.collection_key:
            path = f"/collections/{self.cfg.collection_key}/items"
        else:
            path = "/items"
        return (
            f"{ZOTERO_API_BASE}/{lib}{path}"
            f"?itemType=-attachment"
            f"&include=data&limit={self.cfg.batch_size}&start={start}"
        )

    def _zotero_headers(self) -> Dict[str, str]:
        h = {"Zotero-API-Version": "3"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def _fetch_zotero(self) -> List[Dict]:
        if not _REQUESTS_OK or not self.cfg.library_id:
            logger.warning("ZOTERO_LIBRARY_ID not set — skipping Zotero source")
            return []

        new_items, start = [], 0
        while True:
            try:
                r = requests.get(self._zotero_url(start),
                                 headers=self._zotero_headers(), timeout=20)
            except Exception as exc:
                logger.error(f"Zotero fetch error: {exc}")
                break

            if r.status_code == 200:
                total = int(r.headers.get("Total-Results", 0))
                batch = r.json()
                for item in batch:
                    key = item.get("key", "")
                    pid = f"zotero:{key}"
                    if pid not in self._seen:
                        data = item.get("data", item)
                        title = data.get("title", "").strip()
                        if title:
                            new_items.append({
                                "pid":      pid,
                                "title":    title,
                                "abstract": data.get("abstractNote", ""),
                                "tags":     [t.get("tag", "") for t in data.get("tags", [])],
                                "year":     data.get("date", "")[:4],
                                "doi":      data.get("DOI", ""),
                                "url":      data.get("url", ""),
                                "source":   "zotero",
                            })
                start += len(batch)
                if start >= total or not batch:
                    break
            elif r.status_code == 403:
                logger.error("Zotero 403 — check ZOTERO_API_KEY")
                break
            else:
                logger.warning(f"Zotero HTTP {r.status_code}")
                break

        logger.info(f"Zotero: {len(new_items)} new items")
        return new_items

    # ── Scholar fetch ───────────────────────────────────────────────────────────

    def _fetch_scholar(self) -> List[Dict]:
        new_items = []
        for src in self._scholar_sources:
            for art in src.fetch_articles():
                title = art.get("title", "").strip()
                if not title:
                    continue
                pid = f"scholar:{src.user_id}:{hashlib.md5(title.lower().encode()).hexdigest()[:8]}"
                if pid in self._seen:
                    continue
                new_items.append({
                    "pid":      pid,
                    "title":    title,
                    "abstract": "",
                    "tags":     [],
                    "year":     art.get("year", ""),
                    "doi":      "",
                    "url":      "",
                    "source":   f"scholar:{src.user_id}",
                    "venue":    art.get("venue", ""),
                })
            time.sleep(2)
        return new_items

    # ── Processing ──────────────────────────────────────────────────────────────

    def _enrich_abstracts(self, items: List[Dict]) -> List[Dict]:
        for item in items:
            if item.get("abstract"):
                continue
            abstract = self.enricher.fetch(item["title"],
                                           doi=item.get("doi", ""),
                                           year=item.get("year", ""))
            if abstract:
                item["abstract"] = abstract
        return items

    def _process_paper(self, item: Dict) -> bool:
        title    = item.get("title", "").strip()
        abstract = item.get("abstract", "")[:self.cfg.max_abstract_chars]
        tags     = item.get("tags", [])
        pid      = item["pid"]
        source   = item.get("source", "unknown")
        if not title:
            return False

        hv       = self.encoder.encode_paper(title, abstract, tags)
        insights = extract_insights(title, abstract, tags, item.get("year", ""), source)
        label    = category_label(insights.category)

        self.knowledge_base.add(hv, label)
        self.cognitive_map.store(hv, label=title[:100])
        self.semantic.observe(hv, context=insights.category)

        self.total_dim_mentions.extend(insights.dim_mentions)
        self.total_acc_mentions.extend(insights.acc_mentions)
        self._seen[pid] = title
        self.papers_processed += 1
        self.source_counts[source] = self.source_counts.get(source, 0) + 1

        logger.info(
            f"[{self.papers_processed:4d}] {title[:68]:<68} "
            f"| {insights.category:<24} | {source}"
        )
        return True

    def process_batch(self, items: List[Dict]) -> int:
        items = self._enrich_abstracts(items)
        n = sum(self._process_paper(item) for item in items)
        if n:
            self.knowledge_base.renormalize()
            self.semantic.finalize()
            self._save_seen()
            logger.info(f"Batch +{n}  (total={self.papers_processed})")
        return n

    # ── Query interface ─────────────────────────────────────────────────────────

    def find_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return top-k papers most similar to the query string."""
        q_hv = self.encoder.encode_text(query)
        return self.cognitive_map.query(q_hv, top_k=top_k)

    def category_stats(self) -> Dict[str, int]:
        return {
            cat: int(self.knowledge_base.counts[category_label(cat)].item())
            for cat in _CATEGORY_ORDER
        }

    def source_stats(self) -> Dict[str, int]:
        return dict(self.source_counts)

    def config_recommendations(self) -> Dict[str, Any]:
        rec: Dict[str, Any] = {}
        if self.total_dim_mentions:
            t = torch.tensor(self.total_dim_mentions, dtype=torch.float)
            rec["suggested_hd_dim"] = {
                "median":     int(t.median().item()),
                "mean":       int(t.mean().item()),
                "n_mentions": len(self.total_dim_mentions),
            }
        if self.total_acc_mentions:
            t = torch.tensor(self.total_acc_mentions)
            rec["reported_accuracy"] = {
                "max":        round(float(t.max().item()), 3),
                "mean":       round(float(t.mean().item()), 3),
                "n_mentions": len(self.total_acc_mentions),
            }
        rec["papers_by_topic"]  = self.category_stats()
        rec["papers_by_source"] = self.source_stats()
        rec["total_papers"]     = self.papers_processed
        return rec

    # ── Persistence ─────────────────────────────────────────────────────────────

    def _load_seen(self):
        p = Path(self.cfg.seen_cache)
        if p.exists():
            try:
                self._seen = json.loads(p.read_text())
                logger.info(f"Resumed: {len(self._seen)} previously-seen papers")
            except Exception:
                self._seen = {}

    def _save_seen(self):
        try:
            Path(self.cfg.seen_cache).write_text(json.dumps(self._seen, indent=2))
        except Exception as e:
            logger.warning(f"Could not save seen cache: {e}")

    def save(self, path: Optional[str] = None):
        p = path or self.cfg.state_path
        torch.save({
            "version":          2,
            "papers_processed": self.papers_processed,
            "source_counts":    self.source_counts,
            "kb_prototypes":    self.knowledge_base.prototypes,
            "kb_counts":        self.knowledge_base.counts,
            "seen":             self._seen,
            "dim_mentions":     self.total_dim_mentions,
            "acc_mentions":     self.total_acc_mentions,
            "cogmap_vectors":   [v.cpu() for v in self.cognitive_map.vectors],
            "cogmap_labels":    list(self.cognitive_map.labels),
        }, p)
        logger.info(f"Saved → {p}  ({self.papers_processed} papers)")

    def load(self, path: Optional[str] = None):
        p = path or self.cfg.state_path
        if not Path(p).exists():
            logger.info("No saved state — starting fresh")
            return
        state = torch.load(p, map_location="cpu")
        self.papers_processed           = state["papers_processed"]
        self.source_counts              = state.get("source_counts", {})
        self.knowledge_base.prototypes.copy_(state["kb_prototypes"])
        self.knowledge_base.counts.copy_(state["kb_counts"])
        self._seen                      = state.get("seen", {})
        self.total_dim_mentions         = state.get("dim_mentions", [])
        self.total_acc_mentions         = state.get("acc_mentions", [])
        if "cogmap_vectors" in state:
            self.cognitive_map.vectors = state["cogmap_vectors"]
            self.cognitive_map.labels  = state.get("cogmap_labels", [])
        logger.info(f"Loaded: {self.papers_processed} papers, "
                    f"{len(self.cognitive_map.vectors)} in cognitive map")

    # ── Main loop ────────────────────────────────────────────────────────────────

    def step(self) -> int:
        """One full fetch cycle across all sources. Returns new papers processed."""
        all_items: List[Dict] = []
        logger.info("── Zotero ─────────────────────────")
        all_items.extend(self._fetch_zotero())
        if self._scholar_sources:
            logger.info("── Google Scholar ─────────────────")
            all_items.extend(self._fetch_scholar())
        if not all_items:
            logger.info("No new papers.")
            return 0
        logger.info(f"New items: {len(all_items)} — enriching abstracts…")
        n = self.process_batch(all_items)
        if n:
            self.save()
        return n

    def run(self, max_steps: int = 0):
        """Continuous self-learning daemon (polls every poll_interval_secs)."""
        logger.info(
            f"SNNPaperLearner daemon started\n"
            f"  collection: {self.cfg.collection_key or 'full library'}\n"
            f"  poll interval: {self.cfg.poll_interval_secs}s"
        )
        i = 0
        while True:
            self.step()
            i += 1
            if max_steps and i >= max_steps:
                break
            logger.info(f"Sleeping {self.cfg.poll_interval_secs}s | "
                        f"total: {self.papers_processed}")
            time.sleep(self.cfg.poll_interval_secs)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(
        prog="python -m hdc.zotero_papers",
        description="SNNTraining paper index — Zotero → HDC knowledge base",
    )
    p.add_argument("--sync",     action="store_true", help="One-shot fetch all new papers")
    p.add_argument("--watch",    action="store_true", help="Continuous daemon mode")
    p.add_argument("--query",    type=str, default="", metavar="TEXT",
                   help="Find papers similar to TEXT")
    p.add_argument("--recommend",action="store_true", help="Print HDC config recommendations from literature")
    p.add_argument("--stats",    action="store_true", help="Print per-source and per-topic stats")
    p.add_argument("--interval", type=int, default=3600, help="Poll interval seconds (default: 3600)")
    p.add_argument("--dim",      type=int, default=4096, help="Hypervector dimension (default: 4096)")
    p.add_argument("--top-k",    type=int, default=5,    help="Results for --query (default: 5)")
    args = p.parse_args()

    cfg     = ZoteroConfig(poll_interval_secs=args.interval, hd_dim=args.dim)
    learner = SNNPaperLearner(cfg)
    learner.load()

    if args.query:
        results = learner.find_similar(args.query, top_k=args.top_k)
        print(f'\nTop-{args.top_k} for: "{args.query}"\n')
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.get('similarity', 0):.3f}]  {r.get('label', '<unlabelled>')}")
        return

    if args.recommend:
        import json as _j
        print("\nHDC config recommendations from literature:")
        print(_j.dumps(learner.config_recommendations(), indent=2))
        return

    if args.stats:
        import json as _j
        print("\nSource stats:")
        print(_j.dumps(learner.source_stats(), indent=2))
        print("\nTopic distribution:")
        print(_j.dumps(learner.category_stats(), indent=2))
        print(f"\nTotal: {learner.papers_processed} papers")
        return

    if args.sync:
        n = learner.step()
        print(f"\nSynced {n} new papers  (total: {learner.papers_processed})")
        return

    if args.watch:
        learner.run()
        return

    p.print_help()


if __name__ == "__main__":
    _cli()
