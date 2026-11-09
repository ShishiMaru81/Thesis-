#!/usr/bin/env python3
"""
verify_refs.py — fetch, verify and export the thesis bibliography.

For every reference in refs_spec.py this script
  1. fetches the record from an authoritative source (Crossref for DOIs, the
     arXiv API, the ACL Anthology, DBLP, or the official web page for
     @online sources) and caches the raw response under
     thesis/generated/refs_cache/ (re-runs read the cache; delete a file to
     refetch);
  2. checks that the fetched title matches the expected title;
  3. checks that each evidence phrase recorded for the claim we cite the work
     for occurs in the fetched text (abstract, or page text for @online);
  4. writes thesis/generated/citation_check.csv and
     thesis/bibliography/references.bib (only references that pass).

A reference that fails any check is written to citation_check.csv with
verified = no and is left out of references.bib; the text must then use
\\CITENEEDED{} instead.

Network access is needed only for records that are not cached. No personal
data is sent: requests carry a generic User-Agent and no e-mail address.
"""
from __future__ import annotations

import csv
import html
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "thesis" / "generated" / "refs_cache"
CSV_OUT = ROOT / "thesis" / "generated" / "citation_check.csv"
BIB_OUT = ROOT / "thesis" / "bibliography" / "references.bib"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from refs_spec import REFS, VERIFIED_ON  # noqa: E402

UA = "thesis-reference-check/1.0 (BRAC University undergraduate thesis; python urllib)"


# --------------------------------------------------------------------------- fetch
import urllib.error  # noqa: E402


def fetch(url: str, name: str, accept: str | None = None, pause: float = 1.0) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / name
    if f.exists():
        return f.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": UA, **({"Accept": accept} if accept else {})})
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                enc = r.headers.get_content_charset() or "utf-8"
            text = raw.decode(enc, errors="replace")
            f.write_text(text, encoding="utf-8", newline="\n")
            time.sleep(pause)
            return text
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 401, 403, 404, 410):
                break
            time.sleep(10 * (attempt + 1))        # 429 / 5xx: back off
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"fetch failed for {url}: {last}")




def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", html.unescape(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\\[a-zA-Z]+\s*|[{}$\\]", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def strip_tags(s: str) -> str:
    s = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


# --------------------------------------------------------------------------- sources
def from_arxiv(aid: str) -> dict:
    """arXiv record: the API response if cached, otherwise the arxiv.org abstract page (citation_* meta tags)."""
    if (CACHE / f"arxiv_{slug(aid)}.xml").exists():
        x = fetch("", f"arxiv_{slug(aid)}.xml")
        entry = x.split("<entry>", 1)[1] if "<entry>" in x else ""
        g = lambda tag: (re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", entry, re.S) or [None, ""])[1]
        authors = re.findall(r"<author>\s*<name>(.*?)</name>", entry, re.S)
        return {"title": re.sub(r"\s+", " ", g("title")).strip(), "authors": [a.strip() for a in authors],
                "year": g("published")[:4], "abstract": re.sub(r"\s+", " ", g("summary")).strip(),
                "journal_ref": g("arxiv:journal_ref").strip(), "arxiv_doi": g("arxiv:doi").strip(),
                "url": f"https://arxiv.org/abs/{aid}", "source": "arXiv API"}
    page = fetch(f"https://arxiv.org/abs/{aid}", f"arxivabs_{slug(aid)}.html", pause=3.5)
    meta = lambda n: [html.unescape(m) for m in re.findall(rf'<meta name="{n}" content="(.*?)"\s*/?>', page, re.S)]
    ab = re.search(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>', page, re.S)
    jref = re.search(r'<td class="tablecell jref">(.*?)</td>', page, re.S)
    doi = re.search(r'<td class="tablecell doi">.*?doi\.org/([^"<]+)', page, re.S)
    authors = []
    for a in meta("citation_author"):
        last, _, first = a.partition(",")
        name = f"{first.strip()} {last.strip()}".strip()
        if re.search(r"[A-Za-z]", name):          # the Phi-4-mini page lists ":" as an author
            authors.append(name)
    return {"title": re.sub(r"\s+", " ", (meta("citation_title") or [""])[0]).strip(), "authors": authors,
            "year": (meta("citation_date") or [""])[0][:4],
            "abstract": re.sub(r"^\s*Abstract:\s*", "", strip_tags(ab.group(1))) if ab else "",
            "journal_ref": strip_tags(jref.group(1)) if jref else "", "arxiv_doi": doi.group(1).strip() if doi else "",
            "url": f"https://arxiv.org/abs/{aid}", "source": "arXiv abstract page"}


def from_crossref(doi: str) -> dict:
    j = json.loads(fetch(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}", f"crossref_{slug(doi)}.json"))["message"]
    year = None
    for k in ("published-print", "published-online", "issued", "published"):
        if j.get(k, {}).get("date-parts"):
            year = j[k]["date-parts"][0][0]
            break
    authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in j.get("author", [])]
    btype = {"journal-article": "article", "proceedings-article": "inproceedings"}.get(j.get("type", ""), "article")
    vfield = "journaltitle" if btype == "article" else "booktitle"
    fields = {"author": " and ".join(f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in j.get("author", [])),
              "title": "{" + html.unescape(" ".join(j.get("title", [""]))) + "}",
              vfield: html.unescape(" ".join(j.get("container-title", [""]))),
              "volume": j.get("volume", ""), "number": j.get("issue", ""),
              "pages": (j.get("page", "") or "").replace("–", "--").replace("-", "--").replace("----", "--"),
              "year": str(year or ""), "doi": doi}
    built = f"@{btype}{{KEY,\n" + "\n".join(f"  {k:<12}= {{{v}}}," for k, v in fields.items() if v).rstrip(",") + "\n}"
    return {"title": " ".join(j.get("title", [""])), "authors": authors, "year": str(year or ""),
            "venue": " ".join(j.get("container-title", [""])), "abstract": strip_tags(j.get("abstract", "")),
            "doi": doi, "url": f"https://doi.org/{doi}", "bibtex": built, "source": "Crossref"}


def from_acl(aid: str) -> dict:
    bib = fetch(f"https://aclanthology.org/{aid}.bib", f"acl_{slug(aid)}.bib")
    page = fetch(f"https://aclanthology.org/{aid}/", f"acl_{slug(aid)}.html")
    m = re.search(r'acl-abstract[^>]*>(.*?)</div>', page, re.S)
    abstract = strip_tags(m.group(1)) if m else ""
    abstract = re.sub(r"^Abstract\s*", "", abstract)
    title = (re.search(r'title\s*=\s*"(.*?)",\s*\n', bib, re.S) or [None, ""])[1]
    year = (re.search(r'year\s*=\s*"(\d{4})"', bib) or [None, ""])[1]
    venue = (re.search(r'booktitle\s*=\s*"(.*?)",\s*\n', bib, re.S) or re.search(r'journal\s*=\s*"(.*?)",\s*\n', bib, re.S) or [None, ""])[1]
    authors = (re.search(r'author\s*=\s*"(.*?)",\s*\n', bib, re.S) or [None, ""])[1]
    return {"title": re.sub(r"\s+", " ", title), "authors": [a.strip() for a in re.sub(r"\s+", " ", authors).split(" and ")],
            "year": year, "venue": re.sub(r"\s+", " ", venue), "abstract": abstract, "bibtex": bib,
            "url": f"https://aclanthology.org/{aid}/", "source": "ACL Anthology"}


def from_venue(r: dict) -> dict:
    """Confirm a conference paper that has no DOI against the venue's own record.

    kind = "listing":    the official proceedings index page (NeurIPS, PMLR) must list the title.
    kind = "openreview": the OpenReview record of the paper must carry an accepted-venue string
                         such as "ICLR 2022 Poster"."""
    v = r["venue"]
    if v["kind"] == "listing":
        page = fetch(v["url"], f"venue_{slug(v['url'])}.html", pause=2.0)
        if norm(r["title"]) not in norm(strip_tags(page)):
            raise RuntimeError(f"title not found in proceedings listing {v['url']}")
        found = v["expect"]
    else:
        j = json.loads(fetch(v["url"], f"venue_{slug(r['key'])}.json", pause=2.0))
        found = ""
        for n in j.get("notes", []):
            c = n.get("content", {})
            t = c.get("title", {})
            t = t.get("value", "") if isinstance(t, dict) else t
            ven = c.get("venue", {})
            ven = ven.get("value", "") if isinstance(ven, dict) else ven
            if norm(t) == norm(r["title"]):
                found = ven
                break
        if v["expect"] not in found:
            raise RuntimeError(f"OpenReview venue {found!r} does not contain {v['expect']!r}")
    return {"venue": v["booktitle"], "year": v["year"], "venue_evidence": found, "venue_url": v["url"]}


def from_url(url: str, name: str) -> dict:
    page = fetch(url, f"url_{slug(name)}.html")
    if page.lstrip().startswith("{"):                     # JSON API (Hugging Face model record)
        j = json.loads(page)
        return {"title": j.get("id", ""), "page_text": page, "url": url, "source": "official API record"}
    t = (re.search(r"<title[^>]*>(.*?)</title>", page, re.S | re.I) or [None, ""])[1]
    return {"title": strip_tags(t) or page.strip().splitlines()[0], "page_text": strip_tags(page), "url": url,
            "source": "official web page"}


def from_pubmed(pmid: str) -> dict:
    """Abstract from PubMed (NCBI E-utilities) when the publisher page blocks automated access."""
    x = fetch(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml",
              f"pubmed_{pmid}.xml")
    ab = re.search(r"<Abstract>(.*?)</Abstract>", x, re.S)
    return {"abstract": strip_tags(ab.group(1)) if ab else "", "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}


def from_openalex(query: str, key: str) -> dict:
    """Fallback for a paper whose publisher page blocks automated access (JSTOR)."""
    j = json.loads(fetch(f"https://api.openalex.org/works?search={urllib.parse.quote(query)}&per-page=3",
                         f"openalex_{slug(key)}.json"))
    w = j["results"][0]
    src_ = (w.get("primary_location") or {}).get("source") or {}
    return {"title": w.get("title", ""), "authors": [a["author"]["display_name"] for a in w.get("authorships", [])],
            "year": str(w.get("publication_year", "")), "venue": src_.get("display_name", ""),
            "url": w.get("id", ""), "source": "OpenAlex (publisher page blocks automated access)"}


# --------------------------------------------------------------------------- bibtex
BIB_DROP = ("timestamp", "biburl", "bibsource", "abstract", "editor", "month", "language", "publisher_place")


def rekey(bib: str, key: str) -> str:
    bib = bib.strip()
    bib = re.sub(r"^@(\w+)\s*\{\s*[^,]+,", lambda m: f"@{m.group(1).lower()}{{{key},", bib, count=1)
    lines = []
    skip = False
    for line in bib.splitlines():
        field = re.match(r"\s*(\w+)\s*=", line)
        if field:
            skip = field.group(1).lower() in BIB_DROP
        if not skip:
            lines.append(line)
        if skip and re.search(r"[}\"],?\s*$", line) and line.count("{") <= line.count("}"):
            skip = False
    out = "\n".join(lines)
    out = re.sub(r",\s*\n\}$", "\n}", out)
    return out


def bib_field(k: str, v: str) -> str:
    return f"  {k:<11}= {{{v}}},"


def make_bib(r: dict, f: dict) -> str:
    key = r["key"]
    if r.get("bib"):                      # hand-written entry (@online, books, reports)
        fields = [bib_field(k, v) for k, v in r["bib"].items()]
        return f"@{r['bibtype']}{{{key},\n" + "\n".join(fields).rstrip(",") + "\n}"
    if f.get("bibtex"):
        b = rekey(f["bibtex"], key)
        extra = r.get("add", {})
        if "arxiv" in r and "eprint" not in b:
            extra = {"eprint": r["arxiv"], "eprinttype": "arxiv", **extra}
        if extra:
            b = b.rstrip("}").rstrip().rstrip(",") + ",\n" + "\n".join(bib_field(k, v) for k, v in extra.items()).rstrip(",") + "\n}"
        return b
    if "venue" in r:                      # conference paper without DOI: venue confirmed by from_venue()
        fields = {"author": " and ".join(f["authors"]), "title": "{" + f["title"] + "}", "booktitle": f["venue"],
                  "year": f["year"], "eprint": r["arxiv"], "eprinttype": "arxiv", **r.get("add", {})}
        return f"@inproceedings{{{key},\n" + "\n".join(bib_field(k, v) for k, v in fields.items()).rstrip(",") + "\n}"
    # arXiv-only preprint
    fields = {"author": " and ".join(f["authors"]), "title": "{" + f["title"] + "}", "year": f["year"],
              "eprint": r["arxiv"], "eprinttype": "arxiv", "howpublished": "arXiv preprint", **r.get("add", {})}
    if f.get("arxiv_doi"):
        fields["doi"] = f["arxiv_doi"]
    return f"@misc{{{key},\n" + "\n".join(bib_field(k, v) for k, v in fields.items()).rstrip(",") + "\n}"


# --------------------------------------------------------------------------- main
def resolve(r: dict) -> dict:
    f: dict = {}
    if "doi" in r:
        f.update(from_crossref(r["doi"]))
    if "acl" in r:
        f.update(from_acl(r["acl"]))
    if "arxiv" in r:
        a = from_arxiv(r["arxiv"])
        for k, v in a.items():
            if k == "abstract" and f.get("abstract"):
                f["abstract"] = f["abstract"] + " || arXiv: " + v
            elif k in ("url", "source") and f.get(k):
                f[k] = f[k] + " + " + v
            elif not f.get(k):
                f[k] = v
    if "openalex" in r:
        f.update(from_openalex(r["openalex"], r["key"]))
    if "pmid" in r:
        p = from_pubmed(r["pmid"])
        f["abstract"] = (f.get("abstract", "") + " " + p["abstract"]).strip()
        f["source"] = f.get("source", "") + " + PubMed abstract"
        f["url"] = f.get("url", "") + " + " + p["pubmed_url"]
    if "venue" in r:
        f.update(from_venue(r))
        f["source"] = f.get("source", "") + f" + venue record ({r['venue']['kind']})"
    if "url" in r:
        u = from_url(r["url"], r["key"])
        for k, v in u.items():
            if k in ("url", "source") and f.get(k):
                f[k] = f[k] + " + " + v
            elif not f.get(k):
                f[k] = v
    return f


def main() -> int:
    rows, bibs, fails = [], [], []
    for r in REFS:
        try:
            f = resolve(r)
        except Exception as e:  # noqa: BLE001
            f = {"error": str(e)}
        text = " ".join([f.get("abstract", ""), f.get("page_text", ""), f.get("title", ""), f.get("journal_ref", "")])
        ntext = norm(text)
        ft = norm(f.get("title", ""))
        title_ok = (bool(ft) and norm(r["title"]) in ft) or (len(ft) > 10 and ft in norm(r["title"])) \
            or ("url" in r and norm(r["title"]) in norm(f.get("page_text", "")))
        ev_missing = [e for e in r.get("evidence", []) if norm(e) not in ntext]
        ok = "error" not in f and title_ok and not ev_missing and bool(r.get("evidence")) and r.get("claim_checked", True)
        note = r.get("note", "")
        if r.get("title_only"):
            note += ("title-level check: the source record has no abstract; title, authors, year and venue confirmed, "
                     "and the work is cited only as the original publication of the method or resource it names")
        if f.get("venue_evidence"):
            note += f"; venue confirmed at {f.get('venue_url')} ({f['venue_evidence'][:60]})"
        if "error" in f:
            note = f"fetch error: {f['error']}; " + note
        if not title_ok:
            note = f"title mismatch: fetched '{f.get('title', '')}'; " + note
        if ev_missing:
            note = f"evidence not found: {ev_missing}; " + note
        rows.append({"key": r["key"], "title": f.get("title", r["title"]), "authors": "; ".join(f.get("authors", []))[:300] or r.get("authors", ""),
                     "year": f.get("year", r.get("year", "")), "venue": f.get("venue", "") or f.get("journal_ref", "") or r.get("venue", ""),
                     "doi_or_arxiv": r.get("doi") or (f"arXiv:{r['arxiv']}" if "arxiv" in r else "") or r.get("acl", ""),
                     "url": f.get("url", r.get("url", "")), "verified_on_date": VERIFIED_ON,
                     "claim_we_cite_it_for": r["claim"], "verified": "yes" if ok else "no",
                     "note": (f"source: {f.get('source', '')}; evidence: {' | '.join(r.get('evidence', []))}; " + note).strip("; ")})
        if ok:
            bibs.append(make_bib(r, f))
        else:
            fails.append(r["key"])
    with CSV_OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    BIB_OUT.parent.mkdir(parents=True, exist_ok=True)
    BIB_OUT.write_text("% AUTO-GENERATED by scripts/thesis/verify_refs.py from verified records only.\n"
                       "% Do not edit by hand; edit scripts/thesis/refs_spec.py and re-run.\n\n"
                       + "\n\n".join(bibs) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(REFS)} references; {len(bibs)} verified; {len(fails)} failed: {fails}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
