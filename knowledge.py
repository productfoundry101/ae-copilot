"""Unstructured knowledge: the seven sales enablement documents.

No vector database, no embeddings, deliberately. The corpus is seven small
markdown files (~40KB). The model gets a table of contents (doc names,
descriptions, section headings) and requests exactly the doc or section it
needs. Exact retrieval beats approximate search at this scale, and every
citation points to a real section heading a human can open and check.
"""

from __future__ import annotations

from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# name -> (filename, one-line description the model sees)
DOCS = {
    "sales_playbook": (
        "01_sales_playbook.md",
        "How Personio sells: MEDDIC qualification, deal stages with exit "
        "criteria, multi-threading rules, renewal motion, common failure "
        "modes, pre-call data checklist",
    ),
    "icp": (
        "02_icp.md",
        "Ideal customer profile: sweet-spot table, buying triggers, "
        "anti-patterns to deprioritise, persona briefs (CHRO, People Ops, "
        "CFO, IT, HRBP)",
    ),
    "battlecard_workday": (
        "03_battlecard_workday.md",
        "Competing against Workday: when we win/lose, their pitch, our "
        "counters, objection talk tracks, red-flag combinations",
    ),
    "battlecard_hibob": (
        "04_battlecard_hibob.md",
        "Competing against HiBob: when we win/lose, their pitch, our "
        "counters, objection talk tracks, positioning notes",
    ),
    "objection_handling": (
        "06_objection_handling.md",
        "Objection pattern library: the 4-step method and responses to "
        "common objections (price, 'need to think', proposals, competitors, "
        "security, budget)",
    ),
    "pricing_cheatsheet": (
        "07_pricing_cheatsheet.md",
        "Pricing structure (PEPM tiers, modules, implementation), discount "
        "authority limits, when deal desk approval is required. Contextual: "
        "never quote directly to customers",
    ),
    "customer_case_studies": (
        "08_customer_case_studies.md",
        "Named customer proof points with industry/region/size and 'use as "
        "reference for' tags, plus one lost-deal pattern (Petrichor)",
    ),
}


def _read(name: str) -> str:
    filename, _ = DOCS[name]
    return (KNOWLEDGE_DIR / filename).read_text()


def _sections(text: str) -> dict[str, str]:
    """Split a markdown doc into {heading: body} on '## ' headings."""
    sections: dict[str, str] = {}
    current = "(intro)"
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            sections[current] = "\n".join(buf).strip()
            current = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    sections[current] = "\n".join(buf).strip()
    return sections


def list_docs() -> list[dict]:
    """Table of contents the model uses to decide what to read."""
    out = []
    for name, (filename, desc) in DOCS.items():
        secs = list(_sections(_read(name)).keys())
        out.append({"name": name, "description": desc, "sections": secs})
    return out


def case_studies() -> list[dict]:
    """Structured index of the customer case studies, parsed from the doc.

    Headers look like: '## Sable Hospitality (Iberia, Hospitality, 650 employees)'
    and each entry has a 'Use as a reference for:' line. This powers
    deterministic reference matching in the signal engine.
    """
    import re

    out = []
    for heading, body in _sections(_read("customer_case_studies")).items():
        m = re.match(r"(Lost deal: )?(.+?) \((.+?), (.+?), ([\d,]+) employees\)",
                     heading)
        if not m:
            continue
        use_for = ""
        for line in body.splitlines():
            if "Use as a reference for:" in line:
                use_for = line.split("Use as a reference for:")[-1].strip(" *")
        out.append({
            "name": m.group(2).strip(),
            "is_lost_deal": bool(m.group(1)),
            "region": m.group(3).strip(),
            "industry": m.group(4).strip(),
            "employees": int(m.group(5).replace(",", "")),
            "use_as_reference_for": use_for,
            "section_heading": heading,
        })
    return out


def read_doc(name: str, section: str | None = None) -> dict:
    """Return one whole doc, or a single section of it."""
    if name not in DOCS:
        return {"error": f"Unknown doc '{name}'. Valid: {list(DOCS)}"}
    text = _read(name)
    if not section:
        return {"doc": name, "content": text}
    secs = _sections(text)
    # forgiving match: exact, else case-insensitive substring
    if section in secs:
        return {"doc": name, "section": section, "content": secs[section]}
    for heading, body in secs.items():
        if section.lower() in heading.lower():
            return {"doc": name, "section": heading, "content": body}
    return {"error": f"No section matching '{section}' in {name}. "
                     f"Sections: {list(secs)}"}
