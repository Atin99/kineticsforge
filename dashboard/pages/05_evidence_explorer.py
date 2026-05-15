import streamlit as st
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Evidence Explorer | KineticsForge", page_icon="📋", layout="wide")

st.markdown("# 📋 Evidence Explorer")
st.markdown("Browse registered evidence sources, records, and claim assessments.")

try:
    from core.evidence_registry import build_registry_from_project, default_claims
    registry = build_registry_from_project(PROJECT_ROOT)
    claims = default_claims(registry)

    col1, col2, col3 = st.columns(3)
    col1.metric("Evidence Sources", len(registry.sources))
    col2.metric("Evidence Records", len(registry.records))
    col3.metric("Claims Assessed", len(claims))

    st.markdown("---")
    st.markdown("## Sources")
    for sid, src in registry.sources.items():
        with st.expander(f"**{src.title}** ({src.source_type}) — reliability {src.reliability:.2f}"):
            st.json({"source_id": src.source_id, "uri": src.uri, "license": src.license, "sha256": src.sha256[:16] + "..." if src.sha256 else "", "notes": src.notes})

    st.markdown("---")
    st.markdown("## Claim Assessments")
    for c in claims:
        icon = "✅" if c.verdict == "defensible" else "⚠️" if "plausible" in c.verdict else "❌"
        with st.expander(f"{icon} {c.claim[:80]}... — **{c.verdict}**"):
            st.markdown(f"- **Metric:** {c.metric}")
            st.markdown(f"- **Proposed value:** {c.proposed_value} {c.unit}")
            st.markdown(f"- **Support score:** {c.support_score:.3f}")
            st.markdown(f"- **Contradiction score:** {c.contradiction_score:.3f}")
            st.markdown(f"- **Evidence count:** {c.evidence_count}")
            st.markdown(f"- **Rationale:** {c.rationale}")

    st.markdown("---")
    st.markdown("## Literature Citations")
    citations_path = PROJECT_ROOT / "data" / "real" / "scraped" / "literature_citations.csv"
    if citations_path.exists():
        import csv
        rows = []
        with citations_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("Citations file is empty.")
    else:
        st.warning("No literature citations CSV found.")

except Exception as e:
    st.error(f"Error loading evidence: {e}")
