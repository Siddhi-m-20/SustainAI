"""src/ui/pages/knowledge.py — Sustainability Knowledge page."""
import streamlit as st

from src.ui.components import page_header, section_label


def render(knowledge_retriever) -> None:
    page_header("Knowledge Base", "Search sustainability standards, engineering protocols, and best practices")

    query = st.text_input("Search guidelines…", placeholder="e.g. ASHRAE setback, MNF leak, waste diversion", key="kb_query")

    if query:
        results = knowledge_retriever.search(query, top_k=5)
        if results:
            st.markdown(f"**{len(results)} results** for *\"{query}\"*")
            for r in results:
                score = r.get("similarity_score", 0)
                score_pct = int(score * 100)
                with st.expander(
                    f"📖 {r['title']} &nbsp;·&nbsp; *{r['source_file']}* &nbsp;·&nbsp; relevance {score_pct}%",
                    expanded=(score_pct > 20),
                ):
                    st.markdown(
                        f'<span class="label-fact">{r.get("sdg_tag","")}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(r["content"])
                    st.caption(f"Source: `{r['source_file']}` · Score: {score:.4f}")
        else:
            st.info("No matching content found. Try different keywords.")

    st.markdown("---")
    section_label("Browse All")

    chunks = knowledge_retriever.get_all_chunks()
    if not chunks:
        st.info("Knowledge base is empty.")
        return

    # Group by source file
    from collections import defaultdict
    by_source: dict = defaultdict(list)
    for c in chunks:
        by_source[c.source_file].append(c)

    domain_icons = {
        "energy_efficiency_standards": "⚡",
        "water_conservation_protocols": "💧",
        "waste_diversion_circularity": "♻️",
        "sdg_alignment_framework": "🌍",
    }

    for source_file, chunks_in_file in by_source.items():
        stem = source_file.replace(".md", "")
        icon = domain_icons.get(stem, "📄")
        sdg_tag = chunks_in_file[0].sdg_tag if chunks_in_file else ""
        with st.expander(f"{icon} {stem.replace('_', ' ').title()} &nbsp;·&nbsp; {len(chunks_in_file)} sections", expanded=False):
            st.caption(f"SDG alignment: **{sdg_tag}** · Source: `{source_file}`")
            for c in chunks_in_file:
                st.markdown(f"**{c.title}**")
                st.markdown(c.content)
                st.markdown("---")
