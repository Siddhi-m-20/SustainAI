import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB_DIR = PROJECT_ROOT / "knowledge_base"


@dataclass
class KnowledgeChunk:
    chunk_id: str
    source_file: str
    title: str
    content: str
    sdg_tag: str


class KnowledgeRetriever:
    """
    RAG Knowledge Retriever for EcoSync Resource Manager.
    Indexes sustainability standards, protocols, and SDG frameworks,
    providing high-relevance semantic retrieval for AI Copilot grounding.
    """

    def __init__(self, kb_dir: Optional[Path] = None):
        self.kb_dir = kb_dir or KB_DIR
        self.chunks: List[KnowledgeChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _determine_sdg(self, filename: str, text: str) -> str:
        text_lower = (filename + " " + text).lower()
        if "energy" in text_lower or "ashrae" in text_lower or "sdg 7" in text_lower or "kwh" in text_lower:
            return "SDG 7: Clean Energy"
        elif "water" in text_lower or "leak" in text_lower or "sdg 6" in text_lower:
            return "SDG 6: Clean Water"
        elif "waste" in text_lower or "compost" in text_lower or "sdg 12" in text_lower or "diversion" in text_lower:
            return "SDG 12: Responsible Consumption"
        return "Cross-Cutting SDG"

    def _load_and_index(self) -> None:
        """
        Loads all markdown documents from knowledge_base directory and chunks them by sections.
        """
        self.chunks.clear()
        if not self.kb_dir.exists():
            return

        md_files = list(self.kb_dir.glob("*.md"))
        for file_path in md_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                # Split by markdown headers ## or ###
                sections = re.split(r"\n(?=##?\s)", content)
                for i, sec in enumerate(sections):
                    clean_sec = sec.strip()
                    if not clean_sec or len(clean_sec) < 40:
                        continue

                    lines = clean_sec.split("\n")
                    first_line = lines[0].replace("#", "").strip()
                    title = first_line if first_line else file_path.stem.replace("_", " ").title()

                    sdg_tag = self._determine_sdg(file_path.name, clean_sec)
                    chunk = KnowledgeChunk(
                        chunk_id=f"{file_path.stem}_{i}",
                        source_file=file_path.name,
                        title=title,
                        content=clean_sec,
                        sdg_tag=sdg_tag,
                    )
                    self.chunks.append(chunk)
            except Exception as e:
                print(f"[KnowledgeRetriever] Error loading {file_path}: {e}")

        if self.chunks:
            corpus = [f"{c.title}\n{c.content}\n{c.sdg_tag}" for c in self.chunks]
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs semantic TF-IDF cosine similarity search.
        """
        if not self.chunks or not self.vectorizer or self.tfidf_matrix is None:
            return []

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "title": chunk.title,
                "content": chunk.content,
                "sdg_tag": chunk.sdg_tag,
                "similarity_score": round(score, 4),
            })

        return results

    def search_and_format(self, query: str, top_k: int = 2) -> str:
        """
        Returns nicely formatted reference block for Copilot prompt injection or citations.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return "No matching sustainability standards found in local knowledge base."

        formatted_chunks = []
        for r in results:
            formatted_chunks.append(
                f"### [{r['sdg_tag']}] {r['title']} (Source: {r['source_file']})\n{r['content']}"
            )
        return "\n\n".join(formatted_chunks)

    def get_all_chunks(self) -> List[KnowledgeChunk]:
        return self.chunks
