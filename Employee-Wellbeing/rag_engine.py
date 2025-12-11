# rag_engine.py

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from config import HR_POLICY_DIR, RAG_DB_DIR, EMBEDDING_MODEL_NAME


@dataclass
class PolicyChunk:
    doc_id: str
    section: str
    category: str
    text: str


CATEGORY_MAP = {
    "mental_health_wellness": "Mental Health",
    "burnout_prevention": "Burnout",
    "flexible_work": "Flexible Work",
    "recognition_rewards": "Recognition",         
    "performance_management": "Performance",      
    "leave_time_off": "Leave",
    "employee_wellbeing_framework": "Wellbeing",
    "health_wellness_benefits": "Health Benefits", 
    "workplace_safety_health": "Safety",           
    "manager_training": "Manager Training",        
}


class HRPolicyRAGEngine:
    def __init__(self, policy_dir: Path = HR_POLICY_DIR, persist_dir: Path = RAG_DB_DIR):
        self.policy_dir = policy_dir
        self.persist_dir = persist_dir
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[PolicyChunk] = []
        self.embeddings: Optional[np.ndarray] = None

    # ---------- Chunking ----------
    def _split_into_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Split by numbered headings: '1. ...', '2. ...' etc.
        If doc has no such headings, just single section.
        """
        pattern = r"(?m)^\s*(\d+\. .+)$"
        matches = list(re.finditer(pattern, text))
        if not matches:
            return [{"section": "General", "body": text.strip()}]

        sections = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            heading = m.group(1).strip()
            body = text[start:end].strip()
            sections.append({"section": heading, "body": body})
        return sections

    def _chunk_section(self, doc_id: str, category: str, section: str, body: str,
                       max_tokens: int = 400, overlap_tokens: int = 50) -> List[PolicyChunk]:
        """
        Very rough token control using words. Good enough for demo.
        """
        words = body.split()
        if len(words) <= max_tokens:
            return [PolicyChunk(doc_id, section, category, body)]

        chunks = []
        step = max_tokens - overlap_tokens
        for start in range(0, len(words), step):
            end = min(len(words), start + max_tokens)
            text = " ".join(words[start:end])
            chunks.append(PolicyChunk(doc_id, section, category, text))
            if end == len(words):
                break
        return chunks

    def load_policies(self) -> List[PolicyChunk]:
        self.chunks = []
        for path in sorted(self.policy_dir.glob("*.txt")):
            doc_id = path.stem
            category = CATEGORY_MAP.get(doc_id, "General")
            text = path.read_text(encoding="utf-8")
            sections = self._split_into_sections(text)
            for sec in sections:
                section_name = sec["section"]
                body = sec["body"]
                self.chunks.extend(
                    self._chunk_section(doc_id, category, section_name, body)
                )
        return self.chunks

    def build_index(self) -> None:
        if not self.chunks:
            self.load_policies()

        texts = [c.text for c in self.chunks]
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        # normalize for cosine similarity
        faiss.normalize_L2(emb)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)

        self.index = index
        self.embeddings = emb

        self.persist_dir.mkdir(exist_ok=True)
        faiss.write_index(index, str(self.persist_dir / "hr_policies.index"))
        np.save(self.persist_dir / "hr_policies_meta.npy", emb.shape)

        # serialize chunks as simple list-of-dicts
        # (if you want to be fancy, use pickle)
        import json
        with open(self.persist_dir / "hr_policies_chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.__dict__ for c in self.chunks], f, ensure_ascii=False, indent=2)

    def load_index(self) -> None:
        index_path = self.persist_dir / "hr_policies.index"
        chunks_path = self.persist_dir / "hr_policies_chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            self.build_index()
            return

        self.index = faiss.read_index(str(index_path))
        import json
        with open(chunks_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.chunks = [PolicyChunk(**c) for c in raw]

    def query(
        self,
        question: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> List[PolicyChunk]:
        if self.index is None:
            self.load_index()

        q_emb = self.model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, top_k * 2)

        results = []
        for i in idxs[0]:
            chunk = self.chunks[i]
            if category_filter and chunk.category != category_filter:
                continue
            results.append(chunk)
            if len(results) >= top_k:
                break

        # if filter removed everything, fall back to unfiltered
        if not results and category_filter:
            for i in idxs[0][:top_k]:
                results.append(self.chunks[i])

        return results
