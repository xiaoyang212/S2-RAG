from extract_graph import Extractor
from build_tree import sequential_merge
from typing import List, Tuple, Dict, Set
from itertools import combinations
import networkx as nx
import faiss
import spacy
from collections import defaultdict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import random
import logging

random.seed(1)
import copy
import numpy as np

# Get logger for this module
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        cache_tree,
        G: nx.Graph,
        index,
        appearance_count: Dict[str, int],
        nlp: Extractor,
        **kwargs,
    ) -> None:

        self.cache_tree = cache_tree
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(
            self.cache_tree
        )
        self.G = G
        self.index = index
        self.appearance_count = appearance_count
        # get the inverse index, i.e., chunk_id to noun index.
        self.inverse_index = self.get_inverse_index()
        self.nlp = nlp
        # set up the parameters.
        self.device = kwargs.get("device", "cuda:0")
        self.merge_num = kwargs.get("merge_num", 5)
        self.min_count = kwargs.get("min_count", 2)
        self.overlap = kwargs.get("overlap", 100)
        self.tokenizer = kwargs.get("tokenizer", "/path/to/your/model")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        if kwargs.get("embedder", "BAAI/bge-m3") is not None:
            self.embedder = SentenceTransformer(
                kwargs.get("embedder", "BAAI/bge-m3"), device=self.device
            )
            self.faiss_index = self._build_faiss_index()
        else:
            logger.warning(
                "Warning: the embedder is set to None, dense retrieval is not implemented."
            )
            self.embedder = None
            self.faiss_index = None
        # Cross-Encoder (lazy init)

    def __del__(self):
        """Ensure proper cleanup of resources
        avoid the memory leak.
        """
        try:
            if hasattr(self, "embedder"):
                del self.embedder
            if hasattr(self, "faiss_index"):
                del self.faiss_index
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during Retriever cleanup: {e}")

    def update(self, cache_tree, G, index, appearance_count):
        # update the retriever, from a document to another document.
        self.cache_tree = cache_tree
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(
            self.cache_tree
        )
        self.G = G
        self.index = index
        self.appearance_count = appearance_count
        self.inverse_index = self.get_inverse_index()
        self.docs = self.collapse_tree
        if self.embedder is not None:
            self.faiss_index = self._build_faiss_index()

    def get_inverse_index(self):
        # get the inverse index, i.e., chunk_id to noun index.
        inverse_index = {}
        for key, value in self.index.items():
            for chunk_id in value:
                inverse_index.setdefault(chunk_id, []).append(key)
        return inverse_index

    def _collapse_tree(self, cache_tree: Dict[str, Dict]) -> Dict[str, Dict]:
        # collapse the tree. for dense retrieval
        # return the collapsed tree.
        collapsed_tree = []
        collapsed_tree_ids = []
        for key, value in self.cache_tree.items():
            collapsed_tree.append(value["text"])
            collapsed_tree_ids.append(key)
        return collapsed_tree, collapsed_tree_ids

    def _build_faiss_index(self):
        # build the faiss index.
        # only used when the dense retrieval is implemented.
        # return the faiss index.
        docs = self.collapse_tree
        if self.embedder is None:
            self.embedder = SentenceTransformer("BAAI/bge-m3", device=self.device)
            self.embedder.eval()
            logger.info(
                "the embedder is not set, using the default embedder BAAI/bge-m3."
            )
        doc_embeds = self.embedder.encode(docs, batch_size=16, device=self.device)
        # print("doc_embeds examples", doc_embeds[0:5][0:5])
        # print("doc_embeds shape", doc_embeds.shape)
        vector_database = faiss.IndexFlatIP(doc_embeds.shape[1])
        vector_database.add(doc_embeds)
        return vector_database

    def index_mapping(self, entities: list) -> List[str]:
        # get the chunks from the cache tree.
        # for two types:
        # 1. entity is a list of strings, i.e., the entities are not related,
        chunk_ids = {}

        for entity in entities:
            if isinstance(entity, str):
                if entity in self.index.keys():
                    chunk_ids[entity] = self.index[entity]
            elif isinstance(entity, tuple):
                chunk_ids_set = set()
                entity_key = "_".join(entity)
                for e in entity:
                    if e in self.index.keys():
                        if chunk_ids_set == set():
                            chunk_ids_set = set(self.index[e])
                        else:
                            chunk_ids_set = chunk_ids_set & set(self.index[e])
                chunk_ids[entity_key] = sorted(list(chunk_ids_set))

        return chunk_ids

    def graph_filter(self, entities: List[str], k) -> List[str]:
        # get the shortest path between the entities.
        shortest_path_pairs = []
        for head, tail in combinations(entities, 2):
            if head in self.G.nodes() and tail in self.G.nodes():
                try:
                    shortest_path = nx.shortest_path(self.G, head, tail)
                except nx.NetworkXNoPath:
                    continue
                if len(shortest_path) <= k:
                    shortest_path_pairs.append((head, tail))

        # shortest_path_pairs = self.merge_tuples(shortest_path_pairs)
        return shortest_path_pairs

    def merge_tuples(self, lst):
        graph = defaultdict(set)

        for a, b in lst:
            graph[a].add(b)
            graph[b].add(a)

        visited = set()
        result = []

        def dfs(entity, cluster):
            if entity in visited:
                return
            visited.add(entity)
            cluster.add(entity)
            for neighbor in graph[entity]:
                dfs(neighbor, cluster)

        for a, b in lst:
            if a not in visited:
                cluster = set()
                dfs(a, cluster)
                result.append(tuple(sorted(cluster)))

        return result

    def merge_keys(self, neighbor_nodes: Dict[str, List[str]]) -> Dict[str, List[str]]:
        # merge the nodes with different keys.
        # return with the same format.
        chunks_to_keys = defaultdict(set)
        for key, chunk_lists in neighbor_nodes.items():
            for chunk in chunk_lists:
                chunks_to_keys[chunk].add(key)

        merged_result = {}
        for chunk, keys in chunks_to_keys.items():
            # get the new key.
            if len(keys) > 1:
                all_entities = set()
                for key in keys:
                    all_entities.update(key.split("_"))
                new_key = "_".join(sorted(all_entities))
            else:
                new_key = keys.pop()
            # add the chunk to the new key.
            if new_key in merged_result.keys():
                merged_result.setdefault(new_key, []).append(chunk)
            else:
                merged_result[new_key] = [chunk]
        return merged_result

    def format_res(self, res: Dict[str, List[str]]) -> str:
        res_str = ""
        for key, chunks in res.items():
            chunks = self.detect_contiguous_chunks(chunks)
            for chunk_list in chunks:
                str_of_list = self.get_contiguous_chunks(chunk_list)
                res_str += "{}: {}\n".format(key, str_of_list)
        return res_str

    def _format_flat_chunks(self, flat_hits: List[dict]) -> str:
        lines = []
        for item in flat_hits:
            rank = item.get("rank", 0)
            cid = item.get("chunk_id", "")
            es = item.get("entity_score", 0.0)
            ss = item.get("summary_score", 0.0)
            fs = item.get("fused_score", 0.0)
            src = item.get("source", "")
            text = item.get("text", "")
            lines.append(
                f"[{rank}] {cid} (src={src}; fused={fs:.4f}, E={es:.4f}, S={ss:.4f})"
            )
            if text:
                lines.append(text)
            lines.append("---")
        return "\n".join(lines).rstrip("-\n")

    def _build_flat_hits_from_mapping(
        self,
        chunk_mapping: Dict[str, List[str]],
        entities: List[str],
        default_source: str = "mixed",
    ) -> List[dict]:
        seen = set()
        hits = []
        entities = entities or []
        rank = 1
        for _, chunk_ids in chunk_mapping.items():
            for chunk_id in chunk_ids:
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                text = self.cache_tree.get(chunk_id, {}).get("text", "")
                chunk_entities = [
                    e for e in entities if chunk_id in self.index.get(e, [])
                ]
                source = "entity" if chunk_entities else default_source
                hits.append(
                    {
                        "rank": rank,
                        "chunk_id": chunk_id,
                        "text": text,
                        "entities": chunk_entities,
                        "entity_score": 0.0,
                        "summary_score": 0.0,
                        "fused_score": 0.0,
                        "source": source,
                    }
                )
                rank += 1
        return hits

    def local_retrieval(
        self, entities: List[str], shortest_path_k: int = 4
    ) -> Dict[str, List[str]]:
        # initialize by shortest path
        shortest_path = self.graph_filter(entities, shortest_path_k)
        # it returns the list of pairs existing shortest path shorter than k.

        # initialize the chunks.
        init_chunk_ids = self.index_mapping(shortest_path)

        neighbor_nodes = self.merge_keys(init_chunk_ids)
        return neighbor_nodes

    def dense_retrieval(self, query, k):
        # using dense retrieval to get the chunks.
        query_embed = self.embedder.encode(query).reshape(
            1, -1
        )  # need (1, -1) for faiss.
        _, condidate_chunks_indexs = self.faiss_index.search(query_embed, k=k)
        # the normal faiss index return the (1, k) shape. squeeze it to (k,).
        condidate_chunks_indexs = condidate_chunks_indexs[0]
        condidate_chunk_ids = [
            self.collapse_tree_ids[i] for i in condidate_chunks_indexs
        ]
        res = {"": condidate_chunk_ids}
        return res

    def _count_chunks(self, res: Dict[str, List[str]]) -> int:
        # count the chunks.
        count = 0
        for chunk_ids in res.values():
            count += len(chunk_ids)
        return count

    def _calculate_entity_pagerank(
        self, entities: List[str], damping_factor: float = 0.85
    ) -> Dict[str, float]:

        if not entities or self.G is None:
            return {entity: 1.0 for entity in entities}

        try:
            subgraph_nodes = set(entities)

            for entity in entities:
                if entity in self.G.nodes():
                    subgraph_nodes.update(self.G.neighbors(entity))

            if len(subgraph_nodes) > 1:
                subgraph = self.G.subgraph(subgraph_nodes)

                pagerank_scores = nx.pagerank(subgraph, alpha=damping_factor)

                entity_pagerank = {}
                for entity in entities:
                    entity_pagerank[entity] = pagerank_scores.get(entity, 0.0)

                return entity_pagerank
            else:
                return {entity: 1.0 for entity in entities}

        except Exception as e:
            return {entity: 1.0 for entity in entities}

    def _entity_scoring(
        self, query_entities: List[str], candidate_chunks: Dict[str, List[str]]
    ) -> Dict[str, float]:

        entity_scores = {}

        entity_pagerank = self._calculate_entity_pagerank(query_entities)

        for key, chunk_ids in candidate_chunks.items():
            for chunk_id in chunk_ids:
                entity_score = 0.0
                key_entities = key.split("_") if key else []

                chunk_appearance = self.appearance_count.get(chunk_id, {})

                for entity in query_entities:
                    entity_freq = chunk_appearance.get(entity, 0)
                    pagerank_weight = entity_pagerank.get(entity, 1.0)
                    weighted_score = entity_freq * (1 + pagerank_weight)
                    entity_score += weighted_score

                if len(query_entities) > 1:
                    co_occurrence_bonus = len(
                        [e for e in query_entities if e in key_entities]
                    ) / len(query_entities)

                    matched_entities = [e for e in query_entities if e in key_entities]
                    if matched_entities:
                        avg_pagerank = sum(
                            entity_pagerank.get(e, 1.0) for e in matched_entities
                        ) / len(matched_entities)
                        entity_score *= 1 + co_occurrence_bonus * avg_pagerank
                    else:
                        entity_score *= 1 + co_occurrence_bonus

                entity_scores[chunk_id] = entity_score

        return entity_scores

    def _summary_scoring(self, query: str, k: int = 50) -> Dict[str, float]:
        """Calculate semantic similarity scores for chunks using dense retrieval"""
        if self.embedder is None or self.faiss_index is None:
            return {}

        # Get dense retrieval results with scores
        query_embed = self.embedder.encode(query).reshape(1, -1)
        scores, chunk_indices = self.faiss_index.search(query_embed, k=k)

        # Convert to chunk_id -> score mapping
        summary_scores = {}
        for i, chunk_idx in enumerate(chunk_indices[0]):
            if chunk_idx >= 0:  # Valid index
                chunk_id = self.collapse_tree_ids[chunk_idx]
                summary_scores[chunk_id] = float(scores[0][i])

        return summary_scores

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0,1] range using rank-based normalization (with tie handling)."""
        if not scores:
            return scores

        items = list(scores.items())
        n = len(items)

        if n == 1:
            only_id = items[0][0]
            return {only_id: 0.5}

        sorted_items = sorted(items, key=lambda x: x[1])

        normalized_scores = {}
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_items[j + 1][1] == sorted_items[i][1]:
                j += 1

            avg_rank = ((i + 1) + (j + 1)) / 2.0
            norm_val = (avg_rank - 1) / (n - 1) 

            for k in range(i, j + 1):
                chunk_id = sorted_items[k][0]
                normalized_scores[chunk_id] = norm_val

            i = j + 1

        return normalized_scores

    def _calculate_dynamic_weights(
        self, query: str, entities: List[str]
    ) -> Tuple[float, float]:

        if not query or not query.strip():
            return 0.5, 0.5

        q = query.strip()
        tokens = [t for t in q.split() if t]
        token_len = len(tokens) if tokens else len(q)  
        token_len = max(token_len, 1)

        entity_count = len(entities)
        entity_ratio = entity_count / token_len if token_len > 0 else 0.0
        if entity_ratio < 0:
            entity_ratio = 0.0
        elif entity_ratio > 1:
            entity_ratio = 1.0
        if entity_ratio > 0.5:
            alpha = 0.7
        elif entity_ratio < 0.2:
            alpha = 0.3
        else:
            alpha = 0.5

        beta = 1.0 - alpha

        return round(alpha, 4), round(beta, 4)

    def _fuse_scores(
        self,
        entity_scores: Dict[str, float],
        summary_scores: Dict[str, float],
        alpha: float,
        beta: float,
        threshold: float = 0.2,
    ) -> Dict[str, float]:
        """Fuse normalized entity and summary scores with given weights"""
        # Get all unique chunk IDs
        all_chunk_ids = set(entity_scores.keys()) | set(summary_scores.keys())

        fused_scores = {}
        for chunk_id in all_chunk_ids:
            entity_score = entity_scores.get(chunk_id, 0.0)
            summary_score = summary_scores.get(chunk_id, 0.0)

            # Calculate fused score
            fused_score = alpha * entity_score + beta * summary_score

            # Apply threshold filter
            if fused_score >= threshold:
                fused_scores[chunk_id] = fused_score

        return fused_scores

    def fusion_retrieval(
        self, query: str, entities: List[str], **kwargs
    ) -> Dict[str, List[str]]:
        """
        Fusion retrieval that combines entity graph and summary tree retrieval
        following the pipeline described in the problem statement
        """
        max_chunks = kwargs.get("max_chunk_setting", 25)
        k_candidates = max_chunks  # Retrieve more candidates for fusion

        # Step 1: Independent retrieval and scoring
        # print("Step 1: Independent retrieval and scoring")

        # Entity graph retrieval
        entity_candidates = {}
        if entities:
            # Use local retrieval to get entity-based candidates
            shortest_path_k = kwargs.get("shortest_path_k", 4)
            entity_candidates = self.local_retrieval(entities, shortest_path_k)
            max_chunks = kwargs.get("max_chunk_setting", 25)
            chunk_count = self._count_chunks(entity_candidates)
            while chunk_count > max_chunks and shortest_path_k > 1:
                shortest_path_k -= 1
                entity_candidates = self.local_retrieval(entities, shortest_path_k)
                chunk_count = self._count_chunks(entity_candidates)

        # Calculate entity scores
        entity_scores = self._entity_scoring(entities, entity_candidates)

        # Summary tree retrieval and scoring
        summary_scores = self._summary_scoring(query, k_candidates)

        # Step 2: Score normalization
        # print("Step 2: Score normalization")
        entity_scores_norm = self._normalize_scores(entity_scores)
        summary_scores_norm = self._normalize_scores(summary_scores)

        # Step 3: Dynamic weight calculation and fusion
        # print("Step 3: Dynamic weight calculation and fusion")
        alpha, beta = self._calculate_dynamic_weights(query, entities)
        print(f"Dynamic weights: alpha={alpha}, beta={beta}")

        fused_scores = self._fuse_scores(
            entity_scores_norm, summary_scores_norm, alpha, beta
        )

        sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        reranked_chunk_ids = [
            chunk_id for chunk_id, score in sorted_chunks[:max_chunks]
        ]
        result_dict = {}
        for chunk_id in reranked_chunk_ids:
            chunk_entities = [e for e in entities if chunk_id in self.index.get(e, [])]
            key = "_".join(sorted(chunk_entities)) if chunk_entities else ""
            result_dict.setdefault(key, []).append(chunk_id)
        flat_hits = []
        for rank, chunk_id in enumerate(reranked_chunk_ids, 1):
            chunk_entities = [e for e in entities if chunk_id in self.index.get(e, [])]
            flat_hits.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "text": self.cache_tree.get(chunk_id, {}).get("text", ""),
                    "entities": chunk_entities,
                    "entity_score": entity_scores_norm.get(chunk_id, 0.0),
                    "summary_score": summary_scores_norm.get(chunk_id, 0.0),
                    "fused_score": fused_scores.get(chunk_id, 0.0),
                    "source": "entity" if chunk_entities else "summary",
                }
            )

        score_details = [
            {
                "chunk_id": cid,
                "entity_score": entity_scores_norm.get(cid, 0.0),
                "summary_score": summary_scores_norm.get(cid, 0.0),
                "fused_score": sc,
            }
            for cid, sc in sorted(
                fused_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]

        return {
            "flat_hits": flat_hits,  # 新：扁平重排列表
            "reranked_chunk_ids": reranked_chunk_ids,
            "score_details": score_details,  # 新：得分明细
            "alpha": alpha,
            "beta": beta,
            "grouped_hits": self.merge_keys(result_dict),  # 旧结构（兼容/调试）
        }

    def query_fusion(self, query: str, **kwargs) -> dict:
        # Step 1: Extract entities from query
        entities = self.nlp.naive_extract_graph(query.split("\n")[0])
        entities = entities["nouns"]
        print(f"Extracted entities: {entities}")

        if self.embedder is None or self.faiss_index is None:
            logger.warning(
                "Embedder or FAISS index not available, falling back to original retrieval"
            )
            return self.query(query, **kwargs)

        fusion_result = self.fusion_retrieval(query, entities, **kwargs)

        res_str = self._format_flat_chunks(fusion_result.get("flat_hits", []))
        result = {
            "chunks": res_str,
            "flat_hits": fusion_result.get("flat_hits", []),
            "fusion_details": fusion_result,
        }

        if kwargs.get("debug", True):
            grouped = fusion_result.get("grouped_hits", {})
            supplement_info = self._build_supplement_info(
                grouped,
                entities,
                grouped,
                list(grouped.keys()),
                self._count_chunks(grouped),
                [],
            )
            result.update(supplement_info)
            result["retrieval_type"] = "Fusion Retrieval (flat)"
        return result


    def query(self, query, **kwargs):
        if kwargs.get("use_fusion", False):
            return self.query_fusion(query, **kwargs)


    def _build_supplement_info(
        self,
        chunk_ids,
        entities,
        neighbor_nodes,
        keys,
        len_chunks,
        chunk_counts_history,
    ):
        return {
            "chunk_ids": chunk_ids,
            "entities": entities,
            "neighbor_nodes": neighbor_nodes,
            "keys": keys,
            "len_chunks": len_chunks,
            "chunk_counts_history": chunk_counts_history,
        }
