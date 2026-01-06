"""LLM-based re-ranking for improved search relevance."""

import os
from typing import List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import SearchResult


class LLMReranker:
    """Re-rank search results using an LLM for improved relevance."""
    
    DEFAULT_MODEL = "gpt-4o-mini"
    
    RERANK_PROMPT = """You are a relevance scoring assistant. Given a query and a document chunk, rate how relevant the chunk is to answering the query.

Query: {query}

Document chunk:
{content}

Rate the relevance from 0.0 to 1.0 where:
- 0.0 = Completely irrelevant
- 0.3 = Tangentially related
- 0.5 = Somewhat relevant
- 0.7 = Relevant
- 1.0 = Highly relevant, directly answers the query

Respond with ONLY a single decimal number between 0.0 and 1.0."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        openai_api_key: Optional[str] = None,
    ):
        self.model = model
        self.client = OpenAI(api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"))
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    def _score_single(self, query: str, content: str) -> float:
        """Score a single document's relevance to query."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.RERANK_PROMPT.format(query=query, content=content[:2000]),
                    }
                ],
                temperature=0.0,
                max_tokens=10,
            )
            
            score_text = response.choices[0].message.content.strip()
            return float(score_text)
        except (ValueError, AttributeError):
            return 0.5  # Default score on parse error
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Re-rank search results using LLM scoring.
        
        Args:
            query: The search query
            results: Initial search results
            top_n: Return only top N results (default: all)
        
        Returns:
            Re-ranked list of SearchResult objects
        """
        if not results:
            return results
        
        # Score each result
        scored_results = []
        for result in results:
            llm_score = self._score_single(query, result.content)
            # Combine original score with LLM score (weighted average)
            combined_score = 0.3 * result.score + 0.7 * llm_score
            result.score = combined_score
            result.metadata["llm_score"] = llm_score
            result.metadata["original_score"] = result.score
            scored_results.append(result)
        
        # Sort by new score
        scored_results.sort(key=lambda x: x.score, reverse=True)
        
        if top_n:
            scored_results = scored_results[:top_n]
        
        return scored_results
    
    def rerank_batch(
        self,
        query: str,
        results: List[SearchResult],
        top_n: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Re-rank using a single batched LLM call (more efficient).
        
        Args:
            query: The search query
            results: Initial search results  
            top_n: Return only top N results (default: all)
        
        Returns:
            Re-ranked list of SearchResult objects
        """
        if not results:
            return results
        
        # Build batch prompt
        docs_text = "\n\n".join([
            f"[Doc {i+1}]: {r.content[:500]}..."
            for i, r in enumerate(results)
        ])
        
        batch_prompt = f"""You are a relevance scoring assistant. Given a query and multiple document chunks, rate each chunk's relevance to answering the query.

Query: {query}

Documents:
{docs_text}

For each document, provide a relevance score from 0.0 to 1.0.
Respond with ONLY a comma-separated list of {len(results)} decimal numbers, one for each document in order.
Example for 3 documents: 0.8, 0.3, 0.9"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": batch_prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            
            scores_text = response.choices[0].message.content.strip()
            scores = [float(s.strip()) for s in scores_text.split(",")]
            
            # Apply scores
            for i, result in enumerate(results):
                if i < len(scores):
                    llm_score = max(0.0, min(1.0, scores[i]))
                    combined_score = 0.3 * result.score + 0.7 * llm_score
                    result.metadata["llm_score"] = llm_score
                    result.metadata["original_score"] = result.score
                    result.score = combined_score
        
        except Exception:
            # Fallback: keep original scores
            pass
        
        # Sort by new score
        results.sort(key=lambda x: x.score, reverse=True)
        
        if top_n:
            results = results[:top_n]
        
        return results


def rerank_results(
    query: str,
    results: List[SearchResult],
    model: str = "gpt-4o-mini",
    top_n: Optional[int] = None,
    batch: bool = True,
) -> List[SearchResult]:
    """
    Convenience function to re-rank search results.
    
    Args:
        query: Search query
        results: Initial search results
        model: LLM model for re-ranking
        top_n: Return only top N results
        batch: Use batched scoring (more efficient)
    
    Returns:
        Re-ranked results
    
    Example:
        >>> results = pegasus.search("authentication", k=20)
        >>> reranked = rerank_results("How to set up OAuth2?", results, top_n=5)
    """
    reranker = LLMReranker(model=model)
    if batch:
        return reranker.rerank_batch(query, results, top_n=top_n)
    return reranker.rerank(query, results, top_n=top_n)
