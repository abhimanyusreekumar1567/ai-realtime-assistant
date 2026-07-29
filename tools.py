from duckduckgo_search import DDGS
from typing import List, Dict

def search_internet(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using DuckDuckGo and return a list of result dicts.
    Each dict: {"title": ..., "snippet": ..., "link": ...}
    """
    with DDGS() as ddgs:
        results = []
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r["title"],
                "snippet": r["body"],
                "link": r["href"]
            })
        return results