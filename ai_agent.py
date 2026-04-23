from collections import Counter

from indexer import preprocess_query


RELATED_KEYWORDS = {
    "ai": ["machine learning", "deep learning", "neural networks"],
    "artificial": ["intelligent systems", "machine learning", "data driven models"],
    "machine": ["learning", "classification", "prediction"],
    "learning": ["classification", "training", "neural networks"],
    "deep": ["neural networks", "computer vision", "representation learning"],
    "retrieval": ["indexing", "ranking", "search engine"],
    "search": ["query expansion", "ranking", "relevance feedback"],
    "multimedia": ["images", "audio", "video retrieval"],
    "image": ["computer vision", "feature extraction", "object recognition"],
    "vision": ["images", "deep learning", "object detection"],
    "audio": ["speech processing", "signal analysis", "sound classification"],
    "boolean": ["and", "or", "not"],
    "index": ["inverted index", "term frequency", "document frequency"],
    "indexation": ["inverted index", "retrieval", "documents"],
}


class SearchAIAgent:
    def __init__(self, document_term_frequencies):
        self.document_term_frequencies = document_term_frequencies

    def suggest_related_keywords(self, query):
        query_terms = preprocess_query(query)
        suggestions = []

        for term in query_terms:
            suggestions.extend(RELATED_KEYWORDS.get(term, []))

        if not suggestions:
            suggestions = self._corpus_based_suggestions(query_terms)

        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in unique_suggestions:
                unique_suggestions.append(suggestion)

        return unique_suggestions[:5]

    def _corpus_based_suggestions(self, query_terms):
        counter = Counter()

        for frequencies in self.document_term_frequencies.values():
            overlap = False
            for term in query_terms:
                if term in frequencies:
                    overlap = True
                    break

            if overlap:
                for word, frequency in frequencies.items():
                    if word not in query_terms:
                        counter[word] += frequency

        suggestions = []
        for word, _ in counter.most_common(5):
            suggestions.append(word)
        return suggestions

    def improve_query(self, query):
        suggestions = self.suggest_related_keywords(query)
        if not suggestions:
            return query

        improved_terms = [query]
        for suggestion in suggestions[:2]:
            improved_terms.append(suggestion)

        return " | ".join(improved_terms)

    def explain_results(self, query, results, model_name):
        if not results:
            return (
                f"No document matched the query '{query}'. "
                "Try a more specific keyword or use the AI suggestions."
            )

        top_document = results[0][0] if isinstance(results[0], tuple) else results[0]
        return (
            f"The {model_name} model found {len(results)} relevant document(s). "
            f"The top result is {top_document} because it shares important terms with the query."
        )
