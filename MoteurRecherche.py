import os

from ai_agent import SearchAIAgent
from boolean_model import boolean_search
from evaluation import SAMPLE_RELEVANCE, evaluate_query_results
from indexer import build_and_save_index, read_documents
from vector_model import rank_documents


class SearchEngine:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.documents_dir = os.path.join(self.base_dir, "documents")
        self.images_dir = os.path.join(self.base_dir, "images")
        self.index_path = os.path.join(self.base_dir, "index.json")

        self.documents = read_documents(self.documents_dir)
        self.inverted_index, self.document_term_frequencies = build_and_save_index(
            self.documents_dir,
            self.index_path,
        )

        self.ai_agent = SearchAIAgent(self.document_term_frequencies)
        self.image_keywords = {
            "machine": ["machine_learning.png"],
            "learning": ["machine_learning.png", "neural_networks.png"],
            "deep": ["neural_networks.png"],
            "neural": ["neural_networks.png"],
            "vision": ["computer_vision.png"],
            "image": ["computer_vision.png"],
            "audio": ["audio_signal.png"],
            "sound": ["audio_signal.png"],
            "retrieval": ["information_retrieval.png"],
            "search": ["information_retrieval.png"],
            "index": ["information_retrieval.png"],
            "multimedia": ["multimedia_gallery.png"],
        }

    def rebuild_index(self):
        self.documents = read_documents(self.documents_dir)
        self.inverted_index, self.document_term_frequencies = build_and_save_index(
            self.documents_dir,
            self.index_path,
        )
        self.ai_agent = SearchAIAgent(self.document_term_frequencies)

    def search_vector(self, query):
        return rank_documents(query, self.inverted_index, self.document_term_frequencies)

    def search_boolean(self, query):
        all_documents = list(self.documents.keys())
        matches = boolean_search(query, self.inverted_index, all_documents)
        return [(document_name, 1.0) for document_name in matches]

    def search(self, query, model_name):
        if model_name.lower() == "boolean":
            return self.search_boolean(query)
        return self.search_vector(query)

    def get_document_content(self, document_name):
        return self.documents.get(document_name, "")

    def get_related_images(self, query):
        related_images = []
        query_terms = list(self.ai_agent.suggest_related_keywords(query))
        query_terms.extend(query.lower().split())

        for term in query_terms:
            for image_name in self.image_keywords.get(term.lower(), []):
                image_path = os.path.join(self.images_dir, image_name)
                if image_path not in related_images and os.path.exists(image_path):
                    related_images.append(image_path)

        return related_images[:3]

    def get_ai_response(self, query, results, model_name):
        return {
            "suggestions": self.ai_agent.suggest_related_keywords(query),
            "improved_query": self.ai_agent.improve_query(query),
            "explanation": self.ai_agent.explain_results(query, results, model_name),
        }

    def evaluate_results(self, query, results):
        return evaluate_query_results(query, results)

    def has_evaluation_reference(self, query):
        return query.strip().lower() in SAMPLE_RELEVANCE


def run_terminal_search():
    engine = SearchEngine()

    print("Simple Multimedia Search Engine")
    query = input("Enter query: ").strip()

    print("\nVector model results:")
    vector_results = engine.search_vector(query)
    for document_name, score in vector_results[:10]:
        print(f"{document_name} -> {score:.3f}")

    print("\nBoolean model results:")
    boolean_results = engine.search_boolean(query)
    for document_name, score in boolean_results[:10]:
        print(f"{document_name} -> {score:.1f}")

    ai_response = engine.get_ai_response(query, vector_results, "Vector")
    print("\nAI suggestions:")
    for suggestion in ai_response["suggestions"]:
        print(f"- {suggestion}")


if __name__ == "__main__":
    from gui import launch_gui

    launch_gui()
