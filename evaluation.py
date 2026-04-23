import os

project_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.getenv("TEMP", project_dir)
matplotlib_cache = os.path.join(temp_dir, "search_engine_matplotlib_cache")
os.makedirs(matplotlib_cache, exist_ok=True)
os.environ["MPLCONFIGDIR"] = matplotlib_cache

import matplotlib.pyplot as plt


SAMPLE_RELEVANCE = {
    "machine learning": {"doc1.txt", "doc2.txt", "doc10.txt", "doc14.txt", "doc20.txt"},
    "information retrieval": {"doc3.txt", "doc4.txt", "doc7.txt", "doc15.txt", "doc17.txt"},
    "computer vision": {"doc5.txt", "doc6.txt", "doc11.txt", "doc18.txt"},
    "audio processing": {"doc8.txt", "doc12.txt", "doc19.txt"},
    "boolean model": {"doc4.txt", "doc7.txt", "doc15.txt"},
    "multimedia search": {"doc6.txt", "doc8.txt", "doc13.txt", "doc17.txt", "doc19.txt"},
}


def precision(retrieved_documents, relevant_documents):
    retrieved_set = set(retrieved_documents)
    relevant_set = set(relevant_documents)

    if not retrieved_set:
        return 0.0

    relevant_found = len(retrieved_set & relevant_set)
    return relevant_found / len(retrieved_set)


def recall(retrieved_documents, relevant_documents):
    retrieved_set = set(retrieved_documents)
    relevant_set = set(relevant_documents)

    if not relevant_set:
        return 0.0

    relevant_found = len(retrieved_set & relevant_set)
    return relevant_found / len(relevant_set)


def precision_recall_points(ranked_results, relevant_documents):
    relevant_set = set(relevant_documents)
    retrieved_so_far = []
    points = []

    for item in ranked_results:
        document_name = item[0] if isinstance(item, tuple) else item
        retrieved_so_far.append(document_name)
        current_precision = precision(retrieved_so_far, relevant_set)
        current_recall = recall(retrieved_so_far, relevant_set)
        points.append((current_recall, current_precision))

    return points


def plot_precision_recall_curve(ranked_results, relevant_documents, title="Precision-Recall Curve"):
    points = precision_recall_points(ranked_results, relevant_documents)

    if not points:
        return None

    recalls = [point[0] for point in points]
    precisions = [point[1] for point in points]

    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, marker="o", color="navy")
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.tight_layout()
    plt.show()
    return points


def evaluate_query_results(query, ranked_results):
    normalized_query = query.strip().lower()
    relevant_documents = SAMPLE_RELEVANCE.get(normalized_query, set())
    retrieved_documents = [item[0] if isinstance(item, tuple) else item for item in ranked_results]

    return {
        "relevant_documents": relevant_documents,
        "precision": precision(retrieved_documents, relevant_documents),
        "recall": recall(retrieved_documents, relevant_documents),
    }
