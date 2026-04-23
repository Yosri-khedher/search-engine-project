import math

from indexer import preprocess_query


def compute_document_frequency(inverted_index):
    document_frequency = {}
    for term, postings in inverted_index.items():
        document_frequency[term] = len(postings)
    return document_frequency


def compute_idf(inverted_index, total_documents):
    idf_values = {}
    document_frequency = compute_document_frequency(inverted_index)

    for term, frequency in document_frequency.items():
        if frequency > 0:
            idf_values[term] = math.log(total_documents / frequency)
        else:
            idf_values[term] = 0.0

    return idf_values


def compute_tfidf_vector(term_frequencies, idf_values):
    vector = {}
    for term, frequency in term_frequencies.items():
        vector[term] = frequency * idf_values.get(term, 0.0)
    return vector


def vector_norm(vector):
    total = 0.0
    for value in vector.values():
        total += value * value
    return math.sqrt(total)


def cosine_similarity(vector_a, vector_b):
    dot_product = 0.0
    for term, value in vector_a.items():
        dot_product += value * vector_b.get(term, 0.0)

    norm_a = vector_norm(vector_a)
    norm_b = vector_norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def rank_documents(query, inverted_index, document_term_frequencies):
    if not document_term_frequencies:
        return []

    total_documents = len(document_term_frequencies)
    idf_values = compute_idf(inverted_index, total_documents)
    query_frequencies = preprocess_query(query)
    query_vector = compute_tfidf_vector(query_frequencies, idf_values)

    ranked_documents = []

    for document_name, term_frequencies in document_term_frequencies.items():
        document_vector = compute_tfidf_vector(term_frequencies, idf_values)
        similarity = cosine_similarity(query_vector, document_vector)

        if similarity > 0:
            ranked_documents.append((document_name, similarity))

    ranked_documents.sort(key=lambda item: item[1], reverse=True)
    return ranked_documents
