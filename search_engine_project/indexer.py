import json
import os
from collections import Counter


STOP_WORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "between",
    "both",
    "by",
    "can",
    "do",
    "during",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "many",
    "more",
    "most",
    "not",
    "of",
    "on",
    "one",
    "or",
    "our",
    "out",
    "over",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "under",
    "up",
    "use",
    "using",
    "used",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "within",
}

PUNCTUATION = ".,;:!?()[]{}<>/\\|@#$%^&*_+=~`\"'-"


def simple_stem(word):
    if len(word) <= 3:
        return word

    suffix_rules = [
        ("ization", ""),
        ("ational", "ate"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("iveness", "ive"),
        ("tional", "tion"),
        ("ations", "ate"),
        ("ation", "ate"),
        ("lessly", "less"),
        ("ities", "ity"),
        ("ously", "ous"),
        ("ments", "ment"),
        ("ement", ""),
        ("ment", ""),
        ("ness", ""),
        ("able", ""),
        ("ible", ""),
        ("tion", "t"),
        ("sion", "s"),
        ("ings", ""),
        ("ing", ""),
        ("edly", ""),
        ("ed", ""),
        ("ers", "er"),
        ("ies", "y"),
        ("ied", "y"),
        ("est", ""),
        ("ful", ""),
        ("ous", ""),
        ("ive", ""),
        ("ly", ""),
        ("es", ""),
        ("s", ""),
    ]

    stemmed = word
    for suffix, replacement in suffix_rules:
        if stemmed.endswith(suffix) and len(stemmed) > len(suffix) + 2:
            stemmed = stemmed[: -len(suffix)] + replacement
            break

    if stemmed.endswith("nn") or stemmed.endswith("tt") or stemmed.endswith("ll"):
        stemmed = stemmed[:-1]

    return stemmed


def normalize_text(text):
    translation_table = str.maketrans({character: " " for character in PUNCTUATION})
    return text.lower().translate(translation_table)


def preprocess_text(text):
    cleaned_text = normalize_text(text)
    tokens = cleaned_text.split()

    processed_tokens = []
    for token in tokens:
        if token.isdigit():
            continue
        if token in STOP_WORDS:
            continue

        stemmed_token = simple_stem(token)
        if len(stemmed_token) > 1 and stemmed_token not in STOP_WORDS:
            processed_tokens.append(stemmed_token)

    return dict(Counter(processed_tokens))


def preprocess_query(query):
    return preprocess_text(query)


def read_documents(documents_folder):
    documents = {}

    for file_name in sorted(os.listdir(documents_folder)):
        file_path = os.path.join(documents_folder, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                documents[file_name] = file.read()

    return documents


def build_document_term_frequencies(documents_folder):
    documents = read_documents(documents_folder)
    document_term_frequencies = {}

    for document_name, text in documents.items():
        document_term_frequencies[document_name] = preprocess_text(text)

    return document_term_frequencies


def build_inverted_index(documents_folder):
    document_term_frequencies = build_document_term_frequencies(documents_folder)
    inverted_index = {}

    for document_name, frequencies in document_term_frequencies.items():
        for word, frequency in frequencies.items():
            if word not in inverted_index:
                inverted_index[word] = {}
            inverted_index[word][document_name] = frequency

    return inverted_index, document_term_frequencies


def save_index(index, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(index, file, indent=4, ensure_ascii=False)


def load_index(index_path):
    with open(index_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_and_save_index(documents_folder, output_path):
    index, document_term_frequencies = build_inverted_index(documents_folder)
    save_index(index, output_path)
    return index, document_term_frequencies


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    documents_dir = os.path.join(base_dir, "documents")
    index_path = os.path.join(base_dir, "index.json")

    index, _ = build_and_save_index(documents_dir, index_path)
    print(f"Indexed {len(index)} terms and saved the result to {index_path}")
