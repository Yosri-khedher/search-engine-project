from indexer import STOP_WORDS, normalize_text, simple_stem


BOOLEAN_OPERATORS = {"AND", "OR", "NOT"}


def preprocess_boolean_term(term):
    cleaned_term = normalize_text(term).strip()
    if not cleaned_term:
        return ""

    token = cleaned_term.split()[0]
    if token in STOP_WORDS and token.upper() not in BOOLEAN_OPERATORS:
        return ""

    return simple_stem(token)


def documents_for_term(term, inverted_index):
    processed_term = preprocess_boolean_term(term)
    if not processed_term:
        return set()
    return set(inverted_index.get(processed_term, {}).keys())


def boolean_search(query, inverted_index, all_documents):
    tokens = query.split()
    if not tokens:
        return []

    upper_tokens = [token.upper() if token.upper() in BOOLEAN_OPERATORS else token for token in tokens]

    if len(upper_tokens) == 1:
        return sorted(documents_for_term(upper_tokens[0], inverted_index))

    if upper_tokens[0] == "NOT" and len(upper_tokens) >= 2:
        current_result = set(all_documents) - documents_for_term(upper_tokens[1], inverted_index)
        position = 2
    else:
        current_result = documents_for_term(upper_tokens[0], inverted_index)
        position = 1

    while position < len(upper_tokens):
        operator = upper_tokens[position]

        if operator not in BOOLEAN_OPERATORS:
            position += 1
            continue

        if position + 1 >= len(upper_tokens):
            break

        next_documents = documents_for_term(upper_tokens[position + 1], inverted_index)

        if operator == "AND":
            current_result = current_result & next_documents
        elif operator == "OR":
            current_result = current_result | next_documents
        elif operator == "NOT":
            current_result = current_result - next_documents

        position += 2

    return sorted(current_result)
