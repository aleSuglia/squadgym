import json
from collections import defaultdict

from squadgym.utils.preprocessing import preprocess_text


def squad_reader(squad_data_filename):
    with open(squad_data_filename) as in_file:
        return json.load(in_file)


def generate_env_data(squad_data):
    env_data = defaultdict(lambda: {"context": "", "questions": [], "answers": []})

    for entity_data in squad_data["data"]:
        entity_title = entity_data["title"]

        for paragraph in entity_data["paragraphs"]:
            env_data[entity_title]["context"] = preprocess_text(paragraph["context"])
            for qas in paragraph["qas"]:
                question_tokens = preprocess_text(qas["question"])
                answers_tokens = []
                for ans in qas["answers"]:
                    answer_tokens = preprocess_text(ans["text"])
                    answers_tokens.append(answer_tokens)
                env_data[entity_title]["questions"].append(question_tokens)
                env_data[entity_title]["answers"].append(answers_tokens)

    return env_data
