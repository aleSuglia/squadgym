import pickle

import gym
import numpy
from gym import spaces
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from squadgym.utils.preprocessing import tokens2ids


def prepare_env_data(env_data_json):
    env_data = {}
    token2id = {}
    id2token = {}

    def build_ids(tokens):
        for token in tokens:
            if token not in token2id:
                num_tokens = len(token2id) + 1
                token2id[token] = num_tokens
                id2token[num_tokens] = token

    build_ids(["#pad#", "#eos#", "#sos#"])

    for entity_title, entity_data in env_data_json.items():
        entity_questions = []
        entity_answers = []
        entity_context = entity_data["context"]
        build_ids(entity_context)
        entity_context = tokens2ids(entity_context, token2id)

        for i in range(len(entity_data["questions"])):
            question_tokens = entity_data["questions"][i]
            build_ids(entity_data["questions"][i])
            question_ids = tokens2ids(question_tokens, token2id)
            answers_ids = []

            for answer_tokens in entity_data["answers"][i]:
                build_ids(answer_tokens)
                answers_ids.append(tokens2ids(answer_tokens, token2id))

            entity_questions.append(question_ids)
            entity_answers.append(answers_ids)

        env_data[entity_title] = {
            "context": entity_context,
            "questions": entity_questions,
            "answers": entity_answers
        }

    return {
        "env_data": env_data,
        "token2id": token2id,
        "id2token": id2token
    }


def compute_sequence_reward(predicted_sequence, target_sequences):
    if not predicted_sequence:
        return 0
    smoothing_function = SmoothingFunction().method2
    return sentence_bleu(
        target_sequences,
        predicted_sequence,
        weights=[1.0],
        smoothing_function=smoothing_function
    )


class SquadEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, env_data_filename, mode="single", max_utterance_len=20, max_game_turns=20):
        with open(env_data_filename, mode="rb") as in_file:
            self._env_data = pickle.load(in_file)
        self._mode = mode
        self._max_game_turns = max_game_turns
        self._num_tokens = len(self._env_data["id2token"])
        self._num_entities = len(self._env_data["env_data"])
        self._entities = list(self._env_data["env_data"].keys())
        self.action_space = spaces.Discrete(self._num_tokens)
        self.observation_space = spaces.MultiDiscrete([[0, self._num_tokens] * max_utterance_len])
        self._last_entity = 0
        self._last_question = 0
        self._last_sequence = []
        self._game_turns = None
        self._game_score = 0

    def reset(self):
        # reset last generated sequence
        self._last_sequence.clear()
        self._game_turns = numpy.arange(self._num_entities)
        numpy.random.shuffle(self._game_turns)
        self._game_turns = self._game_turns[:self._max_game_turns]
        self._game_score = 0

        # retrieve a random entity and use its questions
        self._last_entity = 0
        self._last_question = 0

        entity_data = self._env_data["env_data"][self._entities[self._last_entity]]
        # return current question
        return entity_data["context"], entity_data["questions"][self._last_question]

    def step(self, action):
        # generated end-of-sequence token
        if action == self._env_data["token2id"]["#eos#"]:
            entity_data = self._env_data["env_data"][self._entities[self._last_entity]]
            self._last_sequence.append(action)
            reward = compute_sequence_reward(
                self._last_sequence,
                entity_data["answers"][self._last_entity]
            )
            self._game_score += reward

            if self._last_entity >= self._max_game_turns:
                return (None, None), self._game_score, True, None

            self._last_entity += 1
            entity_index = self._game_turns[self._last_entity]
            entity_data = self._env_data["env_data"][self._entities[entity_index]]
            self._last_question = numpy.random.randint(0, len(entity_data["questions"]))

            # return current question
            observation = (entity_data["context"], entity_data["questions"][self._last_question])
            return observation, reward, False, None

        return (None, None), 0, False, None

    @property
    def id2token(self):
        return self._env_data["id2token"]

    @property
    def token2id(self):
        return self._env_data["token2id"]
