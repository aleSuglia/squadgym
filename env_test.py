import argparse

from nltk import word_tokenize

from squadgym.envs import SquadEnv
from squadgym.utils.preprocessing import ids2tokens, tokens2ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_data", type=str, help="Generated environment data filename in JSON format")
    args = parser.parse_args()

    print("-- Initialized environment")
    env = SquadEnv(args.env_data)

    context, question = env.reset()
    done = False

    while not done:
        print("Context ids: {}".format(context))
        print("Question ids: {}".format(question))
        print("Context tokens: {}".format(ids2tokens(context, env.id2token)))
        print("Question tokens: {}".format(ids2tokens(question, env.id2token)))
        answer_tokens = tokens2ids(word_tokenize(input("Answer: ")) + ["#eos#"], env.token2id)

        question_reward = 0
        for token in answer_tokens:
            (context, question), reward, done, _ = env.step(token)
            question_reward += reward

        print("You got {} reward".format(question_reward))


if __name__ == "__main__":
    main()
