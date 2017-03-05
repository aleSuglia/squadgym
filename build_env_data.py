import argparse

import pickle

from squadgym.data.squad import generate_env_data, squad_reader
from squadgym.envs.squadenv import prepare_env_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("squad_data", type=str, help="SQuAD dataset filename in JSON format")
    parser.add_argument("env_data", type=str, help="Generated environment data filename in JSON format")
    args = parser.parse_args()
    squad_data_filename = args.squad_data
    env_data_filename = args.env_data
    squad_data = squad_reader(squad_data_filename)
    print("-- Reading SQuAD dataset from file: {}".format(squad_data_filename))
    env_data = generate_env_data(squad_data)
    print("-- Converting SQuAD dataset")
    env_data = prepare_env_data(env_data)
    print("-- Saving environment data in file: {}".format(env_data_filename))
    with open(env_data_filename, mode="wb") as out_file:
        pickle.dump(env_data, out_file)


if __name__ == "__main__":
    main()
