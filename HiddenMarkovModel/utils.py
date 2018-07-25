import json
import os


def mat_to_list(prob_mat, dtype):
    if isinstance(prob_mat, list):
        return [list(map(dtype, row)) for row in prob_mat]
    return prob_mat.astype(dtype).tolist()


def line(f):
    return f.readline().rstrip()


def br(f):
    f.readline()


def make_model_json(fn, **kwargs):
    with open(fn, "w") as f:
        json.dump(kwargs, f)


def read_known_model(input) -> str:
    with open(input, "r") as f:
        obs_seq = line(f)
        br(f)
        obs_set = line(f).split()
        observations = {i: idx for idx, i in enumerate(obs_set)}
        br(f)
        states = line(f).split()
        br(f)
        br(f)
        A = []
        B = []
        for i in range(len(states)):
            A.append([float(i) for i in line(f).split()[1:]])
        br(f)
        br(f)
        for i in range(len(states)):
            B.append([float(i) for i in line(f).split()[1:]])
        pi = [round(1 / len(states), 4)] * len(states)

    make_model_json("hmm_model_sd.json", states=states, pi=pi, observations=observations,
                    A=A, B=B)

    return obs_seq


def make_learning_data(input, method="bw"):
    A0 = []
    B0 = []
    with open(input, "r") as f:
        max_iter = int(line(f))
        br(f)
        obs_seq = line(f)
        br(f)
        obs_set = line(f).split()
        observations = {i: idx for idx, i in enumerate(obs_set)}
        br(f)
        states = line(f).split()
        br(f)
        br(f)
        for i in range(len(states)):
            A0.append(line(f).split()[1:])
        br(f)
        br(f)
        for i in range(len(states)):
            B0.append(line(f).split()[1:])
    A0 = mat_to_list(A0, float)
    B0 = mat_to_list(B0, float)
    pi = [round(1 / len(states), 4)] * len(states)
    if method == "bw":
        make_model_json("train_hmm.json", states=states, pi=pi, observations=observations)
    else:
        make_model_json("train_hmm.json", states=states, pi=pi, observations=observations,
                        A=A0, B=B0)
    return A0, B0, obs_seq, max_iter


def make_parameter_estimate_data(input):
    with open(input, "r") as f:
        obs_seq = f.readline().rstrip()
        f.readline()
        obs_set = f.readline().rstrip().split()
        observations = {i: idx for idx, i in enumerate(obs_set)}
        f.readline()
        Z = f.readline().rstrip()
        f.readline()
        states = f.readline().rstrip().split()
    make_model_json("train_hmm.json", states=states, observations=observations)
    return Z, obs_seq
