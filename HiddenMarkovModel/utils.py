import json


def make_model_json(input, known_params=False):
    """The input is from Rosalind problem BA10K,
        See more information here: http://rosalind.info/problems/ba10k/"""
    def br():
        f.readline()

    def line(f):
        return f.readline().rstrip()
    A = []
    B = []
    with open(input, "r") as f:
        br()
        br()
        max_iter = int(line(f))
        br()
        obs_seq = line(f)
        br()
        observations = line(f).split()
        observations = {i: idx for idx, i in enumerate(observations)}
        br()
        states = line(f).split()
        br()
        br()
        for i in range(len(states)):
            A.append(line(f).split()[1:])
        br()
        br()
        for i in range(len(states)):
            B.append(line(f).split()[1:])
    pi = [round(1 / len(states), 4)] * len(states)
    data = {"states": states, "pi": pi, "observations": observations}
    if known_params:  # treat A and B as known matrices
        data["A"] = A
        data["B"] = B
        with open("hmm_model.json", "w") as f:
            json.dump(data, f)
    else:  # treat A and B as initial matrices in Baum-Welch learning.
        with open("train_hmm.json", "w") as f:
            json.dump(data, f)
    return A, B, obs_seq, max_iter


def make_parameter_estimate_data(input):
    with open(input, "r") as f:
        obs_seq = f.readline().rstrip()
        f.readline()
        obs_set = f.readline().rstrip().split()
        f.readline()
        pseudo_Z = f.readline().rstrip()
        f.readline()
        states = f.readline().rstrip().split()
    return pseudo_Z, states, obs_seq, obs_set


def params_estimate(pseudo_Z, states, obs_seq, obs_set):
    def overlap_substr_count(s, t):
        """count number of occurrance of t in s.
           python str.count() count non-overlap."""
        n = 0
        for i in range(0, len(s) - len(t) + 1):
            if s[i:i + len(t)] == t:
                n += 1
        return n
    """Instead of pseudo-count, the non-observed ones are imputed by uniform"""
    def B_ik(i, k):
        n = 0
        for idx, s in enumerate(pseudo_Z):
            if s == i and obs_seq[idx] == k:
                n += 1
        return n / pseudo_Z.count(i)
    A = []
    B = []
    for si in states:
        if pseudo_Z[:-2].count(si) == 0:
            A.append([1 / len(states)] * len(states))
        else:
            A.append([overlap_substr_count(pseudo_Z, si + sj) / pseudo_Z[:-1].count(si)
                      for sj in states])
            # A.append([pseudo_Z.count(si + sj) / pseudo_Z.count(si) for sj in states])
    for si in states:
        if pseudo_Z.count(si) == 0:
            B.append([1 / len(obs_set)] * len(obs_set))
        else:
            B.append([B_ik(si, k) for k in obs_set])
    return A, B
