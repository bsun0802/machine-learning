import json
import argparse
import numpy as np
from utils import make_parameter_estimate_data, make_learning_data, to_list
from typing import List, Tuple
Matrix = List[List[float]]


class Hmm:
    """Hidden Markov Model, A stochastic process over state set S, parametrized by theta
        - pi: initial distribution
        - A: Trasition Probability, A[i, j] = P(Z_t+1 = j | Z_t = i)
        - B: Emission Probability, B[i, k] = P(O_t = k | Z_t = i)

        If the model parameters are specified, we could calculate
        - Probability of observing a sequence: P(O | theta). (Forward, Backward algorithm)
        - The most likely hidden path under an observation sequence O: argmax P(Z | theta, O),
                                    (Viterbi algorithm)

        Or given state set S, pi, and observed sequence O, train the parameter A and B.
                (Forward-Backward algorithm with E-M). *Not implemented here.*

        Encoding: to utilize numpy.ndarray, states and emissions are encoded as 0, 1, 2, ...
        All recusion is vectorized as ndarray multiplication, so, fast.
    """

    def __init__(self, model_file):
        """Load HMM model from json format"""
        with open(model_file, 'r') as f:
            data = json.load(f)
            if data.get("A"):
                self.A = np.array(data["A"])
            if data.get("B"):
                self.B = np.array(data["B"])
            if data.get("pi"):
                self.pi = np.array(data["pi"])
            self.states = data["states"]
            self.obs_encode = data["observations"]
            self.obs_set = sorted(self.obs_encode, key=self.obs_encode.get)

    def _encode(self, obs_seq) -> list:
        return [self.obs_encode[i] for i in obs_seq]

    def forward(self, seq) -> np.ndarray:
        """Return forward probabilities alpha.
                alpha[i, t] = P(Z_t = i, O_1..O_t | theta)"""
        obs = self._encode(seq)
        S = len(self.pi)
        N = len(seq)
        alpha = np.zeros((S, N))

        alpha[:, 0] = self.pi * self.B[:, obs[0]]
        for t in range(1, N):
            alpha[:, t] = (np.sum(alpha[:, t - 1].reshape(-1, 1) * self.A, axis=0)
                           * self.B[:, obs[t]])
        return alpha

    def backward(self, seq) -> np.ndarray:
        """Return backward probabilities beta.
                beta[i, t] = P(O_t+1,.., O_n|Z_t = i, theta)"""
        obs = self._encode(seq)

        S = len(self.pi)
        N = len(seq)
        beta = np.zeros((S, N))

        beta[:, -1] = 1
        for t in reversed(range(0, N - 1)):
            beta[:, t] = np.dot(self.A * beta[:, t + 1], self.B[:, obs[t + 1]])

        return beta

    def seq_prob(self, seq):
        """probability of observing the whole sequence, i.e., P(O|theta)"""
        return np.sum(self.forward(seq)[:, -1])

    def viterbi(self, seq):
        """return the most likely hidden path.
                delta[i, t]: the likelihood of the most likely 1..t path that ends with state i.
                psi[i, t - 1]: which state in t-1 that gives delta[i, t]
        """
        obs = self._encode(seq)
        S = len(self.pi)
        N = len(seq)
        delta = np.zeros((S, N))
        path = []

        # init
        delta[:, 0] = self.pi * self.B[:, obs[0]]
        psi = np.zeros((S, N - 1), dtype=int)

        # recursion
        for t in range(1, N):
            pool = delta[:, t - 1].reshape(-1, 1) * self.A * self.B[:, obs[t]]
            delta[:, t] = np.max(pool, axis=0)
            psi[:, t - 1] = np.argmax(pool, axis=0)

        # termination
        bt = N - 1
        state = np.argmax(delta[:, bt])
        path.append(state)

        # backtrack
        while bt > 0:
            bt -= 1
            path.append(psi[state, bt])
            state = psi[state, bt]

        return "".join([self.states[i] for i in reversed(path)])

    def params_estimate(self, Z, seq) -> Tuple[Matrix, Matrix]:
        """Estimate A and B given hidden path Z and observed sequence.
        Instead of pseudo-count, the non-observed ones will be imputed as uniform"""
        def overlap_substr_count(s, t):
            n = 0
            for i in range(0, len(s) - len(t) + 1):
                if s[i:i + len(t)] == t:
                    n += 1
            return n

        def B_ik(i, k):
            n = 0
            for idx, s in enumerate(Z):
                if s == i and seq[idx] == k:
                    n += 1
            return n / Z.count(i)
        A = []
        B = []
        for si in self.states:
            if Z[:-2].count(si) == 0:
                A.append([1 / len(self.states)] * len(self.states))
            else:
                A.append([overlap_substr_count(Z, si + sj) / Z[:-1].count(si)
                          for sj in self.states])
        for si in self.states:
            if Z.count(si) == 0:
                B.append([1 / len(self.obs_set)] * len(self.obs_set))
            else:
                B.append([B_ik(si, k) for k in self.obs_set])

        return A, B

    def train(self, seq, A0="uniform", B0="uniform", max_iter=25, method="bw"):
        """Baum Welch learning, learn the most likely transition A and emission B"""
        def argwhere(li, target):
            return [idx for idx, i in enumerate(li) if i == target]

        obs = self._encode(seq)
        N = len(obs)
        S = len(self.pi)
        K = len(self.obs_encode)

        self.A = np.array(A0, dtype=float)
        self.B = np.array(B0, dtype=float)
        if A0 == "uniform":
            self.A = np.ones((S, S)) / (S ** 2)
        if B0 == "uniform":
            self.B = np.ones((S, K)) / (S * K)
        assert self.A.shape == (S, S)
        assert self.B.shape == (S, K)

        assert method in ("bw", "viterbi")
        if method == "bw":
            xsi = np.zeros((N - 1, S, S))
            for i in range(max_iter):
                # E-step
                alpha = self.forward(seq)
                beta = self.backward(seq)
                denominator = self.seq_prob(seq)
                gamma = alpha * beta / denominator
                for t in range(N - 1):
                    for i in range(S):
                        for j in range(S):
                            xsi[t, i, j] = (alpha[i, t] * self.A[i, j]
                                            * self.B[j, obs[t + 1]] * beta[j, t + 1])
                xsi = xsi / denominator
                # M-step
                A = xsi.sum(axis=0) / xsi.sum(axis=2).sum(axis=0).reshape(-1, 1)
                B = np.ones_like(self.B)
                for j in range(K):
                    B[:, j] = gamma[:, argwhere(obs, j)].sum(axis=1) / gamma.sum(axis=1)
                self.A = A
                self.B = B
        else:
            for i in range(max_iter):
                path = self.viterbi(seq)
                self.A, self.B = map(np.array, self.params_estimate(path, seq))


def main(model_file, obs_seq):
    hmm = Hmm(model_file)
    alpha = hmm.forward(obs_seq)
    beta = hmm.backward(obs_seq)
    seq_prob = hmm.seq_prob(obs_seq)
    prob2 = np.dot(hmm.pi, beta[:, 0] * hmm.B[:, hmm.obs_encode[obs_seq[0]]])
    # prob2 is path probabiliti calculated by backward algorithm
    np.testing.assert_almost_equal(seq_prob, prob2)

    print('Total log probability of observing the sequence %s is (%g, %g). from (forward, backward) algorithm.' % (
        obs_seq, np.log(seq_prob), np.log(prob2)))

    viterbi_path = hmm.viterbi(obs_seq)

    print('Viterbi best path is ')
    for j in viterbi_path:
        print(j, end=' ')


def test_output(outout_file, A, B, states, obs_set):
    with open(outout_file, "w") as fo:
        fo.write("\t".join(states) + "\n")
        for idx, s in enumerate(states):
            fo.write(s + "\t" + "\t".join(A[idx]) + "\n")
        fo.write("-" * 8 + "\n")
        fo.write("\t".join(obs_set) + "\n")
        for idx, s in enumerate(states):
            fo.write(s + "\t" + "\t".join(B[idx]) + "\n")


def test_learning(model_file, obs_seq, A0, B0, max_iter, method):
    hmm = Hmm(model_file)
    hmm.train(obs_seq, A0=A0, B0=B0, max_iter=max_iter, method=method)
    test_output(f"{method}_out.txt", to_list(hmm.A, str), to_list(hmm.B, str),
                hmm.states, hmm.obs_set)


def test_estimate_AB(model_file, Z, obs_seq):
    """Correctness verified by submitting to Rosalind, problem BA10H"""
    hmm = Hmm(model_file)
    A_est, B_est = hmm.params_estimate(Z, obs_seq)
    test_output("param_est_out.txt", to_list(A_est, str), to_list(B_est, str),
                hmm.states, hmm.obs_set)


if __name__ == "__main__":
    # Correctness verified by submitting to Rosalind, problem BA10H.
    Z, obs_seq = make_parameter_estimate_data("params_estimate.txt")
    test_estimate_AB("train_hmm.json", Z, obs_seq)

    # Correctness verified by submitting to Rosalind, problem BA10K.
    A0, B0, obs_seq, max_iter = make_learning_data("baum_welch_input.txt", "bw")
    test_learning("train_hmm.json", obs_seq, A0, B0, max_iter, "bw")

    # Correctness verified by submitting to Rosalind, problem BA10I.
    A0, B0, obs_seq, max_iter = make_learning_data("viterbi_learning_input.txt", "viterbi")
    test_learning("train_hmm.json", obs_seq, A0, B0, max_iter, "viterbi")

    print("testing completed." "\n")
    print("Example: " "\n" "python hmm.py hmm_model.json AGCGTA" "\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", help="json file specifiying HMM parameters")
    parser.add_argument("obs_seq", help="the full observation sequence")
    args = parser.parse_args()
    main(args.model_file, args.obs_seq)
