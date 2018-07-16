import json
import argparse
import numpy as np


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
            self.states = data["states"]
            self.pi = np.array(data["pi"])
            self.obs_encode = data["observations"]

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

        path.reverse()
        return path

    def train(self, seq, A="uniform", B="uniform"):
        def argwhere(li, target):
            return [idx for idx, i in enumerate(li) if i == target]

        obs = self._encode(seq)
        N = len(obs)
        S = len(self.pi)
        K = len(self.obs_encode)
        if A == "uniform":
            self.A = np.ones((S, S)) / (S ** 2)
        if B == "uniform":
            self.B = np.ones((S, K)) / (S * K)
        assert self.A.shape == (S, S)
        assert self.B.shape == (S, K)
        xsi = np.zeros((N - 1, S, S))
        while True:
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
            tol = 1e-4
            if max(np.abs(self.A - A).max(), np.abs(self.B - B).max()) < tol:
                break
            self.A = A
            self.B = B


def main(model_file, obs):
    hmm = Hmm(model_file)
    # Make sure you use train_model.json when you train HMM
    # hmm.train(obs)
    # np.testing.assert_array_almost_equal(hmm.A, [[0, 1], [1, 0]])
    # np.testing.assert_array_almost_equal(hmm.B, [[1, 0], [0, 1]])
    alpha = hmm.forward(obs)
    beta = hmm.backward(obs)
    seq_prob = hmm.seq_prob(obs)
    prob2 = np.dot(hmm.pi, beta[:, 0] * hmm.B[:, hmm.obs_encode[obs[0]]])
    np.testing.assert_almost_equal(seq_prob, prob2)  # prob2 is path probabiliti calculated by backward algorithm

    print('Total log probability of observing the sequence %s is (%g, %g). from (forward, backward) algorithm.' % (
        obs, np.log(seq_prob), np.log(prob2)))

    viterbi_path = hmm.viterbi(obs)

    print('Viterbi best path is ')
    for j in viterbi_path:
        print(hmm.states[j], end=' ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", help="json file specifiying HMM parameters")
    parser.add_argument("obs_seq", help="the full observation sequence")
    args = parser.parse_args()
    main(args.model_file, args.obs_seq)
