import os
import sys
from dataclasses import dataclass
from omegaconf import OmegaConf
import h5py
from tqdm import tqdm

import traceback, pdb, sys
import numpy as np


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type != KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook


def softmax(x: np.ndarray, axis: int = -1):
    exponentiated = np.exp(x)
    summed = np.sum(exponentiated, axis=axis, keepdims=True)
    return exponentiated / summed


def sticky_viterbi(P: np.ndarray, alpha: float):
    """
    A version of the Viterbi algorithm that discourages switching states.

    In order to encourage self-transitions, we set the transition probabilities to be
    uniform across all states, except for self-transitions, which are scaled by `alpha`
    relative to all other transitions. For example, if we have three states and alpha =
    2.0, the transition probability matrix would be:

    \\begin{bmatrix}
    0.5 & 0.25 & 0.25 \\\\
    0.25 & 0.5 & 0.25 \\\\
    0.25 & 0.25 & 0.5
    \\end{bmatrix}
    
    Args:
        P: an array of probabilities, shape [sequence length, state probabilities].
            For example, the probability of a key at each time step.
        alpha: a parameter that controls how self-transitions are weighted.

    If alpha = 1.0, then the output is equivalent to argmax:
    >>> P = np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])
    >>> sticky_viterbi(P, alpha=1.0)
    [0, 1, 0]
    >>> P = np.array([[0.9, 0.1], [0.49, 0.51], [0.51, 0.49], [0.49, 0.51]])
    >>> sticky_viterbi(P, alpha=1.0)
    [0, 1, 0, 1]

    If alpha > 1.0, then switching is disfavored:
    >>> P = np.array([[0.1, 0.9], [0.51, 0.49], [0.51, 0.49], [0.51, 0.49]])
    >>> sticky_viterbi(P, alpha=1.2)
    [1, 1, 1, 1]

    >>> P = np.array([[0.51, 0.49], [0.51, 0.49], [0.51, 0.49], [0.1, 0.9]])
    >>> sticky_viterbi(P, alpha=1.1)
    [0, 0, 0, 1]

    If alpha < 1.0, then switching is actually favored:
    >>> P = np.array([[0.9, 0.1], [0.51, 0.49], [0.51, 0.49], [0.51, 0.49]])
    >>> sticky_viterbi(P, alpha=0.9)
    [0, 1, 0, 1]

    """
    if alpha == 1.0:
        return np.argmax(P, axis=-1).tolist()
    seq_len, n_states = P.shape
    transition_probs = np.ones((n_states, n_states))
    transition_probs[range(n_states), range(n_states)] *= alpha
    transition_probs /= transition_probs.sum(axis=0, keepdims=True)

    # take logs for numerical stability
    transition_probs = np.log(transition_probs)
    P = np.log(P)

    log_prob_of_zero = -1e10
    scores = np.full((seq_len, n_states), fill_value=log_prob_of_zero)
    scores[0, :] = P[0, :]

    traceback = np.ones((seq_len - 1, n_states), dtype=int) * -1
    for seq_i in range(1, seq_len):
        for this_state_i in range(n_states):
            for prev_state_i in range(n_states):
                new_score = (
                    scores[seq_i - 1, prev_state_i]
                    + transition_probs[prev_state_i, this_state_i]
                    + P[seq_i, this_state_i]
                )
                if new_score > scores[seq_i, this_state_i]:
                    scores[seq_i, this_state_i] = new_score
                    traceback[seq_i - 1, this_state_i] = prev_state_i

    assert not (traceback == -1).any()

    state_i = scores[-1, :].argmax()
    out = [state_i]
    for seq_i in range(seq_len - 2, -1, -1):
        state_i = traceback[seq_i, state_i]
        out.append(state_i)
    return list(reversed(out))


@dataclass
class Config:
    logits_h5: str
    output_h5: str
    sticky_viterbi_alpha: float = 2.0
    max_preds: int | None = None


def main():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore

    os.makedirs(os.path.dirname(config.output_h5), exist_ok=True)

    with h5py.File(config.logits_h5, mode="r") as h5file:
        with h5py.File(config.output_h5, mode="w") as h5out:

            n_preds = len(h5file)
            if config.max_preds is not None:
                n_preds = min(n_preds, config.max_preds)

            for i in tqdm(range(n_preds)):
                logits: np.ndarray = (h5file[f"logits_{i}"])[:]  # type:ignore
                probs = softmax(logits)
                predictions = sticky_viterbi(probs, config.sticky_viterbi_alpha)
                h5out.create_dataset(f"predictions_{i}", data=np.array(predictions))

    print(f"Wrote {config.output_h5}")


if __name__ == "__main__":
    main()
