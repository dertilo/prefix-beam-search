import pickle
from collections import defaultdict, Counter
from string import ascii_lowercase
import re
from typing import NamedTuple, Dict, List

import numpy as np


class LanguageModel(object):
    """
    Loads a dictionary mapping between prefixes and probabilities.
    """

    def __init__(self, lm_file):
        """
        Initializes the language model.

        Args:
          lm_file (str): Path to dictionary mapping between prefixes and lm probabilities.
        """
        lm = pickle.load(open(lm_file, "rb"))
        self._model = defaultdict(lambda: 1e-11, lm)

    def __call__(self, prefix):
        """
        Returns the probability of the last word conditioned on all previous ones.

        Args:
          prefix (str): The sentence prefix to be scored.
        """
        return self._model[prefix]


class Search(NamedTuple):
    alpha: float
    beta: float
    BLANK: str
    BLANK_IDX: int
    lm: LanguageModel
    alphabet: List[str]
    prune: float


def prefix_beam_search(ctc, lm=None, k=25, alpha=0.30, beta=5, prune=0.001):
    """
    Performs prefix beam search on the output of a CTC network.

    Args:
            ctc (np.ndarray): The CTC output. Should be a 2D array (timesteps x alphabet_size)
            lm (func): Language model function. Should take as input a string and output a probability.
            k (int): The beam width. Will keep the 'k' most likely candidates at each timestep.
            alpha (float): The language model weight. Should usually be between 0 and 1.
            beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
            prune (float): Only extend prefixes with chars with an emission probability higher than 'prune'.

    Retruns:
            string: The decoded CTC output.
    """

    # if no LM is provided, just set to function returning 1
    lm = (lambda l: 1) if lm is None else lm
    W = lambda l: re.findall(r"\w+[\s|>]", l)
    BLANK = "%"
    alphabet = list(ascii_lowercase) + [" ", ">", BLANK]
    BLANK_IDX = alphabet.index(BLANK)

    search = Search(alpha, beta, BLANK, BLANK_IDX, lm, alphabet, prune)

    O = ""
    pb_t, pNb_t = Counter(), Counter()
    pb_t[O] = 1.0
    pNb_t[O] = 0.0
    prefixes = [O]

    for character_probs in ctc:
        pNb_t, pb_t = step(search, character_probs, pNb_t, pb_t, prefixes)

        A_next = pb_t + pNb_t
        sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** search.beta
        prefixes = sorted(A_next, key=sorter, reverse=True)[:k]

    return prefixes[0].strip(">")


def step(search: Search, character_probs, pNb_b, pb_b, prefixes):
    pruned_alphabet = [
        search.alphabet[i] for i in np.where(character_probs > search.prune)[0]
    ]

    pb_t, pNb_t = Counter(), Counter()
    for pref in prefixes:

        is_phrase_end = len(pref) > 0 and pref[-1] == ">"
        if is_phrase_end:
            pb_t[pref] = pb_b[pref]
            pNb_t[pref] = pNb_b[pref]
            continue  # goto next prefix

        for c in pruned_alphabet:
            c_ix = search.alphabet.index(c)
            # END: STEP 2

            # STEP 3: “Extending” with a blank
            blank_or_not_prob = pb_b[pref] + pNb_b[pref]
            if c == search.BLANK:
                pb_t[pref] += character_probs[search.BLANK_IDX] * blank_or_not_prob
            # END: STEP 3

            # STEP 4: Extending with the end character
            else:
                pref_ext = pref + c  # extended prefix
                is_same_as_before = len(pref) > 0 and c == pref[-1]
                character_prob = character_probs[c_ix]
                token_is_ending = len(pref.replace(" ", "")) > 0 and c in (" ", ">")
                if is_same_as_before:
                    pNb_t[pref_ext] += character_prob * pb_b[pref]
                    pNb_t[pref] += character_prob * pNb_b[pref]
                # END: STEP 4

                # STEP 5: Extending with any other non-blank character and LM constraints
                elif token_is_ending:
                    lm_prob = search.lm(pref_ext.strip(" >")) ** search.alpha
                    pNb_t[pref_ext] += lm_prob * character_prob * blank_or_not_prob
                else:  # within token
                    pNb_t[pref_ext] += character_prob * blank_or_not_prob
                # END: STEP 5

                # STEP 6: Make use of discarded prefixes
                if pref_ext not in prefixes:
                    blank_or_not_pref_ext = pb_b[pref_ext] + pNb_b[pref_ext]
                    pb_t[pref_ext] += (
                        character_probs[search.BLANK_IDX] * blank_or_not_pref_ext
                    )
                    pNb_t[pref_ext] += character_prob * pNb_b[pref_ext]
                # END: STEP 6
    return pNb_t, pb_t
