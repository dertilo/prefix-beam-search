from collections import defaultdict, Counter
from string import ascii_lowercase
import re
import numpy as np


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
    F = ctc.shape[1]
    # just add an imaginative zero'th step (will make indexing more intuitive)
    ctc = np.vstack((np.zeros(F), ctc))
    T = ctc.shape[0]

    # STEP 1: Initiliazation
    O = ""
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][O] = 1
    Pnb[0][O] = 0
    prefixes = [O]
    # END: STEP 1

    # STEP 2: Iterations and pruning
    for t in range(1, T):
        pruned_alphabet = [alphabet[i] for i in np.where(ctc[t] > prune)[0]]
        for pref in prefixes:

            is_phrase_end = len(pref) > 0 and pref[-1] == ">"
            if is_phrase_end:
                Pb[t][pref] = Pb[t - 1][pref]
                Pnb[t][pref] = Pnb[t - 1][pref]
                continue  # goto next prefix

            for c in pruned_alphabet:
                c_ix = alphabet.index(c)
                # END: STEP 2

                # STEP 3: “Extending” with a blank
                blank_or_not_prob = Pb[t - 1][pref] + Pnb[t - 1][pref]
                if c == BLANK:
                    Pb[t][pref] += ctc[t][BLANK_IDX] * (blank_or_not_prob)
                # END: STEP 3

                # STEP 4: Extending with the end character
                else:
                    pref_ext = pref + c  # extended prefix
                    is_same_as_before = len(pref) > 0 and c == pref[-1]
                    character_prob = ctc[t][c_ix]
                    token_is_ending = len(pref.replace(" ", "")) > 0 and c in (" ", ">")
                    if is_same_as_before:
                        Pnb[t][pref_ext] += character_prob * Pb[t - 1][pref]
                        Pnb[t][pref] += character_prob * Pnb[t - 1][pref]
                    # END: STEP 4

                    # STEP 5: Extending with any other non-blank character and LM constraints
                    elif token_is_ending:
                        lm_prob = lm(pref_ext.strip(" >")) ** alpha
                        Pnb[t][pref_ext] += (
                            lm_prob * character_prob * (blank_or_not_prob)
                        )
                    else:  # within token
                        Pnb[t][pref_ext] += character_prob * (blank_or_not_prob)
                    # END: STEP 5

                    # STEP 6: Make use of discarded prefixes
                    if pref_ext not in prefixes:
                        blank_or_not_pref_ext = (
                            Pb[t - 1][pref_ext] + Pnb[t - 1][pref_ext]
                        )
                        Pb[t][pref_ext] += ctc[t][BLANK_IDX] * blank_or_not_pref_ext
                        Pnb[t][pref_ext] += character_prob * Pnb[t - 1][pref_ext]
                    # END: STEP 6

        # STEP 7: Select most probable prefixes
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
        prefixes = sorted(A_next, key=sorter, reverse=True)[:k]
        # END: STEP 7

    return prefixes[0].strip(">")
