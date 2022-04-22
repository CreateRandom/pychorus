import os
import warnings

import librosa
from scipy.signal import convolve
import numpy as np


def find_snippet_in_search_range(raw_snippet, raw_search, sr):
    corr_vec = convolve(raw_search, np.flipud(raw_snippet), 'valid')
    # todo this threshold is currently sr- dependent...
    if max(corr_vec) < 1000:
        return None
    max_ind = corr_vec.argmax()
    # todo find out why this amount of adjustment is needed
    sample_adjustment = int(sr / 668.1818181818181 * 8 + 1)
    if 0 < max_ind < len(raw_search) - sample_adjustment:
        max_ind = max_ind + sample_adjustment
    return max_ind / sr


if __name__ == '__main__':
    search_file = '/home/klux/Downloads/range_2.5_min.mp3'
    # might be useful to also work for low sampling rates
    target_sr = 22050
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_search, sr_b = librosa.load(search_file, res_type='kaiser_fast',
                                        sr=target_sr)

    snippet_folder = '/home/klux/Downloads/snippets15s'
    for file in sorted(os.listdir(snippet_folder)):
        snippet_file = os.path.join(snippet_folder, file)
        print(snippet_file)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_snippet, sr_a = librosa.load(snippet_file, res_type='kaiser_fast',
                                             sr=target_sr)

        result = find_snippet_in_search_range(raw_snippet, raw_search,
                                              target_sr)
        if result is not None:
            print(f'{result} seconds')
        else:
            print('No match found.')