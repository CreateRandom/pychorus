from pychorus import create_chroma
from pychorus.constants import SMOOTHING_SIZE_SEC, OVERLAP_PERCENT_MARGIN
from pychorus.helpers import local_maxima_rows, detect_lines, \
    count_overlapping_lines, best_segment
from pychorus.similarity_matrix_new import TimeTimeSimilarityMatrixNew, TimeLagSimilarityMatrixNew
import numpy as np
if __name__ == '__main__':
    file_a = '/home/klux/Downloads/snippet.mp3'

    file_b = '/home/klux/Downloads/range_3_min.mp3'

    chroma_a, _, sr_a, song_length_sec_a = create_chroma(file_a)
    chroma_b, _, sr_b, song_length_sec_b = create_chroma(file_b)

    # pad with zeros for now
    new_a = np.zeros(chroma_b.shape)
    new_a[:chroma_a.shape[0],:chroma_a.shape[1]] =chroma_a
    time_time_similarity = TimeTimeSimilarityMatrixNew(chroma_b,new_a,sr_a)
    time_lag_similarity = TimeLagSimilarityMatrixNew(chroma_b,new_a,sr_a)

    time_lag_similarity.display()
    num_samples = chroma_a.shape[1]

    # Denoise the time lag matrix
    chroma_sr = num_samples / song_length_sec_a
    smoothing_size_samples = int(SMOOTHING_SIZE_SEC * chroma_sr)
    time_lag_similarity.denoise(time_time_similarity.matrix,
                                smoothing_size_samples)

    time_lag_similarity.display()
    # Detect lines in the image
    clip_length_samples = 30 * chroma_sr
    candidate_rows = local_maxima_rows(time_lag_similarity.matrix)
    lines = detect_lines(time_lag_similarity.matrix, candidate_rows,
                         clip_length_samples)
    if len(lines) == 0:
        print("No choruses were detected. Try a smaller search duration")
    line_scores = count_overlapping_lines(
        lines, OVERLAP_PERCENT_MARGIN * clip_length_samples,
        clip_length_samples)
    repetition = best_segment(line_scores)
    # adjust for sampling rate
    start = repetition.start / chroma_sr

    print("Best chorus found at {0:g} min {1:.2f} sec".format(
        start // 60, start % 60))
