import numpy as np

def segmentor(tokens, segment_num):

    ids = np.rint(np.linspace(0, len(tokens), segment_num, endpoint = False)).astype(int)
    segments_id = zip(ids, np.append(ids[1:], len(tokens)))

    segments_token = [tokens [start: end] for start, end in segments_id]
    segments_text = [' '.join(tokens) for tokens in segments_token]

    return segments_text, segments_token, segments_id


