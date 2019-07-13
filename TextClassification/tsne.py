import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.manifold import TSNE


def load_wd2vec(path):
    assert os.path.exists(path)
    wds = []
    vecs = []
    i = 0
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split()
            if len(tokens) > 2:
                wds.append(tokens[0])
                vecs.append(np.array(tokens[1:]).tolist())
                i += 1
                if i > 1000:
                    break
    return wds, vecs


if __name__ == '__main__':

    wds, vecs = load_wd2vec('./data/word2vec_300.txt')

    tsne_vecs = TSNE(n_components=2, init='pca').fit_transform(vecs)
    print(tsne_vecs.shape)

    plt.figure(figsize=(15, 15))
    plt.scatter(tsne_vecs[:, 0], tsne_vecs[:, 1])

    for i, wd in enumerate(wds):
        plt.text(tsne_vecs[i, 0]+0.1, tsne_vecs[i, 1]+0.1, s=wd, fontsize=8)

    plt.show()