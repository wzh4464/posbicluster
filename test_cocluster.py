

import numpy as np
from sklearn.cluster import SpectralBiclustering
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')



file_path = './video.npy'
frames_matrix = np.load(file_path, allow_pickle=True) # shape(103, 25, 150)
iFile = 0
sig_level = 2
max_nframe_all = 300

nComp = 25*6  #150
frames_matrix_comp_bi = np.zeros((max_nframe_all, 25, nComp), dtype=np.float32)
n_clusters_tuple = (5, 10)

for iFrame in range(5): # range(frames_matrix.shape[0]):
    print(f'iFrame: {iFrame}')

    frames_matrix_comp = frames_matrix[iFrame] # shape(25, 150)
    bicluster_model = SpectralBiclustering(n_clusters=n_clusters_tuple, method="bistochastic", random_state=0)  # bistochastic  log
    bicluster_model.fit(frames_matrix_comp)

    # Reordering first the rows and then the columns.
    reordered_rows = frames_matrix_comp[np.argsort(bicluster_model.row_labels_)]
    reordered_data = reordered_rows[:, np.argsort(bicluster_model.column_labels_)]
    frames_matrix_comp_bi[iFrame] = reordered_data

    acc_fig = plt.figure(figsize=(20, 10))
    filename = f'biclustered_{iFile}_{iFrame}_SigL_{sig_level}_CL_{n_clusters_tuple[0]}_{n_clusters_tuple[1]}.png'
    acc_axe = acc_fig.add_subplot(1, 2, 1)

    acc_axe.matshow(frames_matrix_comp, cmap=plt.cm.Blues)
    plt.title("Before biclustering")
    acc_axe = acc_fig.add_subplot(1, 2, 2)
    acc_axe.matshow(reordered_data, cmap=plt.cm.Blues)
    plt.title(f"After biclustering  {n_clusters_tuple}")

    acc_axe.grid(True, which='minor', ls='--')
    acc_fig.savefig(filename, dpi=300, format='png')  # svg  png
    plt.close()






