'''
Tools for building interactive UCM plots

2018-08-14
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from rank_nullspace import nullspace

# Load single speaker data
df = pd.read_pickle('JW12.pckl')
# Load pal, pha
with open('pal_pha.pckl', 'rb') as pckl:
    pal, pha = pickle.load(pckl)
# Set parameters
artic_col = ['T1x', 'T1y', 'T2x', 'T2y', 'T3x', 'T3y',
             'T4x', 'T4y', 'ULx', 'ULy', 'LLx', 'LLy', 'MNIx', 'MNIy']
acous_col = ['F1', 'F2']
vowel_list = ['AE1', 'AH1', 'AO1', 'EH1', 'IH1', 'AA1', 'IY1', 'UW1', 'UH1']

# Prepare dataset
X_raw = df.loc[:, artic_col].values
Y_raw = df.loc[:, acous_col].values
# Standardize before PCA
# Articulation
X_scaler = StandardScaler().fit(X_raw)
X_std = X_scaler.transform(X_raw)  # cf .inverse_transform()
# Acoustics
Y_scaler = StandardScaler().fit(Y_raw)
Y_std = Y_scaler.transform(Y_raw)

# PCA
pca = PCA(n_components=3)
pca.fit(X_std)
X_reduced = pca.transform(X_std)

# Linear Regression
#  X*w = y
W = np.dot(np.linalg.pinv(X_reduced), Y_std)
# Get null space
nullvec = nullspace(W.T)

# Compute median articulation & acoustic values
medianArtic = np.zeros((len(vowel_list), 14))
medianAcous = np.zeros((len(vowel_list), 2))
for i, v in enumerate(vowel_list):
    x = df.loc[df.Label == v, artic_col].values
    y = df.loc[df.Label == v, acous_col].values
    medianArtic[i, :] = np.median(x, axis=0)  # 7x14
    medianAcous[i, :] = np.median(y, axis=0)  # 7x2

# Estimate F1, F2 for each vowel
y_scaled_vowels = np.dot(pca.transform(X_scaler.transform(medianArtic)), W)
y_vowels = Y_scaler.inverse_transform(y_scaled_vowels)  # 7x2

def pca_forward_plot(pc1, pc2, pc3):
    '''Plot forward mapping from PCs'''
    guided_pca = False

    fig = plt.figure(facecolor='white', figsize=(15, 4))
    # PCA space
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(pc1, pc2, pc3, c='black')
    ax1.set_xlabel('PC1: Height')
    ax1.set_ylabel('PC2: Backness')
    ax1.set_zlabel('PC3')
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-5, 5])
    ax1.set_zlim([-5, 5])

    # Articulator space
    # Get data
    x_reduced = np.array([[pc1, pc2, pc3]])  # 1x3
    if guided_pca:
        # x_recon_scaled = G.inverse_transform(x_reduced)
        pass
    else:
        x_recon_scaled = pca.inverse_transform(x_reduced)
    x_recon = X_scaler.inverse_transform(x_recon_scaled)  # 1x14
    T1x, T1y, T2x, T2y, T3x, T3y, T4x, T4y, ULx, ULy, LLx, LLy, JAWx, JAWy = x_recon[0]
    # Draw pal, pha
    ax2 = fig.add_subplot(132)
    ax2.plot(pal[:, 0], pal[:, 1], color='black')
    ax2.plot(pha[:, 0], pha[:, 1], color='black')
    ax2.set_xlim([-90, 40])
    ax2.set_ylim([-30, 30])
    # Draw articulators
    ax2.plot([T4x, T3x, T2x, T1x], [T4y, T3y, T2y, T1y],
             ls='--', lw=1, marker='o', markersize=5, zorder=1)
    ax2.plot([ULx, LLx], [ULy, LLy],
             ls='None', color='r', marker='o', markersize=5, zorder=1)
    ax2.plot(JAWx, JAWy,
             ls='None', color='g', marker='o', markersize=5, zorder=1)
    # Get UCM, CM space
    # F1-F2 space
    # Estimate F1, F2
    y_scaled = np.dot(x_reduced, W)
    y_recon = Y_scaler.inverse_transform(y_scaled)
    F1, F2 = y_recon[0]
    # Draw F1, F2
    ax3 = fig.add_subplot(133)
    ax3.plot(y_vowels[:, 1], y_vowels[:, 0], ls='None',
             marker='*', markersize=5, color='gray')
    for i, v in enumerate(vowel_list):
        ax3.text(y_vowels[i, 1], y_vowels[i, 0], v[:-1], color='gray')
    ax3.plot(F2, F1, marker='o', markersize=5, color='red')
    ax3.invert_xaxis()
    ax3.invert_yaxis()
    ax3.set_xlim([2100, 800])
    ax3.set_ylim([700, 200])
