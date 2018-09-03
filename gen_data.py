'''
This script generates data for Guided PCA

2018-09-02
'''

import ipdb as pdb
import os
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tqdm
from utils import *

# Load data
df = pd.read_pickle('data/JW12.pckl')
with open('data/pal_pha.pckl', 'rb') as pckl:
    pal, pha = pickle.load(pckl)
artic_col = ['T1x', 'T1y', 'T2x', 'T2y', 'T3x', 'T3y',
             'T4x', 'T4y', 'ULx', 'ULy', 'LLx', 'LLy', 'MNIx', 'MNIy']
acous_col = ['F1', 'F2', 'F3']
# vowel_list = ['AE1', 'AH1', 'AO1', 'EH1', 'IH1', 'AA1', 'IY1', 'UW1', 'UH1']
vowel_list = ['IY1', 'AE1', 'AA1', 'UW1']
labels = [u'æ', u'ʌ', u'ɔ', u'ɛ', u'ɪ', u'a', u'i', u'u', u'ʊ']

X_raw = df.loc[:, artic_col].copy().as_matrix()
Y_raw = df.loc[:, acous_col].copy().as_matrix()

# Standardize before PCA
#   Articulation
X_scaler = StandardScaler().fit(X_raw)
X_std = X_scaler.transform(X_raw)  # cf .inverse_transform()
#   Acoustics
Y_scaler = StandardScaler().fit(Y_raw)
Y_std = Y_scaler.transform(Y_raw)

#['T1x','T1y','T2x','T2y','T3x','T3y','T4x','T4y','ULx','ULy','LLx','LLy','MNIx','MNIy']
factor_matrix = np.array([
    # factor1: JAWy
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    # factor2: T1x,T1y,T2x,T2y,T3x,T3y,T4x,T4y
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    # factor3: T1x,T1y,T2x,T2y,T3x,T3y,T4x,T4y
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    # factor4: ULx,ULy,LLx,LLy
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    # factor5: T1x,T1y,T2x,T2y,T3x,T3y,T4x,T4y
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
])

# Run Guided PCA
G = GuidedPCA(n_comp=5, factor_matrix=factor_matrix)
G.fit(X_std)
X_reduced_guidedPCA = G.transform(X_std)
# Estimate W
W_guidedPCA = np.dot(np.linalg.pinv(X_reduced_guidedPCA), Y_std)
# Prediction
_Y_pred_guidedPCA = np.dot(X_reduced_guidedPCA, W_guidedPCA)
Y_pred_guidedPCA = Y_scaler.inverse_transform(_Y_pred_guidedPCA)
Y_orig_guidedPCA = Y_scaler.inverse_transform(Y_std)

Error = np.mean(np.abs(Y_orig_guidedPCA - Y_pred_guidedPCA), axis=0)
F1_error_lm, F2_error_lm, F3_error_lm = Error[0], Error[1], Error[2]

# Get reference vowels
medianArtic = np.zeros((len(vowel_list), 14))
_medianAcous = np.zeros((len(vowel_list), 3))
for i, v in enumerate(vowel_list):
    x = df.loc[df.Label == v, artic_col].as_matrix()
    y = df.loc[df.Label == v, acous_col].as_matrix()
    medianArtic[i, :] = np.median(x, axis=0)  # 4x14
    _medianAcous[i, :] = np.median(y, axis=0)  # 4x3 (real data)
# Get predicted reference formants
pred = np.dot(G.transform(X_scaler.transform(medianArtic)), W_guidedPCA)
medianAcous = Y_scaler.inverse_transform(pred)

R = pd.DataFrame(np.concatenate((medianArtic, medianAcous), axis=1),
                 columns=artic_col + acous_col)
R['Vowel'] = vowel_list
R.to_pickle('data/ref_vowel.pckl')
print('Reference vowel... saved')
np.save('data/params.npy', [X_scaler, Y_scaler, G, W_guidedPCA])
pdb.set_trace()


def predict(pc1, pc2, pc3, pc4, pc5):
    x_reduced = np.array([[pc1, pc2, pc3, pc4, pc5]])
    x_recon_scaled = G.inverse_transform(x_reduced)
    W = W_guidedPCA
    pellets = X_scaler.inverse_transform(x_recon_scaled)

    y_scaled = np.dot(x_reduced, W)
    formants = Y_scaler.inverse_transform(y_scaled)
    # Returns:
    #  T1x, T1y, T2x, T2y, T3x, T3y, T4x, T4y, ULx, ULy, LLx, LLy, JAWx, JAWy
    #  F1, F2, F3
    return pellets, formants


# Setting
num_pc = 5
xstep = 0.5
xmin = -5
xmax = 5
ticks = np.arange(xmin, xmax + xstep, xstep)

# Stack data
D = np.zeros((len(ticks)**num_pc, 5 + 14 + 3))  # 320,000 x 22
i = 0
for p1 in tqdm.tqdm(ticks, total=len(ticks)):
    for p2 in ticks:
        for p3 in ticks:
            for p4 in ticks:
                for p5 in ticks:
                    pellets, formants = predict(p1, p2, p3, p4, p5)
                    line = np.concatenate(([[p1, p2, p3, p4, p5]],
                                           pellets, formants), axis=1)
                    D[i, :] = line  # 1x22
                    i += 1
pdb.set_trace()
D = pd.DataFrame(D, columns=[
    'PC1', 'PC2', 'PC3', 'PC4', 'PC5',
    'T1x', 'T1y', 'T2x', 'T2y', 'T3x', 'T3y', 'T4x', 'T4y',
    'ULx', 'ULy', 'LLx', 'LLy', 'JAWx', 'JAWy',
    'F1', 'F2', 'F3']).astype('float32')
D[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']] = D[[
    'PC1', 'PC2', 'PC3', 'PC4', 'PC5']].astype('float16')
D.to_pickle('data/JW12_plot_data.pckl')
print('Done')
pdb.set_trace()
