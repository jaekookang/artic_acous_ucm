'''
CM plotter (linear regression)

2018-09-03
'''
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from app import app

import numpy as np
import pandas as pd
import pickle
from utils import *


# Load data
df = pd.read_pickle('data/JW12.pckl')
R = pd.read_pickle('data/ref_vowel.pckl')  # 4x(14+3)
with open('data/pal_pha.pckl', 'rb') as pckl:
    pal, pha = pickle.load(pckl)
artic_col = ['T1x', 'T1y', 'T2x', 'T2y', 'T3x', 'T3y',
             'T4x', 'T4y', 'ULx', 'ULy', 'LLx', 'LLy', 'MNIx', 'MNIy']
acous_col = ['F1', 'F2', 'F3']
vowel_list = ['IY1', 'AE1', 'AA1', 'UW1']
# Load params
X_scaler, Y_scaler, pca, ucm_vec, W \
    = np.load('data/params_pca=3_lr.npy')

# Styles
header_style = {
    'textAlign': 'center',
}
col_style = {
    'width': '30%',
    'display': 'inline-block',
}
col_slider_style = {
    'margin': 'auto',
    'width': '30%',
    'display': 'center',
    'textAlign': 'center',
}
slider_style = {
    'display': 'flex',
    'flexDirection': 'row',
    'padding': '0px 0px 10px 0px'
}
slider_text_style = {
    'padding': '0px 10px 10px 0px',
}
slider_dec_style = {
    'fontSize': '10px',
    'padding': '0px 10px 10px 0px',
    'width': '55px',
}
row_style = {
    'display': 'inline-block',
    'padding': '0px 0px 0px 0px',
}

fig_width = 350
fig_height = 300  # 350
margins = {'l': 50, 'r': 50, 'b': 50, 't': 50}

xmin = -3.0
xmax = 3.0
xstep = 0.5
values = np.arange(xmin, xmax + xstep, xstep, dtype=np.float16)


def add_slider(id):
    return dcc.Slider(
        id=id,
        # TODO: Fix marks!
        # marks={str(v): f'{v:.1f}' for v in values},
        min=xmin,
        max=xmax,
        step=xstep,
        value=0.0,
        updatemode='drag',
    )


# Figures
fig_pharynx = go.Scatter(
    x=pha[:, 0],
    y=pha[:, 1],
    mode='lines',
    name='Pharynx',
    line=dict(
        color=('rgb(0,0,0)')
    ),
    hoverinfo='none',
)
fig_palate = go.Scatter(
    x=pal[:, 0],
    y=pal[:, 1],
    mode='lines',
    name='Palate',
    line=dict(
        color=('rgb(0,0,0)')
    ),
    hoverinfo='none',
)
fig_pca3d = go.Scatter3d(
    x=[0],
    y=[0],
    z=[0],
    mode='markers',
    visible=True,
    marker=dict(
        size=3,
        symbol='circle',
        color='rgb(112,112,112)',
        opacity=0.8
    ),
)
fig_f2f1 = go.Scatter(
    x=R['F2'],
    y=R['F1'],
    name='',
    mode='markers',
    marker=dict(
        size=3,
        symbol='circle',
        color='rgb(112,112,112)',
        opacity=0.8
    ),
)

# Traces
traces_311 = [fig_pca3d]
layout_311 = go.Layout(
    height=fig_height,
    width=fig_width,
    margin={'l': 10, 'r': 10, 'b': 10, 't': 10},
    showlegend=False,
    scene=dict(
        xaxis=dict(
            title='PC1',
            range=[xmin * 1.5, xmax * 1.5],
            dtick=1,
            mirror=True,
            tickfont=dict(
                size=10,
            ),
        ),
        yaxis=dict(
            title='PC2',
            range=[xmin * 1.5, xmax * 1.5],
            dtick=1,
            mirror=True,
            tickfont=dict(
                size=10,
            ),
        ),
        zaxis=dict(
            title='PC3',
            range=[xmin * 1.5, xmax * 1.5],
            dtick=1,
            mirror=True,
            tickfont=dict(
                size=10,
            ),
        ),
        camera=dict(
            eye=dict(x=2.5, y=2.5, z=0.8)
        ),
        aspectmode='manual',
        aspectratio={'x': 1.2, 'y': 1.2, 'z': 1.2},
        annotations=[{
            'text': '(0,0)',
            'showarrow': False,
            'x': -0.2,
            'y': -0.2,
        }]
    )
)

traces_312 = [fig_palate, fig_pharynx]
layout_312 = go.Layout(
    height=fig_height,
    width=fig_width,
    margin=margins,
    showlegend=False,
    # Axis
    xaxis=dict(
        linecolor='black',
        mirror=True,
        title='Back <--> Front (mm)<br />Midsagittal view',
        range=[-90, 40],
        dtick=20,
        tickangle=0,
        zeroline=False,
        tickfont=dict(
            size=10,
        ),
        fixedrange=True,
    ),
    yaxis=dict(
        title='Low <--> High (mm)',
        linecolor='black',
        mirror=True,
        range=[-30, 30],
        dtick=10,
        zeroline=False,
        tickfont=dict(
            size=10,
        ),
        fixedrange=True,
    ),
)

traces_313 = [fig_f2f1]
layout_313 = go.Layout(
    height=fig_height,
    width=fig_width,
    margin=margins,
    showlegend=False,
    xaxis=dict(
        title='F2 (Hz)<br />Formant space (F2-F1)',
        linecolor='black',
        mirror=True,
        range=[2200, 800],
        dtick=250,
        zeroline=False,
        tickfont=dict(
            size=10,
        ),
        fixedrange=True,
    ),
    yaxis=dict(
        title='F1 (Hz)',
        linecolor='black',
        mirror=True,
        range=[800, 200],
        dtick=100,
        zeroline=False,
        tickfont=dict(
            size=10,
        ),
        fixedrange=True,
    ),
    annotations=[{
        'text': v,
        'showarrow': False,
        'xref': 'x',
        'yref': 'y',
        'x': R.loc[R.Vowel == v, 'F2'].values[0] - 20,
        'y': R.loc[R.Vowel == v, 'F1'].values[0] - 20,
    } for v in vowel_list])

# Layout
layout = html.Div([
    # Header
    html.H3('CM space in Articulation',
            style=header_style),
    html.H5('Changes in articulation DO affect acoustics',
            style=header_style),

    # Dropdown menu
    html.Div(className='row', children=[
        dcc.Dropdown(
            id='app2-dropdown',
            options=[
                {'label': f'Vowel - {v}', 'value': i}
                for i, v in enumerate(vowel_list)
            ],
            value=0,
        ),
        html.Div(id='app2-display-value'),
    ], style=col_slider_style),

    html.Br(),

    html.Div(className='row', children=[
        html.H6('CM (2D)'),
        html.Div([
            html.P('CM1', style=slider_text_style),
            add_slider('slider-cm1'),
            html.P('CM2', style=slider_text_style),
            add_slider('slider-cm2'),
        ]),
    ], style=col_slider_style),

    html.Div(className='row', children=[
        # (Left) 3D PCA space
        html.Div([
            html.Div([
                dcc.Graph(
                    id='311_cm',
                    figure={
                        'data': traces_311,
                        'layout': layout_311,
                    },
                ),
            ], style=row_style),
        ], className='three columns', style=col_style),

        # (Center) Midsagittal view
        html.Div([
            html.Div([
                dcc.Graph(
                    id='312_cm',
                    figure={
                        'data': traces_312,
                        'layout': layout_312,
                    },
                ),
            ], style=row_style),
        ], className='three columns', style=col_style),

        # (Right) F2-F1 space
        html.Div([
            html.Div([
                dcc.Graph(
                    id='313_cm',
                    figure={
                        'data': traces_313,
                        'layout': layout_313,
                    },
                ),
            ], style=row_style),
        ], className='three columns', style=col_style),
    ]),

    html.Br(),
    dcc.Link('Go to UCM plots', href='/apps/app_ucm'),
    html.Br(),
    dcc.Link('Back to main page', href='/')
])


@app.callback(
    Output('311_cm', 'figure'),
    [Input('slider-cm1', 'value'),
     Input('slider-cm2', 'value'),
     Input('app2-dropdown', 'value')],
)
def update_311(cm1, cm2, vowel_idx):
    cm1 = float(cm1)
    cm2 = float(cm2)

    which_vowel = vowel_list[vowel_idx]
    xs = df.loc[df.Label == which_vowel, artic_col].as_matrix()
    medianArticV = np.median(xs, axis=0, keepdims=True)  # 1x14
    init_pcs = pca.transform(X_scaler.transform(medianArticV))
    weigthed_W = np.multiply(W, np.tile([cm1, cm2], [3, 1]))
    pc1, pc2, pc3 = (init_pcs + np.sum(weigthed_W, axis=1))[0]
    # Make UCM space
    d = [[(init_pcs + ucm_vec.T * i)[0, 0],
          (init_pcs + ucm_vec.T * i)[0, 1],
          (init_pcs + ucm_vec.T * i)[0, 2]]
         for i in np.arange(xmin, xmax, 0.01)]
    d = np.array(d)
    # Make CM space
    dd = [(init_pcs + W.T * (cm1 + cm2))[0]
          for cm1 in np.arange(xmin, xmax, 0.1)
          for cm2 in np.arange(xmin, xmax, 0.01)]
    dd = np.array(dd)

    t1 = go.Scatter3d(
        x=d[:, 0].tolist(),
        y=d[:, 1].tolist(),
        z=d[:, 2].tolist(),
        mode='lines',
        visible=True,
        marker=dict(
            size=5,
            color='rgb(0,0,200)',
            opacity=0.8
        ),
    )

    t2 = go.Scatter3d(
        x=dd[[0, -1], 0].tolist(),
        y=dd[[0, -1], 1].tolist(),
        z=dd[[0, -1], 2].tolist(),
        mode='lines',
        visible=True,
        marker=dict(
            size=5,
            color='rgb(200,0,0)',
            opacity=0.8
        ),
    )

    t3 = go.Scatter3d(
        x=[pc1],
        y=[pc2],
        z=[pc3],
        mode='markers',
        visible=True,
        marker=dict(
            size=5,
            symbol='circle',
            color='rgb(255,0,0)',
            opacity=0.8
        ),
    )
    return {'data': traces_311 + [t1, t2, t3],
            'layout': layout_311}


@app.callback(
    Output('312_cm', 'figure'),
    [Input('slider-cm1', 'value'),
     Input('slider-cm2', 'value'),
     Input('app2-dropdown', 'value')],
)
def update_312(cm1, cm2, vowel_idx):
    cm1 = float(cm1)
    cm2 = float(cm2)

    which_vowel = vowel_list[vowel_idx]
    xs = df.loc[df.Label == which_vowel, artic_col].as_matrix()
    medianArticV = np.median(xs, axis=0, keepdims=True)  # 1x14
    init_pcs = pca.transform(X_scaler.transform(medianArticV))
    weigthed_W = np.multiply(W, np.tile([cm1, cm2], [3, 1]))
    pc1, pc2, pc3 = (init_pcs + np.sum(weigthed_W, axis=1))[0]
    # Get articulators
    x_reduced = np.array([[pc1, pc2, pc3]])  # 1x3
    x_recon_scaled = pca.inverse_transform(x_reduced)
    x_recon = X_scaler.inverse_transform(x_recon_scaled)  # 1x14
    T1x, T1y, T2x, T2y, T3x, T3y, T4x, T4y, ULx, ULy, LLx, LLy, JAWx, JAWy = x_recon[0]

    trace = go.Scatter(
        x=[T1x, T2x, T3x, T4x, None, ULx, None, LLx, None, JAWx],
        y=[T1y, T2y, T3y, T4y, None, ULy, None, LLy, None, JAWy],
        name='',
        mode='lines+markers',
        line={
            'shape': 'spline',
        },
        marker={
            'size': 5,
            'color': 'rgb(0,0,0)',
        },
    )
    return {'data': traces_312 + [trace],
            'layout': layout_312}


@app.callback(
    Output('313_cm', 'figure'),
    [Input('slider-cm1', 'value'),
     Input('slider-cm2', 'value'),
     Input('app2-dropdown', 'value')],
)
def update_313(cm1, cm2, vowel_idx):
    cm1 = float(cm1)
    cm2 = float(cm2)

    which_vowel = vowel_list[vowel_idx]
    xs = df.loc[df.Label == which_vowel, artic_col].as_matrix()
    medianArticV = np.median(xs, axis=0, keepdims=True)  # 1x14
    init_pcs = pca.transform(X_scaler.transform(medianArticV))
    weigthed_W = np.multiply(W, np.tile([cm1, cm2], [3, 1]))
    pc1, pc2, pc3 = (init_pcs + np.sum(weigthed_W, axis=1))[0]
    # Get formants
    x_reduced = np.array([[pc1, pc2, pc3]])  # 1x3
    y_scaled = np.dot(x_reduced, W)
    y_recon = Y_scaler.inverse_transform(y_scaled)
    F1, F2 = y_recon[0]

    trace = go.Scatter(
        x=[F2],
        y=[F1],
        name='',
        mode='markers',
        marker={
            'size': 5,
            'color': 'rgb(255,0,0)'
        },
    )

    return {'data': traces_313 + [trace],
            'layout': layout_313}


@app.callback(
    Output('app2-display-value', 'children'),
    [Input('app2-dropdown', 'value')]
)
def display_value(value):
    return f'Vowel: {vowel_list[value]} is selected'
