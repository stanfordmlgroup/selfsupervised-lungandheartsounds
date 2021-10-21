import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
pio.templates.default = "simple_white"

data = [['<b>Baseline</b>', '10%', 'linear', .664, .032],
        ['<b>Supervised</b>', '10%', 'fine-tune', .773, .0345],
        ['<b>Supervised (Full)</b>', '10%', 'fine-tune', .889, .024],
        ['<b>Baseline</b>', '100%', 'linear', .803, .043],
        ['<b>Supervised</b>', '100%', 'fine-tune', .930, .025],
        ['<b>Supervised (Full)</b>', '100%', 'fine-tune', .929, .021],

        ['<b>Spectrogram</b>', '10%', 'fine-tune', .756, .0275],
        ['<b>Spectrogram + Split</b>', '10%', 'fine-tune', .787, .037],
        ['<b>Frequency Only</b>', '10%', 'fine-tune', .792, .0265],
        ['<b>Split</b>', '10%', 'fine-tune', .797, .032],
        ['<b>Time Only</b>', '10%', 'fine-tune', .857, .0285],

        ['<b>Spectrogram</b>', '100%', 'fine-tune', .920, .027],
        ['<b>Spectrogram + Split</b>', '100%', 'fine-tune', .924, .027],
        ['<b>Frequency Only</b>', '100%', 'fine-tune', .925, .025],
        ['<b>Split</b>', '100%', 'fine-tune', .927, .0255],
        ['<b>Time Only</b>', '100%', 'fine-tune', .927, .025],

        ['<b>Spectrogram</b>', '10%', 'linear', .660, .0375],
        ['<b>Frequency Only</b>', '10%', 'linear', .666, .0165],
        ['<b>Spectrogram + Split</b>', '10%', 'linear', .752, .038],
        ['<b>Split</b>', '10%', 'linear', .744, .0335],
        ['<b>Time Only</b>', '10%', 'linear', .808, .033],

        ['<b>Spectrogram</b>', '100%', 'linear', .766, .043],
        ['<b>Frequency Only</b>', '100%', 'linear', .782, .041],
        ['<b>Spectrogram + Split</b>', '100%', 'linear', .795, .0385],
        ['<b>Split</b>', '100%', 'linear', .807, .0435],
        ['<b>Time Only</b>', '100%', 'linear', .874, .032],
        ]

fig = make_subplots(cols=2)
metrics_df = pd.DataFrame(data, columns=['Runs', 'data', 'type', 'AUC', 'ci_95'])
data = metrics_df[metrics_df['type'] == 'linear']
data10 = data[data['data'] == "10%"]
data100 = data[data['data'] == "100%"]
color10 = ['rgb(150,150,150)' if ('Baseline' in name) else f'rgb({250 - 10 * i},0,0)' for i, name
           in enumerate(data10['Runs'])]
color100 = ['rgb(120,120,120)' if ('Baseline' in name) else f'rgb({200 - 10 * i},0,0)' for i, name
            in enumerate(data100['Runs'])]
fig.add_bar(x=data10['Runs'], y=data10['AUC'], error_y=dict(type='data', array=data10['ci_95']), row=1, col=1, marker=dict(color=color10))
fig.add_bar(x=data100['Runs'], y=data100['AUC'], error_y=dict(type='data', array=data100['ci_95']), row=1, col=1, marker=dict(color=color100))
data = metrics_df[metrics_df['type'] == 'fine-tune']
data10 = data[data['data'] == "10%"]
data100 = data[data['data'] == "100%"]
color10 = ['rgb(150,150,150)' if ('Supervised' in name) else f'rgb({250 - 10 * i},0,0)' for i, name
           in enumerate(data10['Runs'])]
color100 = ['rgb(120,120,120)' if ('Supervised' in name) else f'rgb({200 - 10 * i},0,0)' for i, name
            in enumerate(data100['Runs'])]
fig.add_bar(x=data10['Runs'], y=data10['AUC'], error_y=dict(type='data', array=data10['ci_95']), row=1, col=2,  marker=dict(color=color10))
fig.add_bar(x=data100['Runs'], y=data100['AUC'], error_y=dict(type='data', array=data100['ci_95']), row=1, col=2, marker=dict(color=color100))
fig.update_layout(font={"size": 24})
fig.update_yaxes(title_text="AUC", range=[0, 1], row=1, col=1, title_font={"size": 36}, title_standoff=40)
fig.update_yaxes(title_text="", range=[0, 1], row=1, col=2, title_font={"size": 36}, title_standoff=40)
fig.update_xaxes(title_text='Linear Evaluation', row=1, col=1, title_font={"size": 36}, title_standoff=20)
fig.update_xaxes(title_text='Fine-Tune Evaluation', row=1, col=2, title_font={"size": 36}, title_standoff=20)
fig.layout.update(showlegend=False)
fig.write_image('output/heart.png', width=2000, height=900)

data = [['<b>Baseline</b>', '10%', 'linear', .512, .026],
        ['<b>Supervised</b>', '10%', 'fine-tune', .628, .044],
        ['<b>Supervised (Full)</b>', '10%', 'fine-tune', .687, .047],
        ['<b>Baseline</b>', '100%', 'linear', .516, .048],
        ['<b>Supervised</b>', '100%', 'fine-tune', .69, .059],
        ['<b>Supervised (Full)</b>', '100%', 'fine-tune', .71, .0595],

        ['<b>Split</b>', '10%', 'fine-tune', .562, .054],
        ['<b>Time Only</b>', '10%', 'fine-tune', .627, .0565],
        ['<b>Spectrogram + Split</b>', '10%', 'fine-tune', .562, .0505],
        ['<b>Frequency Only</b>', '10%', 'fine-tune', .618, .0515],
        ['<b>Spectrogram</b>', '10%', 'fine-tune', .633, .0575],

        ['<b>Split</b>', '100%', 'fine-tune', .584, .063],
        ['<b>Time Only</b>', '100%', 'fine-tune', .627, .0595],
        ['<b>Spectrogram + Split</b>', '100%', 'fine-tune', .65, .066],
        ['<b>Frequency Only</b>', '100%', 'fine-tune', .671, .0595],
        ['<b>Spectrogram</b>', '100%', 'fine-tune', .691, .065],

        ['<b>Split</b>', '10%', 'linear', .558, .028],
        ['<b>Spectrogram + Split</b>', '10%', 'linear', .533, .035],
        ['<b>Time Only</b>', '10%', 'linear', .643, .0485],
        ['<b>Frequency Only</b>', '10%', 'linear', .649, .0415],
        ['<b>Spectrogram</b>', '10%', 'linear', .652, .0535],

        ['<b>Split</b>', '100%', 'linear', .552, .052],
        ['<b>Spectrogram + Split</b>', '100%', 'linear', .609, .0595],
        ['<b>Time Only</b>', '100%', 'linear', .654, .06],
        ['<b>Frequency Only</b>', '100%', 'linear', .656, .058],
        ['<b>Spectrogram</b>', '100%', 'linear', .659, .058],
        ]

fig = make_subplots(cols=2)
metrics_df = pd.DataFrame(data, columns=['Runs', 'data', 'type', 'AUC', 'ci_95'])
data = metrics_df[metrics_df['type'] == 'linear']
data10 = data[data['data'] == "10%"]
data100 = data[data['data'] == "100%"]
color10 = ['rgb(150,150,150)' if ('Baseline' in name) else f'rgb(0,0,{250 - 10 * i})' for i, name
           in enumerate(data10['Runs'])]
color100 = ['rgb(120,120,120)' if ('Baseline' in name) else f'rgb(0,0,{180 - 10 * i})' for i, name
            in enumerate(data100['Runs'])]
fig.add_bar(x=data10['Runs'], y=data10['AUC'], error_y=dict(type='data', array=data10['ci_95']), row=1, col=1, marker=dict(color=color10))
fig.add_bar(x=data100['Runs'], y=data100['AUC'], error_y=dict(type='data', array=data100['ci_95']), row=1, col=1, marker=dict(color=color100))
data = metrics_df[metrics_df['type'] == 'fine-tune']
data10 = data[data['data'] == "10%"]
data100 = data[data['data'] == "100%"]
color10 = ['rgb(150,150,150)' if ('Supervised' in name) else f'rgb(0,0,{250 - 10 * i})' for i, name
           in enumerate(data10['Runs'])]
color100 = ['rgb(120,120,120)' if ('Supervised' in name) else f'rgb(0,0,{180 - 10 * i})' for i, name
            in enumerate(data100['Runs'])]
fig.add_bar(x=data10['Runs'], y=data10['AUC'], error_y=dict(type='data', array=data10['ci_95']), row=1, col=2,  marker=dict(color=color10))
fig.add_bar(x=data100['Runs'], y=data100['AUC'], error_y=dict(type='data', array=data100['ci_95']), row=1, col=2, marker=dict(color=color100))
fig.update_layout(font={"size": 24})
fig.update_yaxes(title_text="AUC", range=[0, 1], row=1, col=1, title_font={"size": 36}, title_standoff=40)
fig.update_yaxes(title_text="", range=[0, 1], row=1, col=2, title_font={"size": 36}, title_standoff=40)
fig.update_xaxes(title_text='Linear Evaluation', row=1, col=1, title_font={"size": 36}, title_standoff=20)
fig.update_xaxes(title_text='Fine-Tune Evaluation', row=1, col=2, title_font={"size": 36}, title_standoff=20)
fig.layout.update(showlegend=False)
fig.write_image('output/lung.png', width=2000, height=900)

data = [
        ['<b>Pos. Sim. Age</b>', '10%', 'fine-tune', .665, .055],
        ['<b>Pos. Dif. Loc.</b>', '10%', 'fine-tune', .681, .0595],
        ['<b>Pos. Same Loc.</b>', '10%', 'fine-tune', .678, .057],
        ['<b>Pos. Same Loc./Neg. Same Loc.</b>', '10%', 'fine-tune', .732, .052],
        ['<b>Neg. Sim. Sex</b>', '10%', 'fine-tune', .754, .0515],
        ['<b>Neg. Sim. Age</b>', '10%', 'fine-tune', .782, .0465],
        ['<b>Neg. Sim. Age + Sex</b>', '10%', 'fine-tune', .822, .036],

        ['<b>Pos. Sim. Age</b>', '100%', 'fine-tune', .695, .057],
        ['<b>Pos. Dif. Loc.</b>', '100%', 'fine-tune', .702, .0605],
        ['<b>Pos. Same Loc.</b>', '100%', 'fine-tune', .692, .0615],
        ['<b>Pos. Same Loc./Neg. Same Loc.</b>', '100%', 'fine-tune', .768, .0505],
        ['<b>Neg. Sim. Sex</b>', '100%', 'fine-tune', .765, .0565],
        ['<b>Neg. Sim. Age</b>', '100%', 'fine-tune', .785, .052],
        ['<b>Neg. Sim. Age + Sex</b>', '100%', 'fine-tune', .842, .0365],

        ['<b>Pos. Sim. Age</b>', '10%', 'linear', .663, .0515],
        ['<b>Pos. Dif. Loc.</b>', '10%', 'linear', .681, .049],
        ['<b>Pos. Same Loc.</b>', '10%', 'linear', .690, .049],
        ['<b>Pos. Same Loc./Neg. Same Loc.</b>', '10%', 'linear', .695, .048],
        ['<b>Neg. Sim. Sex</b>', '10%', 'linear', .723, .0475],
        ['<b>Neg. Sim. Age</b>', '10%', 'linear', .788, .0475],
        ['<b>Neg. Sim. Age + Sex</b>', '10%', 'linear', .854, .0295],

        ['<b>Pos. Sim. Age</b>', '100%', 'linear', .674, .054],
        ['<b>Pos. Dif. Loc.</b>', '100%', 'linear', .689, .0555],
        ['<b>Pos. Same Loc.</b>', '100%', 'linear', .700, .058],
        ['<b>Pos. Same Loc./Neg. Same Loc.</b>', '100%', 'linear', .745, .0525],
        ['<b>Neg. Sim. Sex</b>', '100%', 'linear', .748, .056],
        ['<b>Neg. Sim. Age</b>', '100%', 'linear', .773, .051],
        ['<b>Neg. Sim. Age + Sex</b>', '100%', 'linear', .863, .028],
        ]

fig = make_subplots(cols=2)
metrics_df = pd.DataFrame(data, columns=['Runs', 'data', 'type', 'AUC', 'ci_95'])
data = metrics_df[metrics_df['type'] == 'linear']
data10 = data[data['data'] == "10%"]
data100 = data[data['data'] == "100%"]
color10 = ['rgb(150,150,150)' if ('Baseline' in name) else f'rgb({120 - 10 * i},0,{190 - 10 * i})' for i, name
           in enumerate(data10['Runs'])]
color100 = ['rgb(120,120,120)' if ('Baseline' in name) else f'rgb({90 - 10 * i},0,{160 - 10 * i})' for i, name
            in enumerate(data100['Runs'])]
fig.add_bar(x=data10['Runs'], y=data10['AUC'], error_y=dict(type='data', array=data10['ci_95']), row=1, col=1, marker=dict(color=color10))
fig.add_bar(x=data100['Runs'], y=data100['AUC'], error_y=dict(type='data', array=data100['ci_95']), row=1, col=1, marker=dict(color=color100))
data = metrics_df[metrics_df['type'] == 'fine-tune']
data10 = data[data['data'] == "10%"]
data100 = data[data['data'] == "100%"]
color10 = ['rgb(150,150,150)' if ('Supervised' in name) else f'rgb({120 - 10 * i},0,{190 - 10 * i})' for i, name
           in enumerate(data10['Runs'])]
color100 = ['rgb(120,120,120)' if ('Supervised' in name) else f'rgb({90 - 10 * i},0,{160 - 10 * i})' for i, name
            in enumerate(data100['Runs'])]
fig.add_bar(x=data10['Runs'], y=data10['AUC'], error_y=dict(type='data', array=data10['ci_95']), row=1, col=2,  marker=dict(color=color10))
fig.add_bar(x=data100['Runs'], y=data100['AUC'], error_y=dict(type='data', array=data100['ci_95']), row=1, col=2, marker=dict(color=color100))
fig.update_layout(font={"size": 24})
fig.update_yaxes(title_text="AUC", range=[0, 1], row=1, col=1, title_font={"size": 36}, title_standoff=40)
fig.update_yaxes(title_text="", range=[0, 1], row=1, col=2, title_font={"size": 36}, title_standoff=40)
fig.update_xaxes(title_text='Linear Evaluation', row=1, col=1, title_font={"size": 36}, title_standoff=20)
fig.update_xaxes(title_text='Fine-Tune Evaluation', row=1, col=2, title_font={"size": 36}, title_standoff=20)
fig.layout.update(showlegend=False)
fig.write_image('output/demographics.png', width=2000, height=900)
