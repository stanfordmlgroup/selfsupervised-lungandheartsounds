import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
pio.templates.default = "simple_white"

data = [['Baseline', '10%', 'linear', .664, .032],
        ['Supervised', '10%', 'fine-tune', .773, .0345],
        ['Supervised (Full)', '10%', 'fine-tune', .889, .024],
        ['Baseline', '100%', 'linear', .803, .043],
        ['Supervised', '100%', 'fine-tune', .930, .025],
        ['Supervised (Full)', '100%', 'fine-tune', .929, .021],

        ['Spectrogram', '10%', 'fine-tune', .756, .0275],
        ['Spectrogram + Split', '10%', 'fine-tune', .787, .037],
        ['Frequency Only', '10%', 'fine-tune', .792, .0265],
        ['Split', '10%', 'fine-tune', .797, .032],
        ['Time Only', '10%', 'fine-tune', .857, .0285],

        ['Spectrogram', '100%', 'fine-tune', .920, .027],
        ['Spectrogram + Split', '100%', 'fine-tune', .924, .027],
        ['Frequency Only', '100%', 'fine-tune', .925, .025],
        ['Split', '100%', 'fine-tune', .927, .0255],
        ['Time Only', '100%', 'fine-tune', .927, .025],

        ['Spectrogram', '10%', 'linear', .660, .0375],
        ['Frequency Only', '10%', 'linear', .666, .0165],
        ['Spectrogram + Split', '10%', 'linear', .752, .038],
        ['Split', '10%', 'linear', .744, .0335],
        ['Time Only', '10%', 'linear', .808, .033],

        ['Spectrogram', '100%', 'linear', .766, .043],
        ['Frequency Only', '100%', 'linear', .782, .041],
        ['Spectrogram + Split', '100%', 'linear', .795, .0385],
        ['Split', '100%', 'linear', .807, .0435],
        ['Time Only', '100%', 'linear', .874, .032],
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
fig.update_yaxes(title_text="AUC", range=[0, 1], row=1, col=1)
fig.update_yaxes(title_text="", range=[0, 1], row=1, col=2)
fig.update_xaxes(title_text='Linear Evaluation', row=1, col=1)
fig.update_xaxes(title_text='Fine-Tune Evaluation', row=1, col=2)
#fig.show()
fig.write_image('output/heart.png', width=1500, height=800)

data = [['Baseline', '10%', 'linear', .512, .026],
        ['Supervised', '10%', 'fine-tune', .628, .044],
        ['Supervised (Full)', '10%', 'fine-tune', .687, .047],
        ['Baseline', '100%', 'linear', .516, .048],
        ['Supervised', '100%', 'fine-tune', .69, .059],
        ['Supervised (Full)', '100%', 'fine-tune', .71, .0595],

        ['Split', '10%', 'fine-tune', .562, .054],
        ['Time Only', '10%', 'fine-tune', .627, .0565],
        ['Spectrogram + Split', '10%', 'fine-tune', .562, .0505],
        ['Frequency Only', '10%', 'fine-tune', .618, .0515],
        ['Spectrogram', '10%', 'fine-tune', .633, .0575],

        ['Split', '100%', 'fine-tune', .584, .063],
        ['Time Only', '100%', 'fine-tune', .627, .0595],
        ['Spectrogram + Split', '100%', 'fine-tune', .65, .066],
        ['Frequency Only', '100%', 'fine-tune', .671, .0595],
        ['Spectrogram', '100%', 'fine-tune', .691, .065],

        ['Split', '10%', 'linear', .558, .028],
        ['Spectrogram + Split', '10%', 'linear', .533, .035],
        ['Time Only', '10%', 'linear', .643, .0485],
        ['Frequency Only', '10%', 'linear', .649, .0415],
        ['Spectrogram', '10%', 'linear', .652, .0535],

        ['Split', '100%', 'linear', .552, .052],
        ['Spectrogram + Split', '100%', 'linear', .609, .0595],
        ['Time Only', '100%', 'linear', .654, .06],
        ['Frequency Only', '100%', 'linear', .656, .058],
        ['Spectrogram', '100%', 'linear', .659, .058],

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
fig.update_yaxes(title_text="AUC", range=[0, 1], row=1, col=1)
fig.update_yaxes(title_text="", range=[0, 1], row=1, col=2)
fig.update_xaxes(title_text='Linear Evaluation', row=1, col=1)
fig.update_xaxes(title_text='Fine-Tune Evaluation', row=1, col=2)
#fig.show()
fig.write_image('output/lung.png', width=1500, height=800)

data = [
        ['Pos. Sim. Age', '10%', 'fine-tune', .665, .055],
        ['Pos. Dif. Loc.', '10%', 'fine-tune', .681, .0595],
        ['Pos. Same Loc.', '10%', 'fine-tune', .678, .057],
        ['Pos. Same Loc./Neg. Same Loc.', '10%', 'fine-tune', .732, .052],
        ['Neg. Sim. Sex', '10%', 'fine-tune', .754, .0515],
        ['Neg. Sim. Age', '10%', 'fine-tune', .782, .0465],
        ['Neg. Sim. Age + Sex', '10%', 'fine-tune', .822, .036],

        ['Pos. Sim. Age', '100%', 'fine-tune', .695, .057],
        ['Pos. Dif. Loc.', '100%', 'fine-tune', .702, .0605],
        ['Pos. Same Loc.', '100%', 'fine-tune', .692, .0615],
        ['Pos. Same Loc./Neg. Same Loc.', '100%', 'fine-tune', .768, .0505],
        ['Neg. Sim. Sex', '100%', 'fine-tune', .765, .0565],
        ['Neg. Sim. Age', '100%', 'fine-tune', .785, .052],
        ['Neg. Sim. Age + Sex', '100%', 'fine-tune', .842, .0365],

        ['Pos. Sim. Age', '10%', 'linear', .663, .0515],
        ['Pos. Dif. Loc.', '10%', 'linear', .681, .049],
        ['Pos. Same Loc.', '10%', 'linear', .690, .049],
        ['Pos. Same Loc./Neg. Same Loc.', '10%', 'linear', .695, .048],
        ['Neg. Sim. Sex', '10%', 'linear', .723, .0475],
        ['Neg. Sim. Age', '10%', 'linear', .788, .0475],
        ['Neg. Sim. Age + Sex', '10%', 'linear', .854, .0295],

        ['Pos. Sim. Age', '100%', 'linear', .674, .054],
        ['Pos. Dif. Loc.', '100%', 'linear', .689, .0555],
        ['Pos. Same Loc.', '100%', 'linear', .700, .058],
        ['Pos. Same Loc./Neg. Same Loc.', '100%', 'linear', .745, .0525],
        ['Neg. Sim. Sex', '100%', 'linear', .748, .056],
        ['Neg. Sim. Age', '100%', 'linear', .773, .051],
        ['Neg. Sim. Age + Sex', '100%', 'linear', .863, .028],
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
fig.update_yaxes(title_text="AUC", range=[0, 1], row=1, col=1)
fig.update_yaxes(title_text="", range=[0, 1], row=1, col=2)
fig.update_xaxes(title_text='Linear Evaluation', row=1, col=1)
fig.update_xaxes(title_text='Fine-Tune Evaluation', row=1, col=2)
#fig.show()
fig.write_image('output/demographics.png', width=1500, height=800)
