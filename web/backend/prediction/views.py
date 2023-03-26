from django.shortcuts import render
from .models import nanoparticle_mri_r1_pred, nanoparticle_mri_r2_pred, nanoparticle_sar_pred
import numpy as np
import pandas as pd
from rdkit.Chem.Crippen import MolLogP
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from rdkit.Chem.Lipinski import NumHAcceptors
import re
from rdkit.Chem.Descriptors import *
from prediction.forms import PredictionForm, PredictionFormMedium, PredictionFormBasic
from django_plotly_dash import DjangoDash
from dash import html
from dash import dcc
import plotly.express as px
from dash import dash_table




magnetic_moments = {'Gd': 7.94, 'Fe2+': 4.9, 'Fe3+': 5.92, 'Cu': 1.73, 'Eu': 3.4, 'Zn': 0, 'Mn': 5.92, 'Dy': 10.63,
                    'Pt': 2.83, 'Na': 1.73, 'Al': 1.73, 'Co': 3.88, 'Ni': 2.83, 'O': 0, 'Si': 0, 'Fe': 4.9, 'F': 0,
                    'Ho': 10.5, 'Au': 1.73, 'Cd': 0, 'S': 0, 'Tb': 9.7, 'Mg': 0}

spins = {'Gd': 7/2, 'Fe2+': 2, 'Fe3+': 5/2, 'Cu': 1/2, 'Eu': 3, 'Zn': 0, 'Mn': 5/2, 'Dy': 5/2, 'Pt': 1, 'Na': 1/2,
                    'Al': 1/2, 'Co': 2, 'Ni': 3/2, 'O': 0, 'Si': 0, 'Hollow': 0, 'Fe': 2, 'F': 0, 'Ho': 2, '': '', 'C': 0, 'GO': 0,
                    'Au': 1/2, 'Cd': 0}

db_r1 = pd.DataFrame(list(nanoparticle_mri_r1_pred.objects.values('av', 'mm', 'magnetic_moment', 'sum_surface_spins',
                                                     'squid_sat_mag',
                                                     'org_coating_LogP', 'org_coating_HAcceptors', 'mri_h_val', 'mri_r1')))
db_r1 = db_r1.astype(float)
x1 = db_r1.loc[:, 'av':'mri_h_val'].values
y1 = db_r1.loc[:, 'mri_r1'].values
y_discretized1 = KBinsDiscretizer(n_bins=5,
                                 encode='ordinal',
                                 strategy='uniform').fit_transform(y1.reshape(-1, 1))
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x1, y1.ravel(),
                                                  test_size=0.2,
                                                  random_state=10,
                                                  stratify=y_discretized1)
sc1 = MinMaxScaler( )
x_train_1 = sc1.fit_transform(x_train_1)
y_train_1 = np.log10(y_train_1)
from sklearn.ensemble import ExtraTreesRegressor
model_mri1 = ExtraTreesRegressor(min_samples_leaf=2, n_estimators=102)
model_mri1.fit(x_train_1, y_train_1)


db_r2 = pd.DataFrame(list(nanoparticle_mri_r2_pred.objects.values('av', 'mm', 'magnetic_moment', 'sum_surface_spins',
                                                     'squid_sat_mag',
                                                     'org_coating_LogP', 'org_coating_HAcceptors', 'mri_h_val', 'mri_r2')))
db_r2 = db_r2.astype(float)
x2 = db_r2.loc[:, 'av':'mri_h_val'].values
y2 = db_r2.loc[:, 'mri_r2'].values
y_discretized2 = KBinsDiscretizer(n_bins=5,
                                 encode='ordinal',
                                 strategy='uniform').fit_transform(y2.reshape(-1, 1))
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x2, y2.ravel(),
                                                  test_size=0.2,
                                                  random_state=10,
                                                  stratify=y_discretized2)
sc2 = MinMaxScaler()
x_train_2 = sc2.fit_transform(x_train_2)
y_train_2 = np.log10(y_train_2)
from sklearn.ensemble import ExtraTreesRegressor
model_mri2 = ExtraTreesRegressor(min_samples_leaf=2, n_estimators=102)
model_mri2.fit(x_train_2, y_train_2)


db_sar = pd.DataFrame(list(nanoparticle_sar_pred.objects.values('conc', 'av', 'mm', 'magnetic_moment',
                                                     'squid_sat_mag', 'squid_coerc_f', 'squid_rem_mag',
                                                     'org_coating_LogP', 'org_coating_HAcceptors', 'htherm_h_amp', 'htherm_h_freq', 'sar')))
db_sar = db_sar.astype(float)
x3 = db_sar.loc[:, 'conc':'htherm_h_freq'].values
y3 = db_sar.loc[:, 'sar'].values
y_discretized3 = KBinsDiscretizer(n_bins=5,
                                 encode='ordinal',
                                 strategy='uniform').fit_transform(y3.reshape(-1, 1))
x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(x3, y3.ravel(),
                                                  test_size=0.2,
                                                  random_state=10,
                                                  stratify=y_discretized3)
sc3 = MinMaxScaler()
x_train_3= sc3.fit_transform(x_train_3)
y_train_3 = np.log10(y_train_3)
from lightgbm import LGBMRegressor
model_sar = LGBMRegressor(n_estimators=100, max_depth=2,
                          learning_rate=0.26485162360302365, min_split_gain=9.598896733776042e-05,
                          min_child_weight=0.031405041091885896, min_child_samples=14,
                          reg_alpha=0.15151696313077256, reg_lambda=0.13651480472249508)
model_sar.fit(x_train_3, y_train_3)


#MRI

def main(request):
    form = PredictionForm()
    if request.method == "POST":
        form = PredictionForm(request.POST or None)
        core_comp_0 = str(request.POST['core_comp'])
        shape = str(request.POST['shape'])
        length = float(request.POST['length'])
        width = float(request.POST['width'])
        depth = float(request.POST['depth'])
        coat_comp = str(request.POST['coat_comp'])
        smiles = str(request.POST['smiles'])
        sat_magn = float(request.POST['sat_magn'])
        field_strenght = float(request.POST['field_strenght'])

        core_comp = core_comp_0
        core_comp = core_comp + 'O'
        for i in range(0, len(core_comp)):
            if core_comp[i].islower():
                if ((i == 0 or core_comp[i - 1].islower() or core_comp[i - 1].isnumeric()) and
                        core_comp[i + 1].islower() or core_comp[i] == 'o' and not core_comp[i - 1].isupper()):
                    core_comp = core_comp[:i] + core_comp[i].capitalize() + core_comp[i + 1:]
        core_comp = core_comp[:-1]

        if core_comp == 'Fe3O4':
            core_comp = 'Fe1Fe2O4'

        core_comp = ''.join(re.findall(r'[A-Z][a-z]*[\d.]+', re.sub('[A-Z][a-z]*(?![\da-z])', r'\g<0>1', core_comp)))

        core_elements = re.findall('[A-Z][a-z]?[a-z]?', str(core_comp))
        r = core_elements
        for elem in range(len(r) - 1):
            if r[elem] == 'Fe' and r[elem + 1] == 'Fe':
                r[elem] = 'Fe2+'
            elif r[elem] == 'Fe':
                r[elem] = 'Fe3+'
        core_elements = r
        core_stoi = re.findall('[\d.]+', str(core_comp))
        sum = 0
        for i in core_stoi:
            sum += float(i)
        magnetic_moment_core = 0
        spin = 0
        for i in range(len(core_elements)):
            magnetic_moment_core += (magnetic_moments[core_elements[i]] * float(core_stoi[i]))/sum
            spin += (spins[core_elements[i]] * float(core_stoi[i]))/sum


        if coat_comp != '0':
            coat_elements = re.findall('[A-Z][a-z]?', str(coat_comp))
            r = coat_elements
            for elem in range(len(r) - 1):
                if r[elem] == 'Fe' and r[elem + 1] == 'Fe':
                    r[elem] = 'Fe2+'
                elif r[elem] == 'Fe':
                    r[elem] = 'Fe3+'
            coat_elements = r
            coat_stoi = re.findall('[\d.]+', str(coat_comp))
            sum = 0
            for i in coat_stoi:
                sum += float(i)
            spin = 0
            for i in range(len(coat_elements)):
                spin += (spins[coat_elements[i]] * float(coat_stoi[i])) / sum


        mm = max(float(length), float(width), float(depth)) / min(float(length), float(width), float(depth))

        if shape == 'Spherical':
            av = 3 / float(length)
        elif shape == 'Cubic':
            av = 6 / float(length)
        elif shape == 'Rod':
            r = min(float(length), float(width), float(depth))
            l = max(float(length), float(width), float(depth))
            av = (2 * r + 2 * l) / (r * l)
        elif shape == 'Rectangle':
            r = min(float(length), float(width), float(depth))
            l = max(float(length), float(width), float(depth))
            av = (2 * r + 4 * l) / (r * l)

        if smiles != '0':
            m = Chem.MolFromSmiles(smiles)
            Hacceptors = NumHAcceptors(m)
            LogP = MolLogP(m)
        else:
            Hacceptors = 0
            LogP = 0

        x_pred0 = pd.DataFrame([av, mm, magnetic_moment_core, spin, sat_magn,
                               LogP, Hacceptors, field_strenght]).transpose()

        mri_r11 = round(10 ** float(model_mri1.predict(sc1.transform(x_pred0))), 2)
        mri_r22 = round(10 ** float(model_mri2.predict(sc2.transform(x_pred0))), 2)

        av = np.repeat(av, 38)
        mm = np.repeat(mm, 38)
        magnetic_moment_core = np.repeat(magnetic_moment_core, 38)
        spin = np.repeat(spin, 38)
        sat_magn = np.repeat(sat_magn, 38)
        LogP = np.repeat(LogP, 38)
        Hacceptors = np.repeat(Hacceptors, 38)
        strenth = np.arange(0.5, 10, 0.25)
        valh= 'Field strength, T'
        x_pred = pd.DataFrame([av, mm, magnetic_moment_core, spin, sat_magn,
                             LogP, Hacceptors, strenth]).transpose()
        x_pred1 = sc1.transform(x_pred)
        x_pred2 = sc2.transform(x_pred)
        mri_r1 = 10 ** model_mri1.predict(x_pred1)
        mri_r2 = 10 ** model_mri2.predict(x_pred2)
        xh = x_pred.loc[:, 7].values
        pred = pd.DataFrame({'params': valh, 'values': xh, 'predicted R1': mri_r1, 'predicted R2': mri_r2})

        # %% plots
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        graph = DjangoDash('prediction', external_stylesheets=external_stylesheets)

        fig1 = px.line(x=pred['values'], y=pred['predicted R1'], markers=True, line_shape='spline')
        fig1.update_traces(line_color='rgb(193, 39, 45, 0.93)')
        fig1.update_layout(margin={'l': 10, 'b': 10, 't': 10, 'r': 0})
        fig1.update_xaxes(title='Field strength, T')
        fig1.update_yaxes(title='predicted r₁, mM⁻¹s⁻¹')

        fig2 = px.line(x=pred['values'], y=pred['predicted R2'], markers=True, line_shape='spline')
        fig2.update_traces(line_color='rgb(193, 39, 45, 0.93)')
        fig2.update_layout(margin={'l': 10, 'b': 10, 't': 10, 'r': 0})
        fig2.update_xaxes(title='Field strength, T')
        fig2.update_yaxes(title='predicted r₂, mM⁻¹s⁻¹')


        graph.layout = html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2)
        ])


        result = pd.DataFrame([xh, mri_r1, mri_r2]).transpose().rename(
            columns={0: 'Field strength, T', 1: 'r₁ relaxivity, mM⁻¹s⁻¹', 2: 'r₂ relaxivity, mM⁻¹s⁻¹'}).round(
            {'Field strength, T': 2, 'r₁ relaxivity, mM⁻¹s⁻¹': 2, 'r₂ relaxivity, mM⁻¹s⁻¹': 2})
        external_stylesheets = [{'font-family':'Arial',},]
        graph = DjangoDash('pred', external_stylesheets=external_stylesheets)
        graph.layout = html.Div([
            dash_table.DataTable(
                columns=[{"name": i, "id": i, } for i in result.columns],
                data=result.to_dict('records'),
                style_header={'backgroundColor': 'rgb(15, 4, 76)', 'color': 'white', 'font-family': 'Arial', 'fontSize': 15,
                              'fontWeight': 'bold', 'height': 'auto', 'whiteSpace': 'normal',  },
                style_cell={'textAlign': 'left', 'fontSize': 15, 'height': 'auto', 'whiteSpace': 'normal', 'font-family': 'Arial'}, ),
        ])


        data = []
        data.append({'core_comp_0': core_comp_0, 'field_strenght': field_strenght, 'mri_r11': mri_r11, 'mri_r22': mri_r22})
        return render(request, 'prediction/prediction.html', {'data': data})
    if form.is_valid():
        form.save()
    context = {'form': form}
    return render(request, 'prediction/prediction.html', context)








def medium(request):
    form = PredictionFormMedium()
    if request.method == "POST":
        form = PredictionFormMedium(request.POST or None)
        core_comp_0 = str(request.POST['core_comp'])
        shape = str(request.POST['shape'])
        length = float(request.POST['length'])
        width = float(request.POST['width'])
        depth = float(request.POST['depth'])
        coat_comp = str(request.POST['coat_comp'])
        smiles = str(request.POST['smiles'])
        field_strenght = float(request.POST['field_strenght'])

        core_comp = core_comp_0
        core_comp = core_comp + 'O'
        for i in range(0, len(core_comp)):
            if core_comp[i].islower():
                if ((i == 0 or core_comp[i - 1].islower() or core_comp[i - 1].isnumeric()) and
                        core_comp[i + 1].islower() or core_comp[i] == 'o' and not core_comp[i - 1].isupper()):
                    core_comp = core_comp[:i] + core_comp[i].capitalize() + core_comp[i + 1:]
        core_comp = core_comp[:-1]

        if core_comp == 'Fe3O4':
            core_comp = 'Fe1Fe2O4'

        core_comp = ''.join(re.findall(r'[A-Z][a-z]*[\d.]+', re.sub('[A-Z][a-z]*(?![\da-z])', r'\g<0>1', core_comp)))


        core_elements = re.findall('[A-Z][a-z]?[a-z]?', str(core_comp))
        r = core_elements
        for elem in range(len(r) - 1):
            if r[elem] == 'Fe' and r[elem + 1] == 'Fe':
                r[elem] = 'Fe2+'
            elif r[elem] == 'Fe':
                r[elem] = 'Fe3+'
        core_elements = r
        core_stoi = re.findall('[\d.]+', str(core_comp))
        sum = 0
        for i in core_stoi:
            sum += float(i)
        magnetic_moment_core = 0
        spin = 0
        for i in range(len(core_elements)):
            magnetic_moment_core += (magnetic_moments[core_elements[i]] * float(core_stoi[i])) / sum
            spin += (spins[core_elements[i]] * float(core_stoi[i])) / sum

        if coat_comp != '0':
            coat_elements = re.findall('[A-Z][a-z]?', str(coat_comp))
            r = coat_elements
            for elem in range(len(r) - 1):
                if r[elem] == 'Fe' and r[elem + 1] == 'Fe':
                    r[elem] = 'Fe2+'
                elif r[elem] == 'Fe':
                    r[elem] = 'Fe3+'
            coat_elements = r
            coat_stoi = re.findall('[\d.]+', str(coat_comp))
            sum = 0
            for i in coat_stoi:
                sum += float(i)
            spin = 0
            for i in range(len(coat_elements)):
                spin += (spins[coat_elements[i]] * float(coat_stoi[i])) / sum


        mm = max(float(length), float(width), float(depth)) / min(float(length), float(width), float(depth))

        if shape == 'Spherical':
            av = 3 / float(length)
        elif shape == 'Cubic':
            av = 6 / float(length)
        elif shape == 'Rod':
            r = min(float(length), float(width), float(depth))
            l = max(float(length), float(width), float(depth))
            av = (2 * r + 2 * l) / (r * l)
        elif shape == 'Rectangle':
            r = min(float(length), float(width), float(depth))
            l = max(float(length), float(width), float(depth))
            av = (2 * r + 4 * l) / (r * l)

        if smiles != '0':
            m = Chem.MolFromSmiles(smiles)
            Hacceptors = NumHAcceptors(m)
            LogP = MolLogP(m)
        else:
            Hacceptors = 0
            LogP = 0

        row = {'av': av, 'mm': mm, 'magnetic_moment': magnetic_moment_core, 'sum_surface_spins': spin,
                'squid_sat_mag': np.nan, 'org_coating_LogP': LogP,
                'org_coating_HAcceptors': Hacceptors}

        df_kNN = db_r2.drop(columns=['mri_h_val', 'mri_r2']).append(row, ignore_index=True)

        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=3)
        imputed1 = imputer.fit_transform(df_kNN)
        df_imputed = pd.DataFrame(imputed1, columns=df_kNN.columns)
        sat = round(float(df_imputed[-1:].squid_sat_mag.values), 2)



        x_pred = pd.DataFrame([av, mm, magnetic_moment_core, spin, sat,
                             LogP, Hacceptors, field_strenght]).transpose()
        x_pred1 = sc1.transform(x_pred)
        x_pred2 = sc2.transform(x_pred)
        mri_r11 = round(10 ** float(model_mri1.predict(sc1.transform(x_pred1))), 2)
        mri_r22 = round(10 ** float(model_mri2.predict(sc2.transform(x_pred2))), 2)



        av = np.repeat(av, 38)
        mm = np.repeat(mm, 38)
        magnetic_moment_core = np.repeat(magnetic_moment_core, 38)
        spin = np.repeat(spin, 38)
        sat_magn = np.repeat(sat, 38)
        LogP = np.repeat(LogP, 38)
        Hacceptors = np.repeat(Hacceptors, 38)
        strenth = np.arange(0.5, 10, 0.25)
        valh= 'Field strength, T'
        x_pred = pd.DataFrame([av, mm, magnetic_moment_core, spin, sat_magn,
                             LogP, Hacceptors, strenth]).transpose()
        x_pred1 = sc1.transform(x_pred)
        x_pred2 = sc2.transform(x_pred)
        mri_r1 = 10 ** model_mri1.predict(x_pred1)
        mri_r2 = 10 ** model_mri2.predict(x_pred2)
        xh = x_pred.loc[:, 7].values
        pred = pd.DataFrame({'params': valh, 'values': xh, 'predicted R1': mri_r1, 'predicted R2': mri_r2})

        # %% plots
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        graph = DjangoDash('prediction', external_stylesheets=external_stylesheets)

        fig1 = px.line(x=pred['values'], y=pred['predicted R1'], markers=True, line_shape='spline')
        fig1.update_traces(line_color='rgb(193, 39, 45, 0.93)')
        fig1.update_layout(margin={'l': 10, 'b': 10, 't': 10, 'r': 0})
        fig1.update_xaxes(title='Field strength, T')
        fig1.update_yaxes(title='predicted r₁, mM⁻¹s⁻¹')

        fig2 = px.line(x=pred['values'], y=pred['predicted R2'], markers=True, line_shape='spline')
        fig2.update_traces(line_color='rgb(193, 39, 45, 0.93)')
        fig2.update_layout(margin={'l': 10, 'b': 10, 't': 10, 'r': 0})
        fig2.update_xaxes(title='Field strength, T')
        fig2.update_yaxes(title='predicted r₂, mM⁻¹s⁻¹')


        graph.layout = html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2)
        ])


        result = pd.DataFrame([xh, mri_r1, mri_r2]).transpose().rename(
            columns={0: 'Field strength, T', 1: 'r1 relaxivity, mM⁻¹s⁻¹', 2: 'r2 relaxivity, mM⁻¹s⁻¹'}).round(
            {'Field strength, T': 2, 'r1 relaxivity, mM⁻¹s⁻¹': 2, 'r2 relaxivity, mM⁻¹s⁻¹': 2})
        import dash_table
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        graph = DjangoDash('pred', external_stylesheets=external_stylesheets)
        graph.layout = html.Div([
            dash_table.DataTable(
                columns=[{"name": i, "id": i, } for i in result.columns],
                data=result.to_dict('records'),
                style_header={'backgroundColor': 'rgb(15, 4, 76)', 'color': 'white', 'font-family': 'Arial', 'fontSize': 15,
                              'fontWeight': 'bold', 'height': 'auto', 'whiteSpace': 'normal',  },
                style_cell={'textAlign': 'left', 'fontSize': 15, 'height': 'auto', 'whiteSpace': 'normal', 'font-family': 'Arial'}, ),
        ])


        data = []
        data.append({'core_comp_0': core_comp_0, 'field_strenght': field_strenght, 'mri_r11': mri_r11, 'mri_r22': mri_r22})
        return render(request, 'prediction/prediction.html', {'data': data})
    if form.is_valid():
        form.save()
    context = {'form': form}
    return render(request, 'prediction/prediction_medium.html', context)






def basic(request):
    form = PredictionFormBasic()
    if request.method == "POST":
        form = PredictionFormBasic(request.POST or None)
        core_comp_0 = str(request.POST['core_comp'])
        length = float(request.POST['length'])

        core_comp = core_comp_0
        core_comp = core_comp + 'O'
        for i in range(0, len(core_comp)):
            if core_comp[i].islower():
                if ((i == 0 or core_comp[i - 1].islower() or core_comp[i - 1].isnumeric()) and
                        core_comp[i + 1].islower() or core_comp[i] == 'o' and not core_comp[i - 1].isupper()):
                    core_comp = core_comp[:i] + core_comp[i].capitalize() + core_comp[i + 1:]
        core_comp = core_comp[:-1]

        if core_comp == 'Fe3O4':
            core_comp = 'Fe1Fe2O4'

        core_comp = ''.join(re.findall(r'[A-Z][a-z]*[\d.]+', re.sub('[A-Z][a-z]*(?![\da-z])', r'\g<0>1', core_comp)))


        core_elements = re.findall('[A-Z][a-z]?[a-z]?', str(core_comp))
        r = core_elements
        for elem in range(len(r) - 1):
            if r[elem] == 'Fe' and r[elem + 1] == 'Fe':
                r[elem] = 'Fe2+'
            elif r[elem] == 'Fe':
                r[elem] = 'Fe3+'
        core_elements = r
        core_stoi = re.findall('[\d.]+', str(core_comp))
        sum = 0
        for i in core_stoi:
            sum += float(i)
        magnetic_moment_core = 0
        spin = 0
        for i in range(len(core_elements)):
            magnetic_moment_core += (magnetic_moments[core_elements[i]] * float(core_stoi[i])) / sum
            spin += (spins[core_elements[i]] * float(core_stoi[i])) / sum


        av = 3 / float(length)
        mm = 1
        LogP = 0
        Hacceptors = 0




        row = {'av': av, 'mm': mm, 'magnetic_moment': magnetic_moment_core, 'sum_surface_spins': spin,
                'squid_sat_mag': np.nan, 'org_coating_LogP': LogP,
                'org_coating_HAcceptors': Hacceptors}

        df_kNN = db_r2.drop(columns=['mri_h_val', 'mri_r2']).append(row, ignore_index=True)

        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=3)
        imputed1 = imputer.fit_transform(df_kNN)
        df_imputed = pd.DataFrame(imputed1, columns=df_kNN.columns)
        sat = round(float(df_imputed[-1:].squid_sat_mag.values), 2)



        av = np.repeat(av, 38)
        mm = np.repeat(mm, 38)
        magnetic_moment_core = np.repeat(magnetic_moment_core, 38)
        spin = np.repeat(spin, 38)
        sat_magn = np.repeat(sat, 38)
        LogP = np.repeat(LogP, 38)
        Hacceptors = np.repeat(Hacceptors, 38)
        strenth = np.arange(0.5, 10, 0.25)
        valh= 'Field strength, T'
        x_pred = pd.DataFrame([av, mm, magnetic_moment_core, spin, sat_magn,
                             LogP, Hacceptors, strenth]).transpose()
        x_pred1 = sc1.transform(x_pred)
        x_pred2 = sc2.transform(x_pred)
        mri_r1 = 10 ** model_mri1.predict(x_pred1)
        mri_r2 = 10 ** model_mri2.predict(x_pred2)
        xh = x_pred.loc[:, 7].values
        pred = pd.DataFrame({'params': valh, 'values': xh, 'predicted R1': mri_r1, 'predicted R2': mri_r2})

        # %% plots
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        graph = DjangoDash('prediction', external_stylesheets=external_stylesheets)

        fig1 = px.line(x=pred['values'], y=pred['predicted R1'], markers=True, line_shape='spline')
        fig1.update_traces(line_color='rgb(193, 39, 45, 0.93)')
        fig1.update_layout(margin={'l': 10, 'b': 10, 't': 10, 'r': 0})
        fig1.update_xaxes(title='Field strength, T')
        fig1.update_yaxes(title='predicted r₁, mM⁻¹s⁻¹')

        fig2 = px.line(x=pred['values'], y=pred['predicted R2'], markers=True, line_shape='spline')
        fig2.update_traces(line_color='rgb(193, 39, 45, 0.93)')
        fig2.update_layout(margin={'l': 10, 'b': 10, 't': 10, 'r': 0})
        fig2.update_xaxes(title='Field strength, T')
        fig2.update_yaxes(title='predicted r₂, mM⁻¹s⁻¹')


        graph.layout = html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2)
        ])


        result = pd.DataFrame([xh, mri_r1, mri_r2]).transpose().rename(
            columns={0: 'Field strength, T', 1: 'r1 relaxivity, mM⁻¹s⁻¹', 2: 'r2 relaxivity, mM⁻¹s⁻¹'}).round(
            {'Field strength, T': 2, 'r1 relaxivity, mM⁻¹s⁻¹': 2, 'r2 relaxivity, mM⁻¹s⁻¹': 2})
        import dash_table
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        graph = DjangoDash('pred', external_stylesheets=external_stylesheets)
        graph.layout = html.Div([
            dash_table.DataTable(
                columns=[{"name": i, "id": i, } for i in result.columns],
                data=result.to_dict('records'),
                style_header={'backgroundColor': 'rgb(15, 4, 76)', 'color': 'white', 'font-family': 'Arial',
                              'fontSize': 15,
                              'fontWeight': 'bold', 'height': 'auto', 'whiteSpace': 'normal', },
                style_cell={'textAlign': 'left', 'fontSize': 15, 'height': 'auto', 'whiteSpace': 'normal',
                            'font-family': 'Arial'}, ),
        ])


        data = []
        data.append({'core_comp_0': core_comp_0})
        return render(request, 'prediction/prediction_basic.html', {'data': data})
    if form.is_valid():
        form.save()
    context = {'form': form}
    return render(request, 'prediction/prediction_basic.html', context)




#SAR



def main_sar(request):
    form = PredictionForm()
    if request.method == "POST":
        form = PredictionForm(request.POST or None)
        core_comp_0 = str(request.POST['core_comp'])
        shape = str(request.POST['shape'])
        length = float(request.POST['length'])
        width = float(request.POST['width'])
        depth = float(request.POST['depth'])
        coat_comp = str(request.POST['coat_comp'])
        smiles = str(request.POST['smiles'])
        sat_magn = float(request.POST['sat_magn'])
        coerc = float(request.POST['coerc'])
        rem_magn = float(request.POST['rem_magn'])
        conc = float(request.POST['conc'])
        field_amp = float(request.POST['field_amp'])
        field_freq = float(request.POST['field_freq'])

        core_comp = core_comp_0
        core_comp = core_comp + 'O'
        for i in range(0, len(core_comp)):
            if core_comp[i].islower():
                if ((i == 0 or core_comp[i - 1].islower() or core_comp[i - 1].isnumeric()) and
                        core_comp[i + 1].islower() or core_comp[i] == 'o' and not core_comp[i - 1].isupper()):
                    core_comp = core_comp[:i] + core_comp[i].capitalize() + core_comp[i + 1:]
        core_comp = core_comp[:-1]

        field_amp *= 12.5663706
        field_freq /= 1000
        if core_comp == 'Fe3O4':
            core_comp = 'Fe1Fe2O4'

        core_comp = ''.join(re.findall(r'[A-Z][a-z]*[\d.]+', re.sub('[A-Z][a-z]*(?![\da-z])', r'\g<0>1', core_comp)))


        core_elements = re.findall('[A-Z][a-z]?', str(core_comp))
        r = core_elements
        for elem in range(len(r) - 1):
            if r[elem] == 'Fe' and r[elem + 1] == 'Fe':
                r[elem] = 'Fe2+'
            elif r[elem] == 'Fe':
                r[elem] = 'Fe3+'
        core_elements = r
        core_stoi = re.findall('[\d.]+', str(core_comp))
        sum = 0
        for i in core_stoi:
            sum += float(i)
        magnetic_moment_core = 0
        for i in range(len(core_elements)):
            magnetic_moment_core += (magnetic_moments[core_elements[i]] * float(core_stoi[i]))/sum



        mm = max(float(length), float(width), float(depth)) / min(float(length), float(width), float(depth))

        if shape == 'Spherical':
            av = 3 / float(length)
        elif shape == 'Cubic':
            av = 6 / float(length)
        elif shape == 'Rod':
            r = min(float(length), float(width), float(depth))
            l = max(float(length), float(width), float(depth))
            av = (2 * r + 2 * l) / (r * l)
        elif shape == 'Rectangle':
            r = min(float(length), float(width), float(depth))
            l = max(float(length), float(width), float(depth))
            av = (2 * r + 4 * l) / (r * l)

        if smiles != '0':
            m = Chem.MolFromSmiles(smiles)
            Hacceptors = NumHAcceptors(m)
            LogP = MolLogP(m)
        else:
            Hacceptors = 0
            LogP = 0

        x_pred0 = pd.DataFrame([conc, av, mm, magnetic_moment_core, sat_magn,
                               coerc, rem_magn, LogP, Hacceptors, field_amp, field_freq]).transpose()

        sar1 = round(10 ** float(model_sar.predict(sc3.transform(x_pred0))), 2)

        conc = np.repeat(conc, 49)
        av = np.repeat(av, 49)
        mm = np.repeat(mm, 49)
        magnetic_moment_core = np.repeat(magnetic_moment_core, 49)
        sat_magn = np.repeat(sat_magn, 49)
        coerc = np.repeat(coerc, 49)
        rem_magn = np.repeat(rem_magn, 49)
        LogP = np.repeat(LogP, 49)
        Hacceptors = np.repeat(Hacceptors, 49)
        field_amp_0 = np.repeat(253, 49)
        field_amp_1 = np.arange(65, 800, 15)
        field_freq_0 = np.repeat(0.3, 49)
        field_amp_20 = np.repeat(20, 41)
        field_freq_300 = np.repeat(300, 49)
        field_freq_1 = np.arange(0.2, 0.49, 0.01)
        x_pred_amp = pd.DataFrame([conc, av, mm, magnetic_moment_core, sat_magn,
                             coerc, rem_magn, LogP, Hacceptors, field_amp_1, field_freq_0]).transpose()
        x_pred_freq = pd.DataFrame([conc, av, mm, magnetic_moment_core, sat_magn,
                             coerc, rem_magn, LogP, Hacceptors, field_amp_0, field_freq_1]).transpose()

        x_amp = sc3.transform(x_pred_amp)
        x_freq = sc3.transform(x_pred_freq)
        sar_amp = 10 ** model_sar.predict(x_amp)
        sar_freq = 10 ** model_sar.predict(x_freq)
        amp = x_pred_amp.loc[:, 9].values/12.5663706
        freq = x_pred_freq.loc[:, 10].values*1000
        pred = pd.DataFrame({'amp': amp, 'freq': freq, 'predicted SAR_amp': sar_amp, 'predicted SAR_freq': sar_freq})

        # %% plots
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        graph = DjangoDash('prediction', external_stylesheets=external_stylesheets)

        fig1 = px.line(x=pred['amp'], y=pred['predicted SAR_amp'], markers=True, line_shape='spline')
        fig1.update_traces(line_color='rgb(193, 39, 45, 0.93)')
        fig1.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0})
        fig1.update_xaxes(title='Field amplitude, kA/m')
        fig1.update_yaxes(title='predicted SAR, W/g')
        fig1.update_layout(margin=dict(l=40, r=40, t=20, b=40), paper_bgcolor="LightSteelBlue")
        fig1.add_annotation(dict(font=dict(size=10),
                                x=0,
                                y=-0.12,
                                showarrow=False,
                                text="Field frequency = 300 kHz",
                                textangle=0,
                                xanchor='left',
                                xref="paper",
                                yref="paper"))

        fig2 = px.line(x=pred['freq'], y=pred['predicted SAR_freq'], markers=True, line_shape='spline')
        fig2.update_traces(line_color='rgb(193, 39, 45, 0.93)')
        fig2.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0})
        fig2.update_xaxes(title='Field frequency, kHz')
        fig2.update_yaxes(title='predicted SAR, W/g')
        fig2.update_layout(margin=dict(l=20, r=20, t=20, b=60),paper_bgcolor="LightSteelBlue")
        fig2.add_annotation(dict(font=dict(size=10),
                                x=0,
                                y=-0.12,
                                showarrow=False,
                                text="Field amplitude = 20 kA/m",
                                textangle=0,
                                xanchor='left',
                                xref="paper",
                                yref="paper"))


        graph.layout = html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2)
        ])


        result1 = pd.DataFrame([amp, field_freq_300, sar_amp]).transpose().rename(
            columns={0: 'Field amplitude, kA/m', 1: 'Field frequency, kHz', 2: 'SAR, W/g'}).round(
            {'Field amplitude, kA/m': 2, 'Field frequency, kHz': 2, 'SAR, W/g': 2})
        import dash_table
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        graph = DjangoDash('pred_amp', external_stylesheets=external_stylesheets)
        graph.layout = html.Div([
            dash_table.DataTable(
                columns=[{"name": i, "id": i, } for i in result1.columns],
                data=result1.to_dict('records'),
                style_header={'backgroundColor': 'rgb(15, 4, 76)', 'color': 'white', 'font-family': 'Arial',
                              'fontSize': 15,
                              'fontWeight': 'bold', 'height': 'auto', 'whiteSpace': 'normal', },
                style_cell={'textAlign': 'left', 'fontSize': 15, 'height': 'auto', 'whiteSpace': 'normal',
                            'font-family': 'Arial'}, ),
        ])


        result2 = pd.DataFrame([field_amp_20, field_freq_1*1000, sar_freq[:-8]]).transpose().rename(
            columns={0: 'Field amplitude, kA/m', 1: 'Field frequency, kHz', 2: 'SAR, W/g'}).round(
            {'Field amplitude, kA/m': 2, 'Field frequency, kHz': 2, 'SAR, W/g': 2})
        import dash_table
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        graph = DjangoDash('pred_freq', external_stylesheets=external_stylesheets)
        graph.layout = html.Div([
            dash_table.DataTable(
                columns=[{"name": i, "id": i, } for i in result2.columns],
                data=result2.to_dict('records'),
                style_header={'backgroundColor': 'rgb(15, 4, 76)', 'color': 'white', 'font-family': 'Arial',
                              'fontSize': 15,
                              'fontWeight': 'bold', 'height': 'auto', 'whiteSpace': 'normal', },
                style_cell={'textAlign': 'left', 'fontSize': 15, 'height': 'auto', 'whiteSpace': 'normal',
                            'font-family': 'Arial'}, ),
        ])


        data = []
        data.append({'core_comp_0': core_comp_0, 'field_amp': round(field_amp/12.5663706, 1), 'field_freq': field_freq*1000, 'sar1': sar1})
        return render(request, 'prediction/prediction_sar.html', {'data': data})
    if form.is_valid():
        form.save()
    context = {'form': form}
    return render(request, 'prediction/prediction_sar.html', context)






def medium_sar(request):
    form = PredictionFormMedium()
    if request.method == "POST":
        form = PredictionFormMedium(request.POST or None)
        core_comp_0 = str(request.POST['core_comp'])
        shape = str(request.POST['shape'])
        length = float(request.POST['length'])
        width = float(request.POST['width'])
        depth = float(request.POST['depth'])
        coat_comp = str(request.POST['coat_comp'])
        smiles = str(request.POST['smiles'])
        conc = float(request.POST['conc'])
        field_amp = float(request.POST['field_amp'])
        field_freq = float(request.POST['field_freq'])

        core_comp = core_comp_0
        core_comp = core_comp + 'O'
        for i in range(0, len(core_comp)):
            if core_comp[i].islower():
                if ((i == 0 or core_comp[i - 1].islower() or core_comp[i - 1].isnumeric()) and
                        core_comp[i + 1].islower() or core_comp[i] == 'o' and not core_comp[i - 1].isupper()):
                    core_comp = core_comp[:i] + core_comp[i].capitalize() + core_comp[i + 1:]
        core_comp = core_comp[:-1]

        field_amp *= 12.5663706
        field_freq /= 1000
        if core_comp == 'Fe3O4':
            core_comp = 'Fe1Fe2O4'

        core_comp = ''.join(re.findall(r'[A-Z][a-z]*[\d.]+', re.sub('[A-Z][a-z]*(?![\da-z])', r'\g<0>1', core_comp)))


        core_elements = re.findall('[A-Z][a-z]?', str(core_comp))
        r = core_elements
        for elem in range(len(r) - 1):
            if r[elem] == 'Fe' and r[elem + 1] == 'Fe':
                r[elem] = 'Fe2+'
            elif r[elem] == 'Fe':
                r[elem] = 'Fe3+'
        core_elements = r
        core_stoi = re.findall('[\d.]+', str(core_comp))
        sum = 0
        for i in core_stoi:
            sum += float(i)
        magnetic_moment_core = 0
        for i in range(len(core_elements)):
            magnetic_moment_core += (magnetic_moments[core_elements[i]] * float(core_stoi[i])) / sum

        mm = max(float(length), float(width), float(depth)) / min(float(length), float(width), float(depth))

        if shape == 'Spherical':
            av = 3 / float(length)
        elif shape == 'Cubic':
            av = 6 / float(length)
        elif shape == 'Rod':
            r = min(float(length), float(width), float(depth))
            l = max(float(length), float(width), float(depth))
            av = (2 * r + 2 * l) / (r * l)
        elif shape == 'Rectangle':
            r = min(float(length), float(width), float(depth))
            l = max(float(length), float(width), float(depth))
            av = (2 * r + 4 * l) / (r * l)

        if smiles != '0':
            m = Chem.MolFromSmiles(smiles)
            Hacceptors = NumHAcceptors(m)
            LogP = MolLogP(m)
        else:
            Hacceptors = 0
            LogP = 0

        row = {'conc': conc, 'av': av, 'mm': mm, 'magnetic_moment': magnetic_moment_core,
                'squid_sat_mag': np.nan, 'squid_coerc_f': 0, 'squid_rem_mag': 0, 'org_coating_LogP': LogP,
                'org_coating_HAcceptors': Hacceptors}

        df_kNN = db_sar.drop(columns=['htherm_h_amp', 'htherm_h_freq', 'sar']).append(row, ignore_index=True)

        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=3)
        imputed1 = imputer.fit_transform(df_kNN)
        df_imputed = pd.DataFrame(imputed1, columns=df_kNN.columns)
        sat = round(float(df_imputed[-1:].squid_sat_mag.values), 2)
        coer = 0
        rem = 0

        x_pred0 = pd.DataFrame([conc, av, mm, magnetic_moment_core, sat,
                               coer, rem, LogP, Hacceptors, field_amp, field_freq]).transpose()

        sar1 = round(10 ** float(model_sar.predict(sc3.transform(x_pred0))), 2)

        conc = np.repeat(conc, 49)
        av = np.repeat(av, 49)
        mm = np.repeat(mm, 49)
        magnetic_moment_core = np.repeat(magnetic_moment_core, 49)
        sat_magn = np.repeat(sat, 49)
        coerc = np.repeat(coer, 49)
        rem_magn = np.repeat(rem, 49)
        LogP = np.repeat(LogP, 49)
        Hacceptors = np.repeat(Hacceptors, 49)
        field_amp_0 = np.repeat(253, 49)
        field_amp_1 = np.arange(65, 800, 15)
        field_freq_0 = np.repeat(0.3, 49)
        field_freq_1 = np.arange(0.2, 0.49, 0.01)
        field_amp_20 = np.repeat(20, 41)
        field_freq_300 = np.repeat(300, 49)
        x_pred_amp = pd.DataFrame([conc, av, mm, magnetic_moment_core, sat_magn,
                                   coerc, rem_magn, LogP, Hacceptors, field_amp_1,
                                   field_freq_0]).transpose()
        x_pred_freq = pd.DataFrame([conc, av, mm, magnetic_moment_core, sat_magn,
                                    coerc, rem_magn, LogP, Hacceptors, field_amp_0,
                                    field_freq_1]).transpose()

        x_amp = sc3.transform(x_pred_amp)
        x_freq = sc3.transform(x_pred_freq)
        sar_amp = 10 ** model_sar.predict(x_amp)
        sar_freq = 10 ** model_sar.predict(x_freq)
        amp = x_pred_amp.loc[:, 9].values/12.5663706
        freq = x_pred_freq.loc[:, 10].values*1000
        pred = pd.DataFrame({'amp': amp, 'freq': freq, 'predicted SAR_amp': sar_amp, 'predicted SAR_freq': sar_freq})

        # %% plots
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        graph = DjangoDash('prediction', external_stylesheets=external_stylesheets)

        fig1 = px.line(x=pred['amp'], y=pred['predicted SAR_amp'], markers=True, line_shape='spline')
        fig1.update_traces(line_color='rgb(193, 39, 45, 0.93)')
        fig1.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0})
        fig1.update_xaxes(title='Field amplitude, kA/m')
        fig1.update_yaxes(title='predicted SAR, W/g')
        fig1.update_layout(margin=dict(l=20, r=20, t=20, b=60), paper_bgcolor="LightSteelBlue")
        fig1.add_annotation(dict(font=dict(size=10),
                                 x=0,
                                 y=-0.12,
                                 showarrow=False,
                                 text="Field frequency = 300 kHz",
                                 textangle=0,
                                 xanchor='left',
                                 xref="paper",
                                 yref="paper"))

        fig2 = px.line(x=pred['freq'], y=pred['predicted SAR_freq'], markers=True, line_shape='spline')
        fig2.update_traces(line_color='rgb(193, 39, 45, 0.93)')
        fig2.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0})
        fig2.update_xaxes(title='Field frequency, kHz')
        fig2.update_yaxes(title='predicted SAR, W/g')
        fig2.update_layout(margin=dict(l=20, r=20, t=20, b=60), paper_bgcolor="LightSteelBlue")
        fig2.add_annotation(dict(font=dict(size=10),
                                 x=0,
                                 y=-0.12,
                                 showarrow=False,
                                 text="Field amplitude = 20 kA/m",
                                 textangle=0,
                                 xanchor='left',
                                 xref="paper",
                                 yref="paper"))

        graph.layout = html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2)
        ])


        result1 = pd.DataFrame([amp, field_freq_300, sar_amp]).transpose().rename(
            columns={0: 'Field amplitude, kA/m', 1: 'Field frequency, kHz', 2: 'SAR, W/g'}).round(
            {'Field amplitude, kA/m': 2, 'Field frequency, kHz': 2, 'SAR, W/g': 2})
        import dash_table
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        graph = DjangoDash('pred_amp', external_stylesheets=external_stylesheets)
        graph.layout = html.Div([
            dash_table.DataTable(
                columns=[{"name": i, "id": i, } for i in result1.columns],
                data=result1.to_dict('records'),
                style_header={'backgroundColor': 'rgb(15, 4, 76)', 'color': 'white', 'font-family': 'Arial',
                              'fontSize': 15,
                              'fontWeight': 'bold', 'height': 'auto', 'whiteSpace': 'normal', },
                style_cell={'textAlign': 'left', 'fontSize': 15, 'height': 'auto', 'whiteSpace': 'normal',
                            'font-family': 'Arial'}, ),
        ])


        result2 = pd.DataFrame([field_amp_20, field_freq_1*1000, sar_freq[:-8]]).transpose().rename(
            columns={0: 'Field amplitude, kA/m', 1: 'Field frequency, kHz', 2: 'SAR, W/g'}).round(
            {'Field amplitude, kA/m': 2, 'Field frequency, kHz': 2, 'SAR, W/g': 2})
        import dash_table
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        graph = DjangoDash('pred_freq', external_stylesheets=external_stylesheets)
        graph.layout = html.Div([
            dash_table.DataTable(
                columns=[{"name": i, "id": i, } for i in result2.columns],
                data=result2.to_dict('records'),
                style_header={'backgroundColor': 'rgb(15, 4, 76)', 'color': 'white', 'font-family': 'Arial',
                              'fontSize': 15,
                              'fontWeight': 'bold', 'height': 'auto', 'whiteSpace': 'normal', },
                style_cell={'textAlign': 'left', 'fontSize': 15, 'height': 'auto', 'whiteSpace': 'normal',
                            'font-family': 'Arial'}, ),
        ])



        data = []
        data.append({'core_comp_0': core_comp_0, 'field_amp': round(field_amp/12.5663706, 1), 'field_freq': field_freq*1000, 'sar1': sar1})
        return render(request, 'prediction/prediction_medium_sar.html', {'data': data})
    if form.is_valid():
        form.save()
    context = {'form': form}
    return render(request, 'prediction/prediction_medium_sar.html', context)






def basic_sar(request):
    form = PredictionFormBasic()
    if request.method == "POST":
        form = PredictionFormBasic(request.POST or None)
        core_comp_0 = str(request.POST['core_comp'])
        length = float(request.POST['length'])
        conc = 1
        field_amp = 253
        field_freq = 0.300

        core_comp = core_comp_0
        core_comp = core_comp + 'O'
        for i in range(0, len(core_comp)):
            if core_comp[i].islower():
                if ((i == 0 or core_comp[i - 1].islower() or core_comp[i - 1].isnumeric()) and
                        core_comp[i + 1].islower() or core_comp[i] == 'o' and not core_comp[i - 1].isupper()):
                    core_comp = core_comp[:i] + core_comp[i].capitalize() + core_comp[i + 1:]
        core_comp = core_comp[:-1]

        if core_comp == 'Fe3O4':
            core_comp = 'Fe1Fe2O4'

        core_comp = ''.join(re.findall(r'[A-Z][a-z]*[\d.]+', re.sub('[A-Z][a-z]*(?![\da-z])', r'\g<0>1', core_comp)))


        core_elements = re.findall('[A-Z][a-z]?', str(core_comp))
        r = core_elements
        for elem in range(len(r) - 1):
            if r[elem] == 'Fe' and r[elem + 1] == 'Fe':
                r[elem] = 'Fe2+'
            elif r[elem] == 'Fe':
                r[elem] = 'Fe3+'
        core_elements = r
        core_stoi = re.findall('[\d.]+', str(core_comp))
        sum = 0
        for i in core_stoi:
            sum += float(i)
        magnetic_moment_core = 0
        for i in range(len(core_elements)):
            magnetic_moment_core += (magnetic_moments[core_elements[i]] * float(core_stoi[i])) / sum



        av = 3 / float(length)
        mm = 1
        LogP = 0
        Hacceptors = 0




        row = {'conc': conc, 'av': av, 'mm': mm, 'magnetic_moment': magnetic_moment_core,
                'squid_sat_mag': np.nan, 'squid_coerc_f': 0, 'squid_rem_mag': 0,
                'org_coating_LogP': LogP,
                'org_coating_HAcceptors': Hacceptors}

        df_kNN = db_r2.drop(columns=['mri_h_val', 'mri_r2']).append(row, ignore_index=True)

        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=3)
        imputed1 = imputer.fit_transform(df_kNN)
        df_imputed = pd.DataFrame(imputed1, columns=df_kNN.columns)
        sat = round(float(df_imputed[-1:].squid_sat_mag.values), 2)
        coer = 0
        rem = 0

        x_pred0 = pd.DataFrame([conc, av, mm, magnetic_moment_core, sat,
                               coer, rem, LogP, Hacceptors, field_amp, field_freq]).transpose()

        sar1 = round(10 ** float(model_sar.predict(sc3.transform(x_pred0))), 2)

        conc = np.repeat(conc, 49)
        av = np.repeat(av, 49)
        mm = np.repeat(mm, 49)
        magnetic_moment_core = np.repeat(magnetic_moment_core, 49)
        sat_magn = np.repeat(sat, 49)
        coerc = np.repeat(coer, 49)
        rem_magn = np.repeat(rem, 49)
        LogP = np.repeat(LogP, 49)
        Hacceptors = np.repeat(Hacceptors, 49)
        field_amp_0 = np.repeat(253, 49)
        field_amp_1 = np.arange(65, 800, 15)
        field_freq_0 = np.repeat(0.3, 49)
        field_freq_1 = np.arange(0.2, 0.49, 0.01)
        field_amp_20 = np.repeat(20, 41)
        field_freq_300 = np.repeat(300, 49)
        x_pred_amp = pd.DataFrame([conc, av, mm, magnetic_moment_core, sat_magn,
                                   coerc, rem_magn, LogP, Hacceptors, field_amp_1,
                                   field_freq_0]).transpose()
        x_pred_freq = pd.DataFrame([conc, av, mm, magnetic_moment_core, sat_magn,
                                    coerc, rem_magn, LogP, Hacceptors, field_amp_0,
                                    field_freq_1]).transpose()

        x_amp = sc3.transform(x_pred_amp)
        x_freq = sc3.transform(x_pred_freq)
        sar_amp = 10 ** model_sar.predict(x_amp)
        sar_freq = 10 ** model_sar.predict(x_freq)
        amp = x_pred_amp.loc[:, 9].values/12.5663706
        freq = x_pred_freq.loc[:, 10].values*1000
        pred = pd.DataFrame({'amp': amp, 'freq': freq, 'predicted SAR_amp': sar_amp, 'predicted SAR_freq': sar_freq})

        # %% plots
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        graph = DjangoDash('prediction', external_stylesheets=external_stylesheets)

        fig1 = px.line(x=pred['amp'], y=pred['predicted SAR_amp'], markers=True, line_shape='spline')
        fig1.update_traces(line_color='rgb(193, 39, 45, 0.93)')
        fig1.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0})
        fig1.update_xaxes(title='Field amplitude, kA/m')
        fig1.update_yaxes(title='predicted SAR, W/g')
        fig1.update_layout(margin=dict(l=20, r=20, t=20, b=60), paper_bgcolor="LightSteelBlue")
        fig1.add_annotation(dict(font=dict(size=10),
                                 x=0,
                                 y=-0.12,
                                 showarrow=False,
                                 text="Field frequency = 300 kHz",
                                 textangle=0,
                                 xanchor='left',
                                 xref="paper",
                                 yref="paper"))

        fig2 = px.line(x=pred['freq'], y=pred['predicted SAR_freq'], markers=True, line_shape='spline')
        fig2.update_traces(line_color='rgb(193, 39, 45, 0.93)')
        fig2.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0})
        fig2.update_xaxes(title='Field frequency, kHz')
        fig2.update_yaxes(title='predicted SAR, W/g')
        fig2.update_layout(margin=dict(l=20, r=20, t=20, b=60), paper_bgcolor="LightSteelBlue")
        fig2.add_annotation(dict(font=dict(size=10),
                                 x=0,
                                 y=-0.12,
                                 showarrow=False,
                                 text="Field amplitude = 20 kA/m",
                                 textangle=0,
                                 xanchor='left',
                                 xref="paper",
                                 yref="paper"))

        graph.layout = html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2)
        ])


        result1 = pd.DataFrame([amp, field_freq_300, sar_amp]).transpose().rename(
            columns={0: 'Field amplitude, kA/m', 1: 'Field frequency, kHz', 2: 'SAR, W/g'}).round(
            {'Field amplitude, kA/m': 2, 'Field frequency, kHz': 2, 'SAR, W/g': 2})
        import dash_table
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        graph = DjangoDash('pred_amp', external_stylesheets=external_stylesheets)
        graph.layout = html.Div([
            dash_table.DataTable(
                columns=[{"name": i, "id": i, } for i in result1.columns],
                data=result1.to_dict('records'),
                style_header={'backgroundColor': 'rgb(15, 4, 76)', 'color': 'white', 'font-family': 'Arial',
                              'fontSize': 15,
                              'fontWeight': 'bold', 'height': 'auto', 'whiteSpace': 'normal', },
                style_cell={'textAlign': 'left', 'fontSize': 15, 'height': 'auto', 'whiteSpace': 'normal',
                            'font-family': 'Arial'}, ),
        ])


        result2 = pd.DataFrame([field_amp_20, field_freq_1*1000, sar_freq[:-8]]).transpose().rename(
            columns={0: 'Field amplitude, kA/m', 1: 'Field frequency, kHz', 2: 'SAR, W/g'}).round(
            {'Field amplitude, kA/m': 2, 'Field frequency, kHz': 2, 'SAR, W/g': 2})
        import dash_table
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        graph = DjangoDash('pred_freq', external_stylesheets=external_stylesheets)
        graph.layout = html.Div([
            dash_table.DataTable(
                columns=[{"name": i, "id": i, } for i in result2.columns],
                data=result2.to_dict('records'),
                style_header={'backgroundColor': 'rgb(15, 4, 76)', 'color': 'white', 'font-family': 'Arial',
                              'fontSize': 15,
                              'fontWeight': 'bold', 'height': 'auto', 'whiteSpace': 'normal', },
                style_cell={'textAlign': 'left', 'fontSize': 15, 'height': 'auto', 'whiteSpace': 'normal',
                            'font-family': 'Arial'}, ),
        ])


        data = []
        data.append({'core_comp_0': core_comp_0})
        return render(request, 'prediction/prediction_basic_sar.html', {'data': data})
    if form.is_valid():
        form.save()
    context = {'form': form}
    return render(request, 'prediction/prediction_basic_sar.html', context)