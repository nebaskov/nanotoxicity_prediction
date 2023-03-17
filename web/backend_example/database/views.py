from django.shortcuts import render, redirect
from .models import nanoparticle_mri_r1, nanoparticle_mri_r2, nanoparticle_sar
from dash import dash_table
import pandas as pd
from django_plotly_dash import DjangoDash
from dash.dependencies import Input, Output
from .forms import offerForm
from django.views.decorators.csrf import csrf_exempt



def sar(request):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    df_sar_table = pd.DataFrame(list(
    nanoparticle_sar.objects.values('np_core', 'np_shell_1', 'htherm_h_amp', 'htherm_h_freq', 'sar', 'paper_doi')))
    def f(row):
        return "[{0}]({0})".format(row["paper_doi"])

    df_sar_table['paper_doi'] = df_sar_table.apply(f, axis=1)

    table_sar = DjangoDash('Table_3', external_stylesheets=external_stylesheets)
    table_sar.layout = dash_table.DataTable(
        data=df_sar_table.to_dict('records'),
        sort_action='native',
        filter_action="native",
        columns=[{'name': 'Chemical formula', 'id':'np_core', 'type':'text','presentation':'markdown'},
                 {'name': 'Surface composition', 'id':'np_shell_1', 'type':'text','presentation':'markdown'},
                 {'name': 'Field amplitude, Oe', 'id': 'htherm_h_amp', 'type': 'text', 'presentation': 'markdown'},
                 {'name': 'Field frequency, MHz', 'id': 'htherm_h_freq', 'type': 'text', 'presentation': 'markdown'},
                 {'name': 'SAR, W/g', 'id':'sar', 'type':'numeric','presentation':'markdown'},
                 {'name': 'Link to the article', 'id':'paper_doi','type':'text','presentation':'markdown'}],
        page_size=14,
        fixed_rows={'headers': True},
        markdown_options={'link_target': '_parent'},
        style_header={'backgroundColor': 'rgb(15, 4, 76)', 'color': 'white', 'font-family': 'Arial',
                      'fontSize': 15,
                      'fontWeight': 'bold', 'height': 'auto', 'whiteSpace': 'normal', },
        style_cell={'textAlign': 'left', 'fontSize': 15, 'height': 'auto', 'whiteSpace': 'normal',
                    'font-family': 'Arial'},
        style_data_conditional=[
            {'if': {'column_id': 'np_core'},
             'width': '120px'},
            {'if': {'column_id': 'np_shell_1'},
             'width': '120px'},
            {'if': {'column_id': 'htherm_h_amp'},
             'width': '120px'},
            {'if': {'column_id': 'htherm_h_freq'},
             'width': '120px'},
            {'if': {'column_id': 'sar'},
             'width': '120px'},
            {'if': {'column_id': 'paper_doi'},
             'width': '180px'},
           ],
        sort_mode="multi",
    )

    import plotly.express as px
    from dash import html
    from dash import dcc
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    dash = DjangoDash('New', external_stylesheets=external_stylesheets)
    df_sar_vis = pd.DataFrame(list(nanoparticle_sar.objects.values()))
    df_sar_vis = df_sar_vis.loc[:, 'np_core':'sar'].rename(columns={'conc': 'Concentration of nanoparticles, mg/ml',
                                                            'av': 'Area/volume', 'mm': 'Max/min(size)',
                                                            'magnetic_moment': 'Magnetic moment of core, Borh magneton',
                                                            'squid_sat_mag': 'Saturation magnetization, emu/g',
                                                            'squid_coerc_f': 'Coercivity, Oe',
                                                            'squid_rem_mag': 'Remanent magnetization, emu/g',
                                                            'org_coating_LogP': 'LogP of organic coating',
                                                            'org_coating_HAcceptors': 'Number H acceptors of organic coating',
                                                            'htherm_h_amp': 'Field amplitude, Oe',
                                                            'htherm_h_freq': 'Field frequency, MHz',
                                                            'sar': 'SAR, W/g'}).drop('np_shell_1', axis=1)

    dash.layout = html.Div([
        html.Div([
        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in df_sar_vis.drop('np_core', axis=1).columns],
                value='Area/volume',
                clearable=False, placeholder="Select x-axis value"
            ),
            dcc.RadioItems(
                id='xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
            style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in df_sar_vis.drop('np_core', axis=1).columns],
                value='SAR, W/g',
                clearable=False, placeholder="Select y-axis value"
            ),
            dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    dcc.Graph(id='indicator-graphic'),
    ])

    @dash.callback(
        Output('indicator-graphic', 'figure'),
        Input('xaxis-column', 'value'),
        Input('yaxis-column', 'value'),
        Input('xaxis-type', 'value'),
        Input('yaxis-type', 'value'),
    )
    def update_graph(xaxis_column_name3, yaxis_column_name3,
                     xaxis_type3, yaxis_type3, ):

        fig = px.scatter(x=df_sar_vis[xaxis_column_name3],
                          y=df_sar_vis[yaxis_column_name3],
                          hover_name=df_sar_vis['np_core'],
                          )
        fig.update_traces(marker=dict(color='rgb(193, 39, 45, 0.93)'))
        fig.update_layout({'margin': {'l': 40, 'b': 40, 't': 10, 'r': 0}, "legend_orientation": "h",
                            'paper_bgcolor': 'rgba(0, 0, 0,0)', 'plot_bgcolor': 'rgba(0, 0, 0,0)'})
        fig.update_xaxes(title=xaxis_column_name3,
                          type='linear' if xaxis_type3 == 'Linear' else 'log')
        fig.update_yaxes(title=yaxis_column_name3,
                          type='linear' if yaxis_type3 == 'Linear' else 'log')

        return fig

    return render(request, 'database/sar.html')




def r1(request):
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        df_1_table = pd.DataFrame(list(nanoparticle_mri_r1.objects.values('np_core', 'np_shell_1', 'mri_h_val', 'mri_r1', 'paper_doi')))

        def f(row):
            return "[{0}]({0})".format(row["paper_doi"])

        df_1_table['paper_doi'] = df_1_table.apply(f, axis=1)
        table_r1 = DjangoDash('Table_1', external_stylesheets=external_stylesheets)
        table_r1.layout = dash_table.DataTable(
            data=df_1_table.to_dict('records'),
            sort_action='native',
            filter_action="native",
            columns=[{'name': 'Chemical formula', 'id': 'np_core', 'type': 'text', 'presentation': 'markdown'},
                     {'name': 'Surface composition', 'id': 'np_shell_1', 'type': 'text', 'presentation': 'markdown'},
                     {'name': 'Field strength, T', 'id': 'mri_h_val', 'type': 'numeric', 'presentation': 'markdown'},
                     {'name': 'r₁ relaxivity, mM⁻¹s⁻¹', 'id': 'mri_r1', 'type': 'numeric', 'presentation': 'markdown'},
                     {'name': 'Link to the article', 'id': 'paper_doi', 'type': 'text', 'presentation': 'markdown'}],
            page_size=14,
            fixed_rows={'headers': True},
            markdown_options={'link_target': '_parent'},
            style_header={'backgroundColor': 'rgb(15, 4, 76)', 'color': 'white', 'font-family': 'Arial',
                          'fontSize': 15,
                          'fontWeight': 'bold', 'height': 'auto', 'whiteSpace': 'normal', },
            style_cell={'textAlign': 'left', 'fontSize': 15, 'height': 'auto', 'whiteSpace': 'normal',
                        'font-family': 'Arial'},
            style_data_conditional=[
                {'if': {'column_id': 'np_core'},
                 'width': '150px'},
                {'if': {'column_id': 'np_shell_1'},
                 'width': '150px'},
                {'if': {'column_id': 'mri_h_val'},
                 'width': '150px'},
                {'if': {'column_id': 'mri_r1'},
                 'width': '150px'},
                {'if': {'column_id': 'paper_doi'},
                 'width': '180px'},
            ],
            sort_mode="multi",
        )

        import plotly.express as px
        from dash import html
        from dash import dcc

        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

        dash = DjangoDash('New', external_stylesheets=external_stylesheets)
        df_1_vis = pd.DataFrame(list(nanoparticle_mri_r1.objects.values()))
        df_1_vis = df_1_vis.loc[:, 'np_core':'mri_r1'].rename(columns={'av': 'Area/volume', 'mm': 'Max/min(size)',
                                                               'magnetic_moment': 'Magnetic moment of core, Borh magneton',
                                                               'sum_surface_spins': 'Spin of surface elements',
                                                               'squid_sat_mag': 'Saturation magnetization, emu/g',
                                                               'org_coating_LogP': 'LogP of organic coating',
                                                               'org_coating_HAcceptors': 'Number H acceptors of organic coating',
                                                               'mri_h_val': 'Field strength, T',
                                                               'mri_r1': 'r₁ relaxivity, mM⁻¹s⁻¹'}).drop('np_shell_1', axis=1)

        dash.layout = html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='xaxis-column',
                        options=[{'label': i, 'value': i} for i in df_1_vis.drop('np_core', axis=1).columns],
                        value='Area/volume',
                        clearable=False, placeholder="Select x-axis value"
                    ),
                    dcc.RadioItems(
                        id='xaxis-type',
                        options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                ],
                    style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

                html.Div([
                    dcc.Dropdown(
                        id='yaxis-column',
                        options=[{'label': i, 'value': i} for i in df_1_vis.drop('np_core', axis=1).columns],
                        value='r₁ relaxivity, mM⁻¹s⁻¹',
                        clearable=False, placeholder="Select y-axis value"
                    ),
                    dcc.RadioItems(
                        id='yaxis-type',
                        options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                ], style={'width': '48%', 'display': 'inline-block'})
            ]),
            dcc.Graph(id='indicator-graphic'),
        ])

        @dash.callback(
            Output('indicator-graphic', 'figure'),
            Input('xaxis-column', 'value'),
            Input('yaxis-column', 'value'),
            Input('xaxis-type', 'value'),
            Input('yaxis-type', 'value'),
        )
        def update_graph(xaxis_column_name1, yaxis_column_name1,
                         xaxis_type1, yaxis_type1):
            fig = px.scatter(x=df_1_vis[xaxis_column_name1],
                              y=df_1_vis[yaxis_column_name1],
                              hover_name=df_1_vis['np_core']
                              )
            fig.update_traces(marker=dict(color='rgb(193, 39, 45, 0.93)'))
            fig.update_layout({'margin': {'l': 40, 'b': 40, 't': 10, 'r': 0}, "legend_orientation": "h",
                                'paper_bgcolor': 'rgba(0, 0, 0,0)', 'plot_bgcolor': 'rgba(0, 0, 0,0)'})
            fig.update_xaxes(title=xaxis_column_name1,
                              type='linear' if xaxis_type1 == 'Linear' else 'log')
            fig.update_yaxes(title=yaxis_column_name1,
                              type='linear' if yaxis_type1 == 'Linear' else 'log')
            return fig

        return render(request, 'database/r1.html')


def r2(request):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    df_2_table = pd.DataFrame(
        list(nanoparticle_mri_r2.objects.values('np_core', 'np_shell_1', 'mri_h_val', 'mri_r2', 'paper_doi')))

    def f(row):
        return "[{0}]({0})".format(row["paper_doi"])

    df_2_table['paper_doi'] = df_2_table.apply(f, axis=1)
    table_r2 = DjangoDash('Table_2', external_stylesheets=external_stylesheets)
    table_r2.layout = dash_table.DataTable(
        data=df_2_table.to_dict('records'),
        sort_action='native',
        filter_action="native",
        columns=[{'name': 'Chemical formula', 'id': 'np_core', 'type': 'text', 'presentation': 'markdown'},
                 {'name': 'Surface composition', 'id': 'np_shell_1', 'type': 'text', 'presentation': 'markdown'},
                 {'name': 'Field strength, T', 'id': 'mri_h_val', 'type': 'numeric', 'presentation': 'markdown'},
                 {'name': 'r₂ relaxivity, mM⁻¹s⁻¹', 'id': 'mri_r2', 'type': 'numeric', 'presentation': 'markdown'},
                 {'name': 'Link to the article', 'id': 'paper_doi', 'type': 'text', 'presentation': 'markdown'}],
        page_size=14,
        fixed_rows={'headers': True},
        markdown_options={'link_target': '_parent'},
        style_header={'backgroundColor': 'rgb(15, 4, 76)', 'color': 'white', 'font-family': 'Arial',
                      'fontSize': 15,
                      'fontWeight': 'bold', 'height': 'auto', 'whiteSpace': 'normal', },
        style_cell={'textAlign': 'left', 'fontSize': 15, 'height': 'auto', 'whiteSpace': 'normal',
                    'font-family': 'Arial'},
        style_data_conditional=[
            {'if': {'column_id': 'np_core'},
             'width': '150px'},
            {'if': {'column_id': 'np_shell_1'},
             'width': '150px'},
            {'if': {'column_id': 'mri_h_val'},
             'width': '150px'},
            {'if': {'column_id': 'mri_r2'},
             'width': '150px'},
            {'if': {'column_id': 'paper_doi'},
             'width': '180px'},
        ],
        sort_mode="multi",
    )

    import plotly.express as px
    from dash import html
    from dash import dcc

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    dash = DjangoDash('New', external_stylesheets=external_stylesheets)
    df_2_vis = pd.DataFrame(list(nanoparticle_mri_r2.objects.values()))
    df_2_vis = df_2_vis.loc[:, 'np_core':'mri_r2'].rename(columns={'av': 'Area/volume', 'mm': 'Max/min(size)',
                                                                   'magnetic_moment': 'Magnetic moment of core, Borh magneton',
                                                                   'sum_surface_spins': 'Spin of surface elements',
                                                                   'squid_sat_mag': 'Saturation magnetization, emu/g',
                                                                   'org_coating_LogP': 'LogP of organic coating',
                                                                   'org_coating_HAcceptors': 'Number H acceptors of organic coating',
                                                                   'mri_h_val': 'Field strength, T',
                                                                   'mri_r2': 'r₂ relaxivity, mM⁻¹s⁻¹'}).drop('np_shell_1', axis=1)

    dash.layout = html.Div([
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='xaxis-column',
                    options=[{'label': i, 'value': i} for i in df_2_vis.drop('np_core', axis=1).columns],
                    value='Area/volume',
                    clearable=False, placeholder="Select x-axis value"
                ),
                dcc.RadioItems(
                    id='xaxis-type',
                    options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                    value='Linear',
                    labelStyle={'display': 'inline-block'}
                )
            ],
                style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

            html.Div([
                dcc.Dropdown(
                    id='yaxis-column',
                    options=[{'label': i, 'value': i} for i in df_2_vis.drop('np_core', axis=1).columns],
                    value='r₂ relaxivity, mM⁻¹s⁻¹',
                    clearable=False, placeholder="Select y-axis value"
                ),
                dcc.RadioItems(
                    id='yaxis-type',
                    options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                    value='Linear',
                    labelStyle={'display': 'inline-block'}
                )
            ], style={'width': '48%', 'display': 'inline-block'})
        ]),
        dcc.Graph(id='indicator-graphic'),
    ])

    @dash.callback(
        Output('indicator-graphic', 'figure'),
        Input('xaxis-column', 'value'),
        Input('yaxis-column', 'value'),
        Input('xaxis-type', 'value'),
        Input('yaxis-type', 'value'),
    )
    def update_graph(xaxis_column_name1, yaxis_column_name1,
                     xaxis_type1, yaxis_type1):
        fig = px.scatter(x=df_2_vis[xaxis_column_name1],
                          y=df_2_vis[yaxis_column_name1],
                          hover_name=df_2_vis['np_core']
                          )
        fig.update_traces(marker=dict(color='rgb(193, 39, 45, 0.93)'))
        fig.update_layout({'margin': {'l': 40, 'b': 40, 't': 10, 'r': 0}, "legend_orientation": "h",
                            'paper_bgcolor': 'rgba(0, 0, 0,0)', 'plot_bgcolor': 'rgba(0, 0, 0,0)'})
        fig.update_xaxes(title=xaxis_column_name1,
                          type='linear' if xaxis_type1 == 'Linear' else 'log')
        fig.update_yaxes(title=yaxis_column_name1,
                          type='linear' if yaxis_type1 == 'Linear' else 'log')
        return fig

    return render(request, 'database/r2.html')

@csrf_exempt
def offer(request):
    form = offerForm()
    if request.method == 'POST':
        form = offerForm(request.POST or None)
        if form.is_valid():
            form.save()
            return redirect('database')
    data = {
        'form': form}
    return render(request, 'database/offer.html', data)