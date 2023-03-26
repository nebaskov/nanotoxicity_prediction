from django.shortcuts import render, redirect
from .models import NpDb 
from dash import dash_table
import pandas as pd
from django_plotly_dash import DjangoDash
from dash.dependencies import Input, Output
from .forms import offerForm
from django.views.decorators.csrf import csrf_exempt


def get_db(request):
    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    np_db = pd.DataFrame(list(
    NpDb.objects.values("material", "diameter", "zeta",
                        "electroneg", "ionic_rad", "concentr",
                        "mol_weight", "time",
                        "cell_type", "line_primary_cell",
                        "animal", "cell_organ",
                        "cell_age", "cell_organ",
                        "test", "test_indicator",
                        "viability")))

    table_np = DjangoDash("Table_1", external_stylesheets=external_stylesheets)
    table_np.layout = dash_table.DataTable(
        data=np_db.to_dict("records"),
        sort_action="native",
        filter_action="native",
        columns=[{"name": "NP chemical formula",
                  "id": "material", 
                  "type": "text", 
                  "presentation": "markdown"},
                
                 {"name": "Molecular weight (g/mol)",
                  "id": "mol_weight",
                  "type": "numeric", 
                  "presentation": "markdown"},
                 
                 {"name": "Electronegativity", 
                  "id": "electroneg", 
                  "type": "numeric", 
                  "presentation": "markdown"},
                 
                 {"name": "Ionic radius", 
                  "id": "ionic_rad", 
                  "type": "numeric", 
                  "presentation": "markdown"},
                 
                 {"name": "Zeta potential (mV)", 
                  "id": "zeta", 
                  "type": "numeric", 
                  "presentation": "markdown"},
                 
                 {"name": "Concentration (g/L)", 
                  "id": "concentr", 
                  "type": "numeric", 
                  "presentation": "markdown"},
                 
                 {"name": "Time (h)", 
                  "id": "time", 
                  "type": "numeric", 
                  "presentation": "markdown"},
                 
                 {"name": "Cell type", 
                  "id": "cell_type", 
                  "type": "text", 
                  "presentation": "markdown"},
                 
                 {"name": "Line primary cell", 
                  "id": "line_primary_cell", 
                  "type": "text", 
                  "presentation": "markdown"},
                 
                 {"name": "Cell source", 
                  "id": "animal", 
                  "type": "text", 
                  "presentation": "markdown"},
                 
                 {"name": "Cell morphology", 
                  "id": "cell_morphology", 
                  "type": "text", 
                  "presentation": "markdown"},
                 
                 {"name": "Cell age", 
                  "id": "cell_age", 
                  "type": "text", 
                  "presentation": "markdown"},
                                 
                 {"name": "Cell organ", 
                  "id": "cell_organ", 
                  "type": "text", 
                  "presentation": "markdown"},
                 
                 {"name": "Test", 
                  "id": "test", 
                  "type": "text", 
                  "presentation": "markdown"},
                 
                 {"name": "Test indicator", 
                  "id": "test_indicator", 
                  "type": "text", 
                  "presentation": "markdown"},
                 
                 {"name": "Viability (%)", 
                  "id": "viability", 
                  "type": "numeric", 
                  "presentation": "markdown"},
                 ],
        
        page_size=14,
        fixed_rows={"headers": True},
        markdown_options={"link_target": "_parent"},
        style_header={"backgroundColor": "rgb(15, 4, 76)", "color": "white", "font-family": "Arial",
                      "fontSize": 15,
                      "fontWeight": "bold", "height": "auto", "whiteSpace": "normal", },
        style_cell={"textAlign": "left", "fontSize": 15, "height": "auto", "whiteSpace": "normal",
                    "font-family": "Arial"},
        style_data_conditional=[
            {"if": {"column_id": "material"},
             "width": "120px"},
            {"if": {"column_id": "diameter"},
             "width": "120px"},
            {"if": {"column_id": "zeta"},
             "width": "120px"},
            {"if": {"column_id": "electroneg"},
             "width": "120px"},
            {"if": {"column_id": "ionic_rad"},
             "width": "120px"},
            {"if": {"column_id": "mol_weight"},
             "width": "120px"},
            {"if": {"column_id": "time"},
             "width": "120px"},
            {"if": {"column_id": "cell_type"},
             "width": "120px"},
            {"if": {"column_id": "line_primary_cell"},
             "width": "120px"},
            {"if": {"column_id": "animal"},
             "width": "120px"},
            {"if": {"column_id": "cell_organ"},
             "width": "120px"},
            {"if": {"column_id": "cell_age"},
             "width": "120px"},
            {"if": {"column_id": "cell_organ"},
             "width": "120px"},
            {"if": {"column_id": "test"},
             "width": "120px"},
            {"if": {"column_id": "test_indicator"},
             "width": "120px"},
            {"if": {"column_id": "test_indicator"},
             "width": "120px"},
          ],
        sort_mode="multi",
    )

    import plotly.express as px
    from dash import html
    from dash import dcc
    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    dash = DjangoDash("New", external_stylesheets=external_stylesheets)
    df_sar_vis = pd.DataFrame(list(NpDb.objects.values()))
    df_sar_vis = df_sar_vis.loc[:, "np_core":"sar"].rename(columns={"conc": "Concentration of nanoparticles, mg/ml",
                                                            "av": "Area/volume", "mm": "Max/min(size)",
                                                            "magnetic_moment": "Magnetic moment of core, Borh magneton",
                                                            "squid_sat_mag": "Saturation magnetization, emu/g",
                                                            "squid_coerc_f": "Coercivity, Oe",
                                                            "squid_rem_mag": "Remanent magnetization, emu/g",
                                                            "org_coating_LogP": "LogP of organic coating",
                                                            "org_coating_HAcceptors": "Number H acceptors of organic coating",
                                                            "htherm_h_amp": "Field amplitude, Oe",
                                                            "htherm_h_freq": "Field frequency, MHz",
                                                            "sar": "SAR, W/g"}).drop("np_shell_1", axis=1)

    dash.layout = html.Div([
        html.Div([
        html.Div([
            dcc.Dropdown(
                id="xaxis-column",
                options=[{"label": i, "value": i} for i in df_sar_vis.drop("np_core", axis=1).columns],
                value="Area/volume",
                clearable=False, placeholder="Select x-axis value"
            ),
            dcc.RadioItems(
                id="xaxis-type",
                options=[{"label": i, "value": i} for i in ["Linear", "Log"]],
                value="Linear",
                labelStyle={"display": "inline-block"}
            )
        ],
            style={"width": "48%", "float": "right", "display": "inline-block"}),

        html.Div([
            dcc.Dropdown(
                id="yaxis-column",
                options=[{"label": i, "value": i} for i in df_sar_vis.drop("np_core", axis=1).columns],
                value="SAR, W/g",
                clearable=False, placeholder="Select y-axis value"
            ),
            dcc.RadioItems(
                id="yaxis-type",
                options=[{"label": i, "value": i} for i in ["Linear", "Log"]],
                value="Linear",
                labelStyle={"display": "inline-block"}
            )
        ], style={"width": "48%", "display": "inline-block"})
    ]),
    dcc.Graph(id="indicator-graphic"),
    ])

    @dash.callback(
        Output("indicator-graphic", "figure"),
        Input("xaxis-column", "value"),
        Input("yaxis-column", "value"),
        Input("xaxis-type", "value"),
        Input("yaxis-type", "value"),
    )
    def update_graph(xaxis_column_name3, yaxis_column_name3,
                     xaxis_type3, yaxis_type3, ):

        fig = px.scatter(x=df_sar_vis[xaxis_column_name3],
                          y=df_sar_vis[yaxis_column_name3],
                          hover_name=df_sar_vis["np_core"],
                          )
        fig.update_traces(marker=dict(color="rgb(193, 39, 45, 0.93)"))
        fig.update_layout({"margin": {"l": 40, "b": 40, "t": 10, "r": 0}, "legend_orientation": "h",
                            "paper_bgcolor": "rgba(0, 0, 0,0)", "plot_bgcolor": "rgba(0, 0, 0,0)"})
        fig.update_xaxes(title=xaxis_column_name3,
                          type="linear" if xaxis_type3 == "Linear" else "log")
        fig.update_yaxes(title=yaxis_column_name3,
                          type="linear" if yaxis_type3 == "Linear" else "log")

        return fig

    return render(request, "database/sar.html")

@csrf_exempt
def offer(request):
    form = offerForm()
    if request.method == "POST":
        form = offerForm(request.POST or None)
        if form.is_valid():
            form.save()
            return redirect("database")
    data = {
        "form": form}
    return render(request, "database/offer.html", data)