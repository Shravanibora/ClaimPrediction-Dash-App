import pandas as pd
import numpy as np

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dash_table

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer 	
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# =========================================================
# Memory optimization helpers
# =========================================================
def reduce_memory(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('string')
    return df

# =========================================================
# 1. DATA LOADING
# =========================================================
import pandas as pd
import numpy as np

# Direct-download URLs for filtered CSVs on Google Drive
claims_url = "https://drive.google.com/uc?export=download&id=1SgsAesNi3SHouEtESiNnhY0KdFkvXsr9"
claims_transactions_url = "https://drive.google.com/uc?export=download&id=1CXiodxDFeTDxGc0iyIovXY2BDekHtKtd"
encounters_url = "https://drive.google.com/uc?export=download&id=1LZCyHiy8Q2v4Mq-4xuZY-bOax-pX2OsM"
patients_url = "https://drive.google.com/uc?export=download&id=1aVTH4eFmTn8MZxNyQpZ1cMuQR_BfiVh6"
payer_transitions_url = "https://drive.google.com/uc?export=download&id=1iatTcuuwb-8-Bbc5sclHv_voo8NJZybw"

# Load all main tables from Drive
patients = pd.read_csv(
    patients_url,
    usecols=["Id", "BIRTHDATE", "GENDER"],
    dtype={"Id": "string", "GENDER": "string"}
)
patients = reduce_memory(patients)

# Encounters
encounters = pd.read_csv(
    encounters_url,
    usecols=["PATIENT", "DENIAL_REASON", "ENCOUNTERCLASS", "REASONDESCRIPTION", "PAYER", "TOTAL_CLAIM_COST"],
    dtype={"PATIENT": "string", "DENIAL_REASON": "string", "ENCOUNTERCLASS": "string",
           "REASONDESCRIPTION": "string", "PAYER": "string", "TOTAL_CLAIM_COST": "float32"}
)
encounters = reduce_memory(encounters)

# Claims
claims = pd.read_csv(
    claims_url,
    usecols=["PATIENTID", "STATUS1", "APPOINTMENTID", "STATUS2", "STATUSP"],
    dtype={"PATIENTID": "string", "STATUS1": "string", "STATUS2": "string", "STATUSP": "string",
           "APPOINTMENTID": "string"}
)
claims = reduce_memory(claims)

# Claims trans
claims_transactions = pd.read_csv(
    claims_transactions_url,
    sep="\t",
    skipinitialspace=True,
)
claims_transactions.columns = claims_transactions.columns.str.strip()

# Try to locate the TODATE-like column
candidate_cols = [c for c in claims_transactions.columns if c.upper().replace("_", "") == "TODATE"]
if candidate_cols:
    todate_col = candidate_cols[0]
    claims_transactions[todate_col] = pd.to_datetime(
        claims_transactions[todate_col], errors="coerce"
    )
    if pd.api.types.is_datetime64tz_dtype(claims_transactions[todate_col].dtype):
        claims_transactions[todate_col] = claims_transactions[todate_col].dt.tz_convert(None)
    # Optionally standardize the name to exactly "TODATE"
    claims_transactions = claims_transactions.rename(columns={todate_col: "TODATE"})
else:
    # No TODATE-like column found; create an all-NaT column so later code still runs
    claims_transactions["TODATE"] = pd.NaT



# Payer Transitions
payer_transitions = pd.read_csv(
    payer_transitions_url,
    usecols=["PATIENT", "PAYER"],
    dtype={"PATIENT": "string", "PAYER": "string"}
)
payer_transitions = reduce_memory(payer_transitions)

# Type fixes
claims["PATIENTID"] = claims["PATIENTID"].astype(str)
payer_transitions["PATIENT"] = payer_transitions["PATIENT"].astype(str)
encounters["PATIENT"] = encounters["PATIENT"].astype(str)
patients["Id"] = patients["Id"].astype(str)



USER_CREDENTIALS = {"MRPRCM1": "Password@123"}



# =========================================================
# 2. MODEL TRAINING USING encounters_filtered
# =========================================================
df = encounters.copy()
df["DENIED"] = np.where(df["DENIAL_REASON"].notna() & (df["DENIAL_REASON"].str.lower() != "approved"), 1, 0)

df = df.merge(patients[["Id", "BIRTHDATE"]], left_on="PATIENT", right_on="Id", how="left")
if "BIRTHDATE" in df.columns:
    df["BIRTHDATE"] = pd.to_datetime(df["BIRTHDATE"], errors="coerce")
    ref_date = pd.to_datetime("today")
    df["AGE"] = ((ref_date - df["BIRTHDATE"]).dt.days / 365.25).clip(0, 100)
else:
    df["AGE"] = np.nan

features = ["PAYER", "ENCOUNTERCLASS", "REASONDESCRIPTION", "TOTAL_CLAIM_COST", "AGE"]
df_model = df[features + ["DENIED"]].copy()

for col in ["PAYER", "ENCOUNTERCLASS", "REASONDESCRIPTION"]:
    df_model[col] = df_model[col].astype(str)
df_model = df_model.dropna(subset=["PAYER", "ENCOUNTERCLASS", "TOTAL_CLAIM_COST"])

X = df_model[features]
y = df_model["DENIED"]
cat_features = ["PAYER", "ENCOUNTERCLASS", "REASONDESCRIPTION"]
num_features = ["TOTAL_CLAIM_COST", "AGE"]

preprocess = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
                  ("num", StandardScaler(), num_features)]
)

log_reg = LogisticRegression(max_iter=200, class_weight="balanced", solver="lbfgs")
denial_model = Pipeline([("preprocess", preprocess), ("model", log_reg)])

if len(X) > 0 and y.nunique() > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    denial_model.fit(X_train, y_train)
    try:
        y_pred_proba = denial_model.predict_proba(X_test)[:, 1]
        print("Denial model ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    except:
        print("Could not compute ROC AUC")
else:
    denial_model = None
    print("Warning: Not enough data to train denial model.")
# =========================================================
# 3. DASH APP SETUP
# =========================================================
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://use.fontawesome.com/releases/v5.8.1/css/all.css",
]

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
)
server = app.server

# =========================================================
# 4. GLOBAL STYLES AND DROPDOWNS
# =========================================================
highlight_style = {
    "fontWeight": "bold",
    "fontSize": "2.3rem",
    "color": "#1a237e",
    "textShadow": "1px 1px 6px #009FFD, 0 2px 14px #86A8E7",
    "background": "linear-gradient(90deg, #E0EAFC 60%, #CFDEF3 100%)",
    "padding": "18px 0 18px 0",
    "marginBottom": "12px",
    "marginTop": "24px",
    "borderBottom": "4px solid #009FFD",
    "borderRadius": "8px",
    "boxShadow": "0px 3px 18px -3px #009FFD88",
}
tab_style = {
    "border": "1px solid #ddd",
    "padding": "12px",
    "fontWeight": "bold",
    "fontSize": "1.1rem",
    "color": "#444",
    "background": "#E3F0FF",
}
tab_selected_style = {
    "border": "2px solid #009FFD",
    "padding": "16px",
    "fontWeight": "bold",
    "color": "#fff",
    "background": "linear-gradient(90deg,#009FFD 80%, #2A2A72 100%)",
    "boxShadow": "0 4px 18px 0 rgba(27,92,245,0.12)",
}

health_issues = encounters["REASONDESCRIPTION"].dropna().astype(str).unique()
health_issues = sorted(health_issues)  # Python sort returning a list
health_dropdown_options = [{"label": issue, "value": issue} for issue in health_issues]


# =========================================================
# 5. LOGIN LAYOUT
# =========================================================
login_layout = dbc.Container(
    dbc.Row(
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.Div(
                                html.I(
                                    className="fas fa-lock fa-3x",
                                    style={"color": "#009FFD"},
                                ),
                                style={
                                    "textAlign": "center",
                                    "marginBottom": "15px",
                                },
                            ),
                            html.H3(
                                "Login",
                                className="text-center",
                                style={"color": "#009FFD", "marginBottom": "10px"},
                            ),
                            html.H5(
                                "Real-Time Insurance Claim Denial Prediction",
                                className="text-center",
                                style={
                                    "marginBottom": "20px",
                                    "color": "#2A2A72",
                                },
                            ),
                            dbc.Label("Username", style={"fontWeight": "bold"}),
                            dbc.Input(
                                id="username",
                                type="text",
                                placeholder="Enter username",
                                style={"marginBottom": "10px"},
                            ),
                            dbc.Label(
                                "Password",
                                style={"fontWeight": "bold", "marginTop": "5px"},
                            ),
                            dbc.Input(
                                id="password",
                                type="password",
                                placeholder="Enter password",
                                style={"marginBottom": "15px"},
                            ),
                            dbc.Button(
                                "Login",
                                id="login-button",
                                color="primary",
                                style={
                                    "marginBottom": "10px",
                                    "width": "100%",
                                },
                            ),
                            html.Div(
                                id="login-message",
                                className="text-danger mt-2",
                                style={"textAlign": "center"},
                            ),
                            html.Div(
                                "Demo user: MRPRCM1 | Password: Password@123",
                                className="text-muted small",
                                style={
                                    "marginTop": "14px",
                                    "textAlign": "center",
                                },
                            ),
                        ]
                    )
                ],
                style={
                    "marginTop": "70px",
                    "marginBottom": "40px",
                    "maxWidth": "340px",
                    "marginLeft": "auto",
                    "marginRight": "auto",
                    "boxShadow": "0 2px 12px 2px #EEF3F8",
                },
            )
        )
    )
)

# =========================================================
# 6. DASHBOARD LAYOUT (TABS + CARDS)
# =========================================================
def dashboard_layout():
    card_styles = [
        {"header": "Patients", "icon": "fa-users", "color": "#009FFD"},
        {"header": "Encounters", "icon": "fa-stethoscope", "color": "#2A2A72"},
        {"header": "Claims", "icon": "fa-file-medical", "color": "#F6F930"},
        {"header": "Payers", "icon": "fa-university", "color": "#EA5455"},
    ]
    stats = [
        patients.shape[0],
        encounters.shape[0],
        claims.shape[0],
        payer_transitions["PAYER"].nunique(),
    ]
    return dbc.Container(
        [
            html.H2(
                "Dashboard: Real-Time Insurance Claim Denial Prediction",
                className="text-center",
                style=highlight_style,
            ),
            html.Div(
                [
                    dcc.Tabs(
                        id="tabs",
                        value="tab-overview",
                        children=[
                            dcc.Tab(
                                label="Overview",
                                value="tab-overview",
                                style=tab_style,
                                selected_style=tab_selected_style,
                            ),
                            dcc.Tab(
                                label="Patients",
                                value="tab-patients",
                                style=tab_style,
                                selected_style=tab_selected_style,
                            ),
                            dcc.Tab(
                                label="Encounters",
                                value="tab-encounters",
                                style=tab_style,
                                selected_style=tab_selected_style,
                            ),
                            dcc.Tab(
                                label="Claims",
                                value="tab-claims",
                                style=tab_style,
                                selected_style=tab_selected_style,
                            ),
                            dcc.Tab(
                                label="Denial Reasons",
                                value="tab-denial",
                                style=tab_style,
                                selected_style=tab_selected_style,
                            ),
                            dcc.Tab(
                                label="Payers",
                                value="tab-payers",
                                style=tab_style,
                                selected_style=tab_selected_style,
                            ),
                            dcc.Tab(
                                label="Predict Denial",
                                value="tab-predict",
                                style=tab_style,
                                selected_style=tab_selected_style,
                            ),
                        ],
                        style={"marginBottom": "10px"},
                    ),
                    html.Div(id="tabs-content", className="mt-4"),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            [
                                                html.I(
                                                    className=f"fas {card['icon']}",
                                                    style={
                                                        "color": card["color"],
                                                        "fontSize": "2em",
                                                        "marginBottom": "8px",
                                                    },
                                                ),
                                                html.H5(
                                                    card["header"],
                                                    className="card-title",
                                                ),
                                                html.H2(
                                                    f"{stat:,}",
                                                    style={
                                                        "fontWeight": "bold",
                                                        "color": card["color"],
                                                    },
                                                ),
                                            ],
                                            style={"textAlign": "center"},
                                        )
                                    ]
                                )
                            ],
                            style={
                                "background": "#fff",
                                "marginBottom": "24px",
                                "boxShadow": "0 2px 8px 0 #CBCBCC",
                                "borderRadius": "12px",
                                "border": f"2px solid {card['color']}",
                            },
                        ),
                        width=3,
                    )
                    for card, stat in zip(card_styles, stats)
                ],
                className="mt-4",
            ),

        ],
        style={"paddingBottom": "70px"},
    )


# =========================================================
# 7. TAB CONTENT CALLBACK
# =========================================================
@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    genders = patients["GENDER"].dropna().unique()
    encounter_classes = encounters["ENCOUNTERCLASS"].dropna().unique()
    statuses = claims["STATUS1"].dropna().unique()
    payers_list = payer_transitions["PAYER"].dropna().unique()

    if tab == "tab-overview":
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                "Overview: Use the tabs above to explore claims, denial reasons, patients, and payers.",
                                style={
                                    "fontWeight": "bold",
                                    "fontSize": 20,
                                    "marginTop": 40,
                                    "marginBottom": 20,
                                    "color": "#2A2A72",
                                    "textAlign": "center",
                                },
                            ),
                            width=12,
                        )
                    ]
                )
            ]
        )

    elif tab == "tab-patients":
        return dbc.Container(
            [
                dbc.Alert(
                    "View and analyze the distribution of patients by gender and birthdate. Use filters to drill down.",
                    color="info",
                    dismissable=True,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Filter by Gender"),
                                dcc.Dropdown(
                                    id="patient-gender-filter",
                                    options=[{"label": g, "value": g} for g in genders],
                                    multi=True,
                                ),
                            ],
                            width=3,
                        )
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [dbc.Col(dcc.Graph(id="patient-birthdate-graph"), width=12)]
                ),
            ]
        )

    elif tab == "tab-encounters":
        return dbc.Container(
            [
                dbc.Alert(
                    "View encounter breakdown and class filters for deeper understanding of encounter types.",
                    color="info",
                    dismissable=True,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Filter by Encounter Class"),
                                dcc.Dropdown(
                                    id="encounter-class-filter",
                                    options=[
                                        {"label": c, "value": c}
                                        for c in encounter_classes
                                    ],
                                    multi=True,
                                ),
                            ],
                            width=3,
                        )
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [dbc.Col(dcc.Graph(id="encounter-class-graph"), width=12)]
                ),
            ]
        )

    elif tab == "tab-claims":
        # table columns
        columns = [
            {"name": "Patient ID", "id": "PATIENTID"},
            {"name": "Payer", "id": "PAYER"},
            {"name": "Status", "id": "STATUS1"},
            {"name": "Appointment Id", "id": "APPOINTMENTID"},
            {"name": "Status 2", "id": "STATUS2"},
            {"name": "Status P", "id": "STATUSP"},
        ]
        columns = [c for c in columns if c["id"] in claims.columns]

        # Last 10 years
        from datetime import datetime
        today = datetime.today().date()
        ten_years_ago = today.replace(year=today.year - 10)

        return dbc.Container(
            [
                dbc.Alert(
                    "Review claims status and payer details. Filter by status, payer, and service date. "
                    "Track 10-year trends and total financial exposure.",
                    color="info",
                    dismissable=True,
                ),

                # Row 1 – Filters (status + payer only)
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Filter by Claim Status",
                                    style={"fontWeight": "bold", "color": "#000"},
                                ),
                                dcc.Dropdown(
                                    id="claim-status-filter",
                                    options=[
                                        {"label": s, "value": s} for s in statuses
                                    ],
                                    multi=True,
                                    placeholder="Select status",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Filter by Payer",
                                    style={"fontWeight": "bold", "color": "#000"},
                                ),
                                dcc.Dropdown(
                                    id="claim-payer-filter",
                                    options=[
                                        {"label": p, "value": p} for p in payers_list
                                    ],
                                    multi=True,
                                    placeholder="Select payer",
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    className="mb-3",
                ),

                # Row 2 – Status graph
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="claims-status-graph",
                                style={"height": "420px", "width": "100%"},
                            ),
                            width=12,
                        )
                    ],
                    className="mb-4",
                ),

                # Row 3 – Filtered Claims Table
                html.Div(
                    "Filtered Claims Table",
                    style={"fontWeight": "bold", "marginTop": "15px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dash_table.DataTable(
                                id="claim-detail-table",
                                columns=columns,
                                page_size=10,
                                style_table={"overflowX": "auto"},
                                style_header={
                                    "backgroundColor": "#f9f9f9",
                                    "fontWeight": "bold",
                                },
                                style_cell={
                                    "padding": "8px",
                                    "textAlign": "left",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"row_index": "odd"},
                                        "backgroundColor": "#FAFAFA",
                                    },
                                ],
                                sort_action="native",
                            ),
                            width=12,
                        )
                    ],
                    className="mb-4",
                ),

                # Row 4 – Service Date filter (Last 10 Years)
                html.Div(
                    "Filter by Service Date (Last 10 Years)",
                    style={"fontWeight": "bold", "marginTop": "25px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.DatePickerRange(
                                    id="claim-date-range",
                                    min_date_allowed=ten_years_ago,
                                    max_date_allowed=today,
                                    start_date=ten_years_ago,
                                    end_date=today,
                                ),
                                dbc.Button(
                                    "Apply",
                                    id="apply-claims-filters",
                                    color="primary",
                                    size="sm",
                                    style={
                                        "marginLeft": "10px",
                                        "marginTop": "3px",
                                    },
                                ),
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),

                # Row 5 – Trend + Exposure
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="claims-trend-graph",
                                style={"height": "420px"},
                            ),
                            width=9,
                        ),
                        dbc.Col(
                            [
                                html.H5("Total Financial Exposure"),
                                html.H2(
                                    id="claims-exposure-text",
                                    style={
                                        "fontWeight": "bold",
                                        "color": "#C82333",
                                        "marginTop": "15px",
                                    },
                                ),
                                html.P(
                                    "Exposure = Sum of AMOUNT from claims_transactions_filtered for filtered claim lines.",
                                    style={
                                        "fontSize": "0.8rem",
                                        "marginTop": "10px",
                                    },
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    className="mb-5",
                ),
            ],
            fluid=True,
        )

    elif tab == "tab-denial":
        return dbc.Container(
            [
                dbc.Alert(
                    "Explore top denial reasons.",
                    color="info",
                    dismissable=True,
                ),
                dbc.Row([dbc.Col(dcc.Graph(id="denial-reason-graph"), width=12)]),
            ]
        )

    elif tab == "tab-payers":
        fig_payer = px.histogram(
            payer_transitions,
            x="PAYER",
            title="Payer Distribution",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_payer.update_layout(plot_bgcolor="#F5F6FA")
        return dbc.Container(
            [
                dbc.Alert(
                    "See distribution of payers involved in claims and transitions.",
                    color="info",
                    dismissable=True,
                ),
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_payer), width=12)]),
            ]
        )

    elif tab == "tab-predict":
        return dbc.Container(
            [
                html.H4("Predict Claim Denial", className="mb-3 text-primary"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Select Payer"),
                                dcc.Dropdown(
                                    id="predict-payer",
                                    options=[
                                        {"label": p, "value": p}
                                        for p in payer_transitions["PAYER"]
                                        .dropna()
                                        .unique()
                                    ],
                                    placeholder="Choose a payer",
                                    style={"width": "100%", "fontSize": "16px"},
                                    className="mb-2",
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Select Encounter Class"),
                                dcc.Dropdown(
                                    id="predict-encounter",
                                    options=[
                                        {"label": c, "value": c}
                                        for c in encounters["ENCOUNTERCLASS"]
                                        .dropna()
                                        .unique()
                                    ],
                                    placeholder="Choose encounter class",
                                    style={"width": "100%", "fontSize": "16px"},
                                    className="mb-2",
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Health Issue / Reason"),
                                dcc.Dropdown(
                                    id="predict-health-issue",
                                    options=health_dropdown_options,
                                    placeholder="Select Health Issue",
                                    style={
                                        "width": "100%",
                                        "fontSize": "16px",
                                        "maxHeight": "400px",
                                    },
                                    className="mb-2",
                                ),
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Procedure Cost"),
                                dbc.Input(
                                    id="predict-cost",
                                    type="number",
                                    placeholder="Enter cost",
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Patient Age"),
                                dbc.Input(
                                    id="predict-age",
                                    type="number",
                                    placeholder="Enter age",
                                ),
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Button(
                    "Predict Denial",
                    id="predict-button",
                    color="primary",
                    className="mt-2",
                    style={"marginRight": "10px"},
                ),
                dbc.Button(
                    "Clear Fields",
                    id="clear-button",
                    color="secondary",
                    className="mt-2",
                ),
                html.Div(id="predict-result", className="mt-4"),
            ],
            style={"paddingBottom": "70px"},
        )

    return html.Div("Tab not found")


# =========================================================
# 8. ANALYTIC GRAPH CALLBACKS
# =========================================================
@app.callback(
    Output("patient-birthdate-graph", "figure"),
    Input("patient-gender-filter", "value"),
)
def update_patient_graph(genders_selected):
    df = patients.copy()
    color_map = {
        "male": "#4361EE",
        "female": "#F72585",
        "other": "#4CC9F0",
        "": "#FFB100",
    }
    if genders_selected:
        df = df[df["GENDER"].isin(genders_selected)]
    fig = px.histogram(
        df,
        x="BIRTHDATE",
        color="GENDER",
        nbins=20,
        title="Patients by Birthdate",
        color_discrete_map=color_map,
    )
    fig.update_traces(marker_line_width=0.5, opacity=0.75)
    fig.update_layout(
        bargap=0.18,
        legend_title_text="Gender",
        plot_bgcolor="#F9F9F9",
        font=dict(family="Roboto", size=14),
        yaxis_title="Patient Count",
    )
    return fig


@app.callback(
    Output("encounter-class-graph", "figure"),
    Input("encounter-class-filter", "value"),
)
def update_encounter_graph(classes_selected):
    df = encounters.copy()
    if classes_selected:
        df = df[df["ENCOUNTERCLASS"].isin(classes_selected)]
    fig = px.histogram(
        df,
        x="ENCOUNTERCLASS",
        title="Encounters by Class",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(marker_line_width=0.5, opacity=0.75)
    fig.update_layout(
        xaxis_title="Encounter Class",
        yaxis_title="Count",
        plot_bgcolor="#F9F9F9",
    )
    return fig


@app.callback(
    Output("claims-status-graph", "figure"),
    Input("claim-status-filter", "value"),
    Input("claim-payer-filter", "value"),
)
def update_claims_graph(status_selected, payer_selected):
    df = claims.copy()

    # Filter by status only if STATUS1 exists
    if status_selected and "STATUS1" in df.columns:
        df = df[df["STATUS1"].isin(status_selected)]

    # Filter by payer only if both PATIENTID and PAYER data exist
    if payer_selected and len(payer_selected) > 0:
        if {"PATIENT", "PAYER"}.issubset(payer_transitions.columns) and "PATIENTID" in df.columns:
            payer_df = payer_transitions[["PATIENT", "PAYER"]].drop_duplicates()
            payer_df["PATIENT"] = payer_df["PATIENT"].astype(str)
            df["PATIENTID"] = df["PATIENTID"].astype(str)
            df = df.merge(
                payer_df, left_on="PATIENTID", right_on="PATIENT", how="left"
            )
            df = df[df["PAYER"].isin(payer_selected)]

    # If STATUS1 missing or no rows, build an empty-safe frame
    if "STATUS1" not in df.columns or df.empty:
        status_counts = (
            pd.Series(["BILLED", "CLOSED"], name="Status")
            .to_frame()
            .assign(Count=0)
        )
    else:
        status_counts = (
            df["STATUS1"]
            .value_counts()
            .reindex(["BILLED", "CLOSED"], fill_value=0)
            .reset_index()
        )
        status_counts.columns = ["Status", "Count"]

    fig = px.bar(
        status_counts,
        x="Status",
        y="Count",
        color="Status",
        color_discrete_map={"BILLED": "#EE6055", "CLOSED": "#60D394"},
        title="Claims by Status",
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        barmode="group",
        xaxis_title="Status",
        yaxis_title="Count",
        plot_bgcolor="#F9F9F9",
        font=dict(family="Roboto", size=14),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="#CCCCCC",
        ),
    )
    return fig


@app.callback(
    Output("claim-detail-table", "data"),
    Input("claim-status-filter", "value"),
    Input("claim-payer-filter", "value"),
    Input("tabs", "value"),
)
def update_claims_table(status_selected, payer_selected, current_tab):
    df = claims.copy()

    if current_tab != "tab-claims":
        return []

    if status_selected:
        df = df[df["STATUS1"].isin(status_selected)]

    if payer_selected and len(payer_selected) > 0:
        payer_df = payer_transitions[["PATIENT", "PAYER"]].drop_duplicates()
        payer_df["PATIENT"] = payer_df["PATIENT"].astype(str)
        df["PATIENTID"] = df["PATIENTID"].astype(str)
        df = df.merge(
            payer_df, left_on="PATIENTID", right_on="PATIENT", how="left"
        )
        df = df[df["PAYER"].isin(payer_selected)]

    columns = [
        "PATIENTID",
        "PAYER",
        "STATUS1",
        "APPOINTMENTID",
        "STATUS2",
        "STATUSP",
    ]
    columns = [c for c in columns if c in df.columns]
    return df[columns].to_dict("records")


@app.callback(
    Output("denial-reason-graph", "figure"),
    Input("tabs", "value"),
)
def update_denial_graph(tab_value):
    df = encounters.copy()
    df = df[df["DENIAL_REASON"].notnull()]
    denial_counts = df["DENIAL_REASON"].value_counts().reset_index()
    denial_counts.columns = ["Denial Reason", "Count"]
    fig = px.bar(
        denial_counts,
        x="Denial Reason",
        y="Count",
        title="Most Frequent Denial Reasons",
        color="Denial Reason",
        color_discrete_sequence=px.colors.qualitative.Bold,
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-45, plot_bgcolor="#EEF3F8")
    return fig


#new callback
from datetime import datetime

@app.callback(
    [
        Output("claims-trend-graph", "figure"),
        Output("claims-exposure-text", "children"),
    ],
    Input("apply-claims-filters", "n_clicks"),
    [
        State("claim-status-filter", "value"),
        State("claim-payer-filter", "value"),
        State("claim-date-range", "start_date"),
        State("claim-date-range", "end_date"),
    ],
)
def update_claims_trend_and_exposure(
    n_clicks,
    status_selected,
    payer_selected,
    start_date,
    end_date,
):
    if not n_clicks:
        fig = px.bar(title="Select filters and click Apply")
        return fig, ""

    # 1. Start from transactions table (has TODATE, AMOUNT)
    df = claims_transactions.copy()
    print("initial rows:", len(df))
    print("non-null TODATE:", df["TODATE"].notna().sum())
    print("TODATE min/max:", df["TODATE"].min(), df["TODATE"].max())

    # 2. Join to claims to bring STATUS1 and PATIENTID
    base_cols = [c for c in ["Id", "PATIENTID", "STATUS1"] if c in claims.columns]
    base = claims[base_cols].copy()
    if "Id" in base.columns and "CLAIMID" in df.columns:
        base["Id"] = base["Id"].astype(str)
        df["CLAIMID"] = df["CLAIMID"].astype(str)
        df = df.merge(
            base,
            left_on="CLAIMID",
            right_on="Id",
            how="left",
        )
    print("after join:", len(df))

    # 3. Limit to last 10 years by TODATE
    today = datetime.today().date()
    ten_years_ago = today.replace(year=today.year - 10)
    start_limit = pd.to_datetime(ten_years_ago)
    end_limit = pd.to_datetime(today)

    df = df[
        (df["TODATE"] >= start_limit) &
        (df["TODATE"] <= end_limit)
    ]
    print("after 10y filter:", len(df))

    # 4. Status filter (from claims.STATUS1)
    if status_selected and "STATUS1" in df.columns:
        df = df[df["STATUS1"].isin(status_selected)]
    print("after status filter:", len(df))

    # 5. Payer filter (join to payer_transitions by PATIENTID)
    if payer_selected and len(payer_selected) > 0:
        if {"PATIENT", "PAYER"}.issubset(payer_transitions.columns) and "PATIENTID" in df.columns:
            payer_df = payer_transitions[["PATIENT", "PAYER"]].drop_duplicates()
            payer_df["PATIENT"] = payer_df["PATIENT"].astype(str)
            df["PATIENTID"] = df["PATIENTID"].astype(str)

            df = df.merge(
                payer_df,
                left_on="PATIENTID",
                right_on="PATIENT",
                how="left",
            )
            df = df[df["PAYER"].isin(payer_selected)]
    print("after payer filter:", len(df))

    # 6. Date picker filter (using TODATE)
    if start_date:
        df = df[df["TODATE"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["TODATE"] <= pd.to_datetime(end_date)]
    print("after datepicker filter:", len(df))

    # 7. If nothing left
    if df.empty:
        fig = px.bar(title="No claims for selected filters")
        fig.update_layout(
            xaxis_title="Service Month (by TODATE)",
            yaxis_title="Number of Claim",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig, "0 (no claims)"

    # 8. Use AMOUNT as exposure per transaction line
    df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce").fillna(0)

    # 9. Monthly trend by TODATE
    df["SERVICEMONTH"] = df["TODATE"].dt.to_period("M").dt.to_timestamp()
    trend_counts = (
        df.groupby("SERVICEMONTH")
        .size()
        .reset_index(name="claim_count")
    )

    fig = px.bar(
        trend_counts,
        x="SERVICEMONTH",
        y="claim_count",
        title="Claims Over Time (Monthly, Last 10 Years, by TODATE)",
        text="claim_count",
    )
    fig.update_traces(
        textposition="outside",
        marker_color="#1f77b4",
    )
    fig.update_layout(
        xaxis_title="Service Month (by TODATE)",
        yaxis_title="Number of Claim",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=60),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="#CCCCCC",
            tick0=0,
            dtick=5000,
        ),
    )

    # 10. Total financial exposure = sum of AMOUNTS
    total_exposure = df["AMOUNT"].sum()
    exposure_text = f"${total_exposure:,.2f}"

    return fig, exposure_text

# =========================================================
# 9. PREDICTION CALLBACK (USES MODEL)
# =========================================================
@app.callback(
    [
        Output("predict-payer", "value"),
        Output("predict-encounter", "value"),
        Output("predict-health-issue", "value"),
        Output("predict-cost", "value"),
        Output("predict-age", "value"),
        Output("predict-result", "children"),
    ],
    [Input("clear-button", "n_clicks"), Input("predict-button", "n_clicks")],
    [
        State("predict-payer", "value"),
        State("predict-encounter", "value"),
        State("predict-health-issue", "value"),
        State("predict-cost", "value"),
        State("predict-age", "value"),
    ],
    prevent_initial_call=True,
)
def predict_and_clear(
    clear_clicks,
    predict_clicks,
    payer,
    encounter,
    health_issue,
    cost,
    age,
):
    from dash import ctx

    triggered = ctx.triggered_id if hasattr(ctx, "triggered_id") else None

    # Clear button resets all fields + result
    if triggered == "clear-button":
        return None, None, None, None, None, ""

    if triggered == "predict-button":
        # basic validation
        if not all([payer, encounter, cost, age]):
            return (
                payer,
                encounter,
                health_issue,
                cost,
                age,
                dbc.Alert(
                    "Please fill in all (except health issue) fields before predicting.",
                    color="warning",
                ),
            )

        if denial_model is None:
            return (
                payer,
                encounter,
                health_issue,
                cost,
                age,
                dbc.Alert(
                    "Model could not be trained (insufficient data).",
                    color="danger",
                ),
            )

        # ---------------------------
        # 1. Model probability
        # ---------------------------
        row = {
            "PAYER": [str(payer)],
            "ENCOUNTERCLASS": [str(encounter)],
            "REASONDESCRIPTION": [str(health_issue) if health_issue else ""],
            "TOTAL_CLAIM_COST": [float(cost)],
            "AGE": [float(age)],
        }
        X_new = pd.DataFrame(row)

        proba_denied = float(denial_model.predict_proba(X_new)[:, 1][0])
        pct = proba_denied * 100

        if pct < 40:
            header_text = "LOW Risk of Denial"
            header_color = "#218838"
        elif pct < 70:
            header_text = "MODERATE Risk of Denial"
            header_color = "#FFC107"
        else:
            header_text = "HIGH Risk of Denial"
            header_color = "#C82333"

        header = html.H5(
            f"Predicted Denial Probability: {pct:.1f}% — {header_text}",
            style={"color": header_color, "fontWeight": "bold"},
        )

        # ---------------------------
        # 2. Historical denial rates
        # ---------------------------
        hist = encounters.copy()
        # build same DENIED flag as in training
        hist["DENIED"] = np.where(
            hist["DENIAL_REASON"].notna()
            & (hist["DENIAL_REASON"].str.lower() != "approved"),
            1,
            0,
        )

        # similar encounters = same payer + encounter class
        similar = hist[
            (hist["PAYER"] == payer) & (hist["ENCOUNTERCLASS"] == encounter)
        ]

        if health_issue and "REASONDESCRIPTION" in hist.columns:
            similar_hi = similar[similar["REASONDESCRIPTION"] == health_issue]
            if not similar_hi.empty:
                similar = similar_hi

        if not similar.empty:
            hist_rate = similar["DENIED"].mean() * 100
            hist_text = (
                f"Historical denial rate for this payer & encounter class"
                + (f" and reason '{health_issue}'" if health_issue else "")
                + f": {hist_rate:.1f}% (based on {len(similar)} similar encounters)."
            )
        else:
            hist_rate = None
            hist_text = (
                "No similar historical encounters found for this payer and "
                "encounter class, so the model relies more on overall patterns."
            )

        # ---------------------------
        # 3. Best payer suggestion for this encounter class
        # ---------------------------
        class_group = hist[hist["ENCOUNTERCLASS"] == encounter]
        if not class_group.empty:
            by_payer = class_group.groupby("PAYER")["DENIED"].mean()
            best_payer = by_payer.idxmin()
            best_rate = by_payer.min() * 100
            cur_rate = (
                by_payer.get(payer) * 100 if payer in by_payer.index else None
            )

            if cur_rate is not None and best_payer != payer and best_rate + 1 < cur_rate:
                payer_text = (
                    f"Best historical payer for this encounter class is {best_payer} "
                    f"with a denial rate of {best_rate:.1f}%. "
                    f"Current payerâ€™s historical rate is {cur_rate:.1f}%."
                )
            elif cur_rate is not None:
                payer_text = (
                    f"This payer already has one of the lowest denial rates "
                    f"for this encounter class (about {cur_rate:.1f}%)."
                )
            else:
                payer_text = (
                    "No reliable historical denial rate found for this payer in "
                    "this encounter class."
                )
        else:
            payer_text = (
                "No historical encounters found for this encounter class to compare payers."
            )

        # ---------------------------
        # 4. Top denial reasons in similar cases
        # ---------------------------
        if not similar.empty:
            top_reasons = (
                similar["DENIAL_REASON"]
                .fillna("Unknown")
                .value_counts()
                .head(2)
            )
            reason_parts = [
                f"{reason} ({count / len(similar) * 100:.0f}%)"
                for reason, count in top_reasons.items()
            ]
            reasons_text = (
                "Most frequent denial reasons in similar encounters: "
                + " | ".join(reason_parts)
            )
        else:
            reasons_text = (
                "No denial-reason breakdown available for similar historical encounters."
            )

        # ---------------------------
        # 5. Build rich explanation block
        # ---------------------------
        detail_block = html.Div(
            [
                dbc.Alert(hist_text, color="info"),
                dbc.Alert(reasons_text, color="secondary"),
                dbc.Alert(payer_text, color="warning"),
            ]
        )

        return (
            payer,
            encounter,
            health_issue,
            cost,
            age,
            html.Div([header, detail_block]),
        )

    return payer, encounter, health_issue, cost, age, ""


# =========================================================
# 10. MAIN PAGE ROUTING (LOGIN)
# =========================================================
app.layout = html.Div(id="page-content", children=login_layout)


@app.callback(
    Output("page-content", "children"),
    Output("login-message", "children"),
    Input("login-button", "n_clicks"),
    State("username", "value"),
    State("password", "value"),
)
def update_output(n_clicks, username, password):
    if n_clicks:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            return dashboard_layout(), ""
        return login_layout, "Invalid username or password. Try again."
    return login_layout, ""

# =========================================================
# 11. RUN APP
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)

















