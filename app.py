
import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dash import Dash, dcc, html, Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from scipy import stats


DATA_PATH = "game.csv"

# ===== Academic / Professor-Friendly Theme =====
APP_BG = "#f3f5f9"          # slightly softer light gray (updated)
CARD_BG = "#ffffff"         # white cards
PLOT_BG = "#ffffff"
PAPER_BG = "#ffffff"

TITLE_BLUE = "#1f4fd8"      # deep academic blue (title)
AXIS_RED = "#8b0000"        # dark red (axis labels)
TEXT_DARK = "#222222"
GRID = "rgba(0,0,0,0.10)"

WHITE = "#ffffff"
ORANGE = TITLE_BLUE         # reuse blue for accents
NAVY = TITLE_BLUE
NAVY2 = "#173a8a"
LIGHTGREY = "#666666"

TITLE_FONT = dict(family="serif", size=26, color="blue")
AXIS_LABEL_FONT = dict(family="serif", size=18, color="darkred")
TICK_FONT = dict(family="serif", size=14, color=TEXT_DARK)
DECIMAL_FORMAT = ".2f"

HEADING_FONT = "'Bebas Neue', Arial, sans-serif"
BODY_FONT = "'Lato', Arial, sans-serif"
BRIGHT_QUAL = px.colors.qualitative.Set3

# ✅ NEW: bordered section container like your screenshot/tab panel
SECTION_CONTAINER_STYLE = {
    "backgroundColor": "#f2f4f8",
    "border": f"2px solid {TITLE_BLUE}",
    "borderRadius": "16px",
    "padding": "14px 16px",
    "marginTop": "12px",
}

# ✅ UPDATED: tabs with clearer separators (bottom border)
TAB_STYLE = {
    "padding": "10px 14px",
    "fontFamily": "serif",
    "backgroundColor": "#ffffff",
    "border": "1px solid #dddddd",
    "borderBottom": f"3px solid {GRID}",   # added
    "color": TEXT_DARK,
}

TAB_SELECTED_STYLE = {
    "padding": "10px 14px",
    "fontFamily": "serif",
    "backgroundColor": "rgba(31,79,216,0.10)",
    "border": f"2px solid {TITLE_BLUE}",
    "borderBottom": f"3px solid {TITLE_BLUE}",
    "color": TEXT_DARK,
    "fontWeight": "700",
}

# =========================
# NBA TEAM ABBREVIATIONS
# =========================
NBA_TEAM_ABBREVS = {
    "ATL","BOS","BKN","BRK","CHA","CHH","CHI","CLE","DAL","DEN",
    "DET","GSW","HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN",
    "NOP","NOH","NYK","OKC","ORL","PHI","PHX","POR","SAC","SAS",
    "SEA","TOR","UTA","WAS","WSB","VAN"
}


def erfinv(y):
    a = 0.147
    sign = np.sign(y)
    ln = np.log(1 - y * y)
    first = 2 / (np.pi * a) + ln / 2
    second = ln / a
    return sign * np.sqrt(np.sqrt(first * first - second) - first)


def season_start_year_from_id(series: pd.Series) -> pd.Series:
    return series.astype(int) % 10000


def era_label(y: int | float | str) -> str:
    y = int(y)
    return f"{(y // 10) * 10}s"


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b == 0, np.nan, a / b)


def z_transform(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=1)
    if std == 0:
        return series * 0.0
    return (series - mean) / std


def style_fig_app(fig: go.Figure, x_title=None, y_title=None):
    fig.update_layout(
        template="simple_white",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        title_font=dict(family="serif", size=26, color=TITLE_BLUE),
        font=dict(family="serif", size=14, color=TEXT_DARK),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        margin=dict(l=60, r=40, t=80, b=60),
    )

    fig.update_xaxes(
        title_text=x_title,
        title_font=dict(family="serif", size=16, color=AXIS_RED),
        tickformat=".2f",
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
    )

    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(family="serif", size=16, color=AXIS_RED),
        tickformat=".2f",
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
    )

    return fig


def split_numeric_columns_for_analysis(df_in: pd.DataFrame):
    """
    Returns:
      analysis_num: numeric columns that behave like real measurements
      id_like: numeric columns that are identifiers and should be treated as categorical
    """
    num_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()

    id_like = []
    for c in num_cols:
        cl = c.lower()

        # 1) name-based rules
        if cl.endswith("_id") or cl in {"id", "game_id", "team_id", "season_id"} or "id" in cl:
            id_like.append(c)
            continue

        # 2) “code-like” integers with many unique values
        s = df_in[c]
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) > 0.8 * len(s):
            id_like.append(c)
            continue

        # 3) year-ish columns you may want OUT of generic numeric dist plots
        if cl in {"season_start_year"}:
            id_like.append(c)
            continue

    analysis_num = [c for c in num_cols if c not in id_like]
    return analysis_num, id_like


def clean_category_text(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"[\u2010-\u2015]", "-", regex=True)
    out = out.str.replace(r"\s*-\s*", "-", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True)
    out = out.str.title()
    return out


# ✅ UPDATED: stronger card border + slightly softer shadow
def card(*children):
    return html.Div(
        list(children),
        style={
            "background": CARD_BG,
            "boxShadow": "0 3px 10px rgba(0,0,0,0.12)",
            "border": f"2px solid {TITLE_BLUE}",
            "borderRadius": "14px",
            "padding": "12px 14px",
            "margin": "10px 0",
        },
    )


# =========================
# LOAD + CLEAN DATA
# =========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Could not find {DATA_PATH}. Put game.csv next to this script or fix DATA_PATH.")

df_raw = pd.read_csv(DATA_PATH)
df = df_raw.copy()
df.columns = [c.strip().lower() for c in df.columns]

# keep only NBA teams
if "team_abbreviation_home" not in df.columns:
    raise ValueError("Expected column 'team_abbreviation_home' in game.csv")

df["team_abbreviation_home"] = df["team_abbreviation_home"].astype(str).str.strip().str.upper()
df = df[df["team_abbreviation_home"].isin(NBA_TEAM_ABBREVS)].copy()

df["season_type"] = df["season_type"].replace({
    "All Star": "All-Star",
    "Allstar": "All-Star",
    "All-Star Game": "All-Star",
    "Allstar Game": "All-Star",
    "All-Stargame": "All-Star",
})

if "season_id" not in df.columns:
    raise ValueError("season_id column is required in game.csv")

df["season_start_year"] = season_start_year_from_id(df["season_id"])
df["era"] = df["season_start_year"].apply(era_label)


def mk_tot(stat: str):
    h, a = f"{stat}_home", f"{stat}_away"
    if h in df.columns and a in df.columns:
        df[f"{stat}_tot"] = df[h].astype(float).fillna(0) + df[a].astype(float).fillna(0)


for stat_name in ["pts", "ast", "reb", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm"]:
    mk_tot(stat_name)

if "pts_home" in df.columns:
    df["pts_home_team"] = df["pts_home"]
if "pts_away" in df.columns:
    df["pts_away_team"] = df["pts_away"]

if {"fgm_tot", "fga_tot"}.issubset(df.columns):
    df["fg_pct"] = safe_div(df["fgm_tot"], df["fga_tot"])
if {"fg3m_tot", "fg3a_tot"}.issubset(df.columns):
    df["fg3_pct"] = safe_div(df["fg3m_tot"], df["fg3a_tot"])
if {"ftm_tot", "fta_tot"}.issubset(df.columns):
    df["ft_pct"] = safe_div(df["ftm_tot"], df["fta_tot"])
if {"fg3a_tot", "fga_tot"}.issubset(df.columns):
    df["share_3pa"] = safe_div(df["fg3a_tot"], df["fga_tot"])
    df["share_2pa"] = 1 - df["share_3pa"]

df = df[(df["season_start_year"] >= 1946) & (df["season_start_year"] <= 2025)].copy()

num_cols_for_season = [
    c for c in [
        "pts_tot", "ast_tot", "reb_tot", "fga_tot", "fgm_tot",
        "fg3a_tot", "fg3m_tot", "fta_tot", "ftm_tot",
        "fg_pct", "fg3_pct", "ft_pct",
        "share_3pa", "share_2pa",
        "pts_home_team", "pts_away_team",
    ] if c in df.columns
]

league_season = df.groupby("season_start_year")[num_cols_for_season].mean(numeric_only=True).reset_index()
league_season["era"] = league_season["season_start_year"].apply(era_label)

team_name_col = None
for cand in ["team_abbreviation_home", "home_team", "team_home", "team_name_home"]:
    if cand in df.columns:
        team_name_col = cand
        break
# =========================
# TEAM-SEASON (FIXED: use BOTH home + away)
# =========================
team_season = None

need_cols = [
    "season_start_year",
    "team_abbreviation_home", "team_abbreviation_away",
    "pts_home", "pts_away",
    "ast_home", "ast_away",
    "reb_home", "reb_away",
    "fga_home", "fga_away",
    "fgm_home", "fgm_away",
    "fg3a_home", "fg3a_away",
    "fg3m_home", "fg3m_away",
    "fta_home", "fta_away",
    "ftm_home", "ftm_away",
]
need_cols = [c for c in need_cols if c in df.columns]

if all(c in df.columns for c in ["team_abbreviation_home", "team_abbreviation_away", "season_start_year"]):
    # HOME rows
    home = pd.DataFrame({
        "team": df["team_abbreviation_home"].astype(str).str.strip().str.upper(),
        "season_start_year": df["season_start_year"],
    })

    # AWAY rows
    away = pd.DataFrame({
        "team": df["team_abbreviation_away"].astype(str).str.strip().str.upper(),
        "season_start_year": df["season_start_year"],
    })

    # Add stats if present (keep names consistent as *_tot per team-game)
    if "pts_home" in df.columns and "pts_away" in df.columns:
        home["pts_tot"] = pd.to_numeric(df["pts_home"], errors="coerce")
        away["pts_tot"] = pd.to_numeric(df["pts_away"], errors="coerce")

    if "ast_home" in df.columns and "ast_away" in df.columns:
        home["ast_tot"] = pd.to_numeric(df["ast_home"], errors="coerce")
        away["ast_tot"] = pd.to_numeric(df["ast_away"], errors="coerce")

    if "reb_home" in df.columns and "reb_away" in df.columns:
        home["reb_tot"] = pd.to_numeric(df["reb_home"], errors="coerce")
        away["reb_tot"] = pd.to_numeric(df["reb_away"], errors="coerce")

    if "fga_home" in df.columns and "fga_away" in df.columns:
        home["fga_tot"] = pd.to_numeric(df["fga_home"], errors="coerce")
        away["fga_tot"] = pd.to_numeric(df["fga_away"], errors="coerce")

    if "fgm_home" in df.columns and "fgm_away" in df.columns:
        home["fgm_tot"] = pd.to_numeric(df["fgm_home"], errors="coerce")
        away["fgm_tot"] = pd.to_numeric(df["fgm_away"], errors="coerce")

    if "fg3a_home" in df.columns and "fg3a_away" in df.columns:
        home["fg3a_tot"] = pd.to_numeric(df["fg3a_home"], errors="coerce")
        away["fg3a_tot"] = pd.to_numeric(df["fg3a_away"], errors="coerce")

    if "fg3m_home" in df.columns and "fg3m_away" in df.columns:
        home["fg3m_tot"] = pd.to_numeric(df["fg3m_home"], errors="coerce")
        away["fg3m_tot"] = pd.to_numeric(df["fg3m_away"], errors="coerce")

    if "fta_home" in df.columns and "fta_away" in df.columns:
        home["fta_tot"] = pd.to_numeric(df["fta_home"], errors="coerce")
        away["fta_tot"] = pd.to_numeric(df["fta_away"], errors="coerce")

    if "ftm_home" in df.columns and "ftm_away" in df.columns:
        home["ftm_tot"] = pd.to_numeric(df["ftm_home"], errors="coerce")
        away["ftm_tot"] = pd.to_numeric(df["ftm_away"], errors="coerce")

    team_long = pd.concat([home, away], ignore_index=True)

    # keep only real NBA abbreviations
    team_long["team"] = team_long["team"].astype(str).str.upper().str.strip()
    team_long = team_long[team_long["team"].isin(NBA_TEAM_ABBREVS)].copy()

    # per-team percentages
    if {"fgm_tot", "fga_tot"}.issubset(team_long.columns):
        team_long["fg_pct"] = safe_div(team_long["fgm_tot"], team_long["fga_tot"])
    if {"fg3m_tot", "fg3a_tot"}.issubset(team_long.columns):
        team_long["fg3_pct"] = safe_div(team_long["fg3m_tot"], team_long["fg3a_tot"])
    if {"ftm_tot", "fta_tot"}.issubset(team_long.columns):
        team_long["ft_pct"] = safe_div(team_long["ftm_tot"], team_long["fta_tot"])
    if {"fg3a_tot", "fga_tot"}.issubset(team_long.columns):
        team_long["share_3pa"] = safe_div(team_long["fg3a_tot"], team_long["fga_tot"])
        team_long["share_2pa"] = 1 - team_long["share_3pa"]

    # aggregate to season means per team
    team_season = (
        team_long
        .groupby(["team", "season_start_year"], as_index=False)
        .mean(numeric_only=True)
    )

pca_features = [c for c in ["pts_tot", "ast_tot", "reb_tot", "fga_tot", "fg3a_tot", "fta_tot", "fg_pct", "fg3_pct"] if c in league_season]
pca_df, pca_var = None, None
if len(pca_features) >= 3 and len(league_season) > 2:
    X = league_season[pca_features].fillna(0.0).values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(Xs)
    pca_df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "season_start_year": league_season["season_start_year"],
        "era": league_season["era"],
    })
    pca_var = pca.explained_variance_ratio_

missing_before = df_raw.isna().sum().sort_values(ascending=False)
missing_before = missing_before[missing_before > 0].rename("missing").reset_index().rename(columns={"index": "column"})

df_after = df.copy()
fill_zero_cols = [c for c in df_after.columns if any(k in c for k in ["_home", "_away", "_tot"]) and df_after[c].dtype != "O"]
df_after[fill_zero_cols] = df_after[fill_zero_cols].fillna(0)
missing_after = df_after.isna().sum().sort_values(ascending=False)
missing_after = missing_after[missing_after > 0].rename("missing").reset_index().rename(columns={"index": "column"})

MIN_YEAR = int(league_season["season_start_year"].min())
MAX_YEAR = int(league_season["season_start_year"].max())
slider_marks = {y: str(y) for y in range(MIN_YEAR, MAX_YEAR + 1, 5)}

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Lato:wght@300;400;700&display=swap",
]

my_app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server=my_app.server
my_app.title = "Visualizing the Evolution of the NBA (1979–2023)"


# =========================
# HEADER
# =========================
header = html.Div(
    [
        html.Img(
            src="/assets/NBA_logo.png",
            style={"height": "58px", "marginRight": "16px", "borderRadius": "4px"},
        ),
        html.Div(
            [
                html.H1(
                    "Visualizing the Evolution of the NBA (1979–2023)",
                    style={
                        "color": TITLE_BLUE,
                        "fontFamily": "serif",
                        "fontWeight": "bold",
                        "margin": "0",
                    },
                ),
                html.Div(
                    "Data Visualization of Complex Data — DATS 6401",
                    style={
                        "color": TEXT_DARK,
                        "opacity": 0.9,
                        "fontFamily": "serif",
                        "marginTop": "2px",
                    },
                ),
            ],
            style={"display": "flex", "flexDirection": "column"},
        ),
    ],
    style={
        "display": "flex",
        "alignItems": "center",
        "gap": "12px",
        "padding": "10px 20px",
        "backgroundColor": "#ffffff",
        "borderBottom": f"3px solid {TITLE_BLUE}",
    },
)

section_tabs = [
    {"label": "🏠 Home", "value": "home"},
    {"label": "🧹 Data Cleaning & Analysis", "value": "sec-ds"},
    {"label": "🏀 NBA Evolution", "value": "sec-nba"},
]

ds_subtabs = [
    {"label": "Missing Values", "value": "ds-missing"},
    {"label": "Outliers", "value": "ds-outliers"},
    {"label": "Transformations", "value": "ds-transform"},
    {"label": "Correlation", "value": "ds-corr"},
    {"label": "PCA", "value": "ds-pca"},
    {"label": "Normality Testing", "value": "ds-normality"},
    {"label": "Numeric Explorer", "value": "ds-num-explore"},
    {"label": "Categorical Explorer", "value": "ds-cat-explore"},
]

nba_subtabs = [
    {"label": "Overview & Trends", "value": "nba-trends"},
    {"label": "3-Point Revolution", "value": "nba-3pt"},
    {"label": "Home vs Away", "value": "nba-context"},
    {"label": "Outliers & Standouts", "value": "nba-outliers"},
    {"label": "Bar Chart Race", "value": "nba-race"},
    {"label": "📖 Team Story", "value": "nba-story"},
]

controls_row = html.Div(
    [
        html.Div(
            [
                html.Label("Teams (optional — up to 10)", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                dcc.Dropdown(
                    id="team-select",
                    options=(
                        [{"label": t, "value": t} for t in sorted(team_season["team"].unique())]
                        if team_season is not None
                        else []
                    ),
                    value=[],
                    multi=True,
                    style={"width": "320px", "color": "#111"},
                    placeholder="Select team(s) or leave empty for league-only",
                ),
            ]
        ),
        html.Div(
            [
                html.Label("Chart Type", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                dcc.RadioItems(
                    id="chart-type",
                    options=[{"label": " Line", "value": "line"}, {"label": " Bar", "value": "bar"}],
                    value="line",
                    inputStyle={"marginRight": "6px"},
                    style={"color": TEXT_DARK, "fontFamily": BODY_FONT},
                ),
            ]
        ),
    ],
    style={"display": "flex", "gap": "24px", "alignItems": "flex-end"},
)

my_app.layout = html.Div(
    style={"minHeight": "100vh", "backgroundColor": APP_BG},
    children=[
        header,
        dbc.Container(
            [
                dcc.Tabs(
                    id="section-tabs",
                    value="home",
                    children=[
                        dcc.Tab(label=t["label"], value=t["value"], style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE)
                        for t in section_tabs
                    ],
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Season Range (start year)", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                                dcc.RangeSlider(
                                    id="year-range",
                                    min=MIN_YEAR,
                                    max=MAX_YEAR,
                                    value=[MIN_YEAR, MAX_YEAR],
                                    marks=slider_marks,
                                    step=1,
                                    allowCross=False,
                                ),
                            ],
                            style={"flex": 1},
                        ),
                        controls_row,
                    ],
                    id="nba-slider-wrap",
                    style={"display": "none", "padding": "8px 12px", "borderTop": f"1px solid {GRID}"},
                ),
                html.Div(id="section-content", style={"padding": "10px 12px"}),
                html.Div(
                    f"Built for Academic Demonstration — NBA Data {MIN_YEAR}–{MAX_YEAR}",
                    style={
                        "textAlign": "center",
                        "color": LIGHTGREY,
                        "fontFamily": BODY_FONT,
                        "padding": "10px 0",
                        "opacity": 0.9,
                    },
                ),
            ],
            fluid=True,
        ),
    ],
)


def filter_years(df_in: pd.DataFrame, yr):
    lo, hi = yr
    return df_in[(df_in["season_start_year"] >= lo) & (df_in["season_start_year"] <= hi)].copy()


@my_app.callback(Output("nba-slider-wrap", "style"), Input("section-tabs", "value"))
def toggle_slider(section):
    return (
        {"display": "block", "padding": "8px 12px", "borderTop": f"1px solid {GRID}"}
        if section == "sec-nba"
        else {"display": "none"}
    )


@my_app.callback(
    Output("section-content", "children"),
    Input("section-tabs", "value"),
    Input("year-range", "value"),
    Input("team-select", "value"),
    Input("chart-type", "value"),
)
def render_section(section, yr_range, team_val, chart_type):
    if section == "home":
        n_rows, n_cols = df_raw.shape
        n_num = df_raw.select_dtypes(include=[np.number]).shape[1]
        n_cat = n_cols - n_num

        head_tbl = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in df_raw.columns[:8]],
            data=df_raw.head(10).to_dict("records"),
            style_table={"overflowX": "auto", "background": "rgba(0,0,0,0)", "border": f"1.5px solid {LIGHTGREY}"},
            style_header={
                "backgroundColor": "#f0f4ff",
                "color": TITLE_BLUE,
                "fontWeight": "700",
                "fontFamily": "serif",
                "border": "1px solid #dddddd",
            },
            style_cell={
                "fontFamily": "serif",
                "fontSize": 12,
                "color": TEXT_DARK,
                "backgroundColor": "#ffffff",
                "border": "1px solid #dddddd",
                "whiteSpace": "nowrap",
                "textOverflow": "ellipsis",
            },
        )

        desc_num = df_raw.select_dtypes(include=[np.number]).describe().round(2)
        desc_tbl = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in desc_num.columns[:8]],
            data=desc_num.reset_index().to_dict("records"),
            style_table={"overflowX": "auto", "background": "#ffffff", "border": "1px solid #dddddd"},
            style_header={
                "backgroundColor": "#f0f4ff",
                "color": TITLE_BLUE,
                "fontWeight": "700",
                "fontFamily": "serif",
                "border": "1px solid #dddddd",
            },
            style_cell={
                "fontFamily": "serif",
                "fontSize": 12,
                "color": TEXT_DARK,
                "backgroundColor": "#ffffff",
                "border": "1px solid #dddddd",
                "whiteSpace": "nowrap",
                "textOverflow": "ellipsis",
            },
        )

        return html.Div(
            [
                card(
                    html.H3("OVERVIEW & DATASET DESCRIPTION", style={"fontFamily": "serif", "color": TEXT_DARK}),
                    html.P(
                        f"Dataset size: {n_rows:,} rows × {n_cols} columns. Numeric features: {n_num}; Categorical features: {n_cat}.",
                        style={"fontFamily": BODY_FONT, "color": LIGHTGREY},
                    ),
                ),
                card(html.H4("SAMPLE ROWS", style={"fontFamily": "serif", "color": TEXT_DARK}), head_tbl),
                card(html.H4("NUMERIC SUMMARY (FIRST 8)", style={"fontFamily": "serif", "color": TEXT_DARK}), desc_tbl),
            ]
        )

    if section == "sec-ds":
        return html.Div(
            [
                html.H3("DATA CLEANING & ANALYSIS", style={"fontFamily": "serif", "color": TEXT_DARK}),
                dcc.Tabs(
                    id="ds-tabs",
                    value="ds-missing",
                    children=[
                        dcc.Tab(label=t["label"], value=t["value"], style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE)
                        for t in ds_subtabs
                    ],
                ),
                # ✅ KEY CHANGE: section background + border like your screenshot
                html.Div(id="ds-tab-content", style=SECTION_CONTAINER_STYLE),
            ]
        )

    # sec-nba
    return html.Div(
        [
            html.H3("NBA EVOLUTION (LONG-RUN)", style={"fontFamily": "serif", "color": TEXT_DARK}),
            dcc.Tabs(
                id="nba-tabs",
                value="nba-trends",
                children=[
                    dcc.Tab(label=t["label"], value=t["value"], style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE)
                    for t in nba_subtabs
                ],
            ),
            # ✅ apply same panel styling to NBA section for consistency
            html.Div(id="nba-tab-content", style=SECTION_CONTAINER_STYLE),
        ]
    )


@my_app.callback(Output("ds-tab-content", "children"), Input("ds-tabs", "value"))
def render_ds_tab(tab):
    if tab == "ds-missing":
        fig_before = px.bar(missing_before, x="column", y="missing", title="Missing Values — BEFORE Cleaning")
        style_fig_app(fig_before, x_title="Column", y_title="Missing Count").update_xaxes(tickangle=70)

        if len(missing_after) > 0:
            fig_after = px.bar(missing_after, x="column", y="missing", title="Missing Values — AFTER Simple Imputation")
            style_fig_app(fig_after, x_title="Column", y_title="Missing Count").update_xaxes(tickangle=70)
        else:
            fig_after = go.Figure()
            fig_after.add_annotation(
                text="No missing values remain after simple imputations.",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color=TEXT_DARK),
            )
            style_fig_app(fig_after)

        desc = html.P(
            "Numeric made/attempt stats are filled with 0 so totals aggregate cleanly. Percentages remain NaN when attempts are zero.",
            style={"fontFamily": BODY_FONT, "color": LIGHTGREY},
        )
        return [card(dcc.Graph(figure=fig_before)), card(dcc.Graph(figure=fig_after)), card(desc)]

    if tab == "ds-outliers":
        if "pts_tot" not in df.columns:
            return card(html.Div("Outlier view needs pts_tot.", style={"fontFamily": BODY_FONT, "color": TEXT_DARK}))
        fig = px.box(df, y="pts_tot", points="all", title="Outliers — Total Points (Two-Team per Game)")
        style_fig_app(fig, y_title="PTS (two-team per game)")
        explain = html.P(
            "Box shows the middle 50%. Points beyond whiskers are statistical outliers in scoring.",
            style={"fontFamily": BODY_FONT, "color": LIGHTGREY},
        )
        return [card(dcc.Graph(figure=fig)), card(explain)]

    if tab == "ds-transform":
        options = []
        for col, label in [("pts_tot", "Total Points"), ("ast_tot", "Assists"), ("reb_tot", "Rebounds")]:
            if col in league_season.columns:
                options.append({"label": label, "value": col})
        if not options:
            return card(html.Div("No suitable season-level numeric features found.", style={"fontFamily": BODY_FONT, "color": TEXT_DARK}))

        controls = dbc.Col(
            [
                html.H5("Select feature", style={"fontFamily": BODY_FONT, "color": TEXT_DARK}),
                dcc.RadioItems(
                    id="ds-transform-feature",
                    options=options,
                    value=options[0]["value"],
                    style={"color": TEXT_DARK, "fontFamily": BODY_FONT},
                ),
                html.Br(),
                html.H5("Data to plot", style={"fontFamily": BODY_FONT, "color": TEXT_DARK}),
                dcc.RadioItems(
                    id="ds-transform-view",
                    options=[
                        {"label": " Raw only", "value": "raw"},
                        {"label": " Z-transformed only", "value": "z"},
                        {"label": " Both", "value": "both"},
                    ],
                    value="both",
                    style={"color": TEXT_DARK, "fontFamily": BODY_FONT},
                ),
                html.Br(),
                html.P(
                    "Z-transformation standardizes the feature to mean 0 and std 1.",
                    style={"fontFamily": BODY_FONT, "color": LIGHTGREY, "fontSize": "13px"},
                ),
            ],
            md=4,
        )

        graph_col = dbc.Col(
            [
                dbc.Card(
                    [
                        dbc.CardHeader("Raw vs Z-Transformed League Averages", style={"fontFamily": BODY_FONT, "fontWeight": "600"}),
                        dbc.CardBody([dcc.Graph(id="ds-transform-fig")]),
                    ],
                    style={"border": f"1px solid #dddddd", "backgroundColor": "#ffffff"},
                )
            ],
            md=8,
        )

        return card(dbc.Row([controls, graph_col], className="g-3"))

    if tab == "ds-corr":
        corr_cols = [
            c for c in [
                "season_start_year", "pts_tot", "ast_tot", "reb_tot", "fga_tot", "fgm_tot",
                "fg3a_tot", "fg3m_tot", "fta_tot", "ftm_tot", "fg_pct", "fg3_pct", "ft_pct",
                "share_3pa", "share_2pa", "pts_home_team", "pts_away_team",
            ] if c in league_season.columns
        ]
        corr = league_season[corr_cols].corr(numeric_only=True).round(2)
        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            title="Correlation Heatmap (Season-Level)",
            aspect="auto",
        )
        fig.update_layout(width=900, height=600)
        style_fig_app(fig, x_title="", y_title="")
        fig.update_xaxes(side="bottom", tickangle=45)
        fig.update_yaxes(automargin=True)
        return card(dcc.Graph(figure=fig))

    if tab == "ds-pca":
        if pca_df is None:
            return card(html.Div("PCA not available.", style={"fontFamily": BODY_FONT, "color": TEXT_DARK}))
        subtitle = ""
        if pca_var is not None:
            subtitle = f"Explained variance: PC1={pca_var[0] * 100:.2f}% | PC2={pca_var[1] * 100:.2f}%"
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="era",
            hover_data=["season_start_year"],
            title="PCA (Season-Level): Similar Statistical Profiles Cluster Together",
        )
        style_fig_app(fig, x_title="PC1", y_title="PC2")
        return [card(dcc.Graph(figure=fig)), card(html.Small(subtitle, style={"fontFamily": BODY_FONT, "color": LIGHTGREY}))]

    if tab == "ds-normality":
        candidate_cols = [
            c for c in [
                "pts_tot","ast_tot","reb_tot","fga_tot","fgm_tot","fg3a_tot","fg3m_tot",
                "fta_tot","ftm_tot","fg_pct","fg3_pct","ft_pct","share_3pa","share_2pa"
            ] if c in df.columns
        ]
        if not candidate_cols:
            return card(html.Div("No suitable numeric columns found.", style={"fontFamily": BODY_FONT, "color": TEXT_DARK}))

        left = dbc.Col(
            [
                dbc.Card(
                    [
                        dbc.CardHeader("QQ Plot (Visual Check)", style={"fontFamily": BODY_FONT, "fontWeight": "600"}),
                        dbc.CardBody(
                            [
                                html.Label("QQ feature", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                                dcc.Dropdown(
                                    id="ds-qq-feature",
                                    options=[{"label": c, "value": c} for c in candidate_cols],
                                    value=candidate_cols[0],
                                    clearable=False,
                                    style={"color": "#111", "marginBottom": "10px"},
                                ),
                                dcc.Graph(id="ds-qq-fig"),
                                html.P(id="ds-qq-explain", style={"fontFamily": BODY_FONT, "color": LIGHTGREY, "marginTop": "8px"}),
                            ]
                        ),
                    ],
                    style={"border": "1px solid #dddddd", "backgroundColor": "#ffffff"},
                )
            ],
            md=6,
        )

        right = dbc.Col(
            [
                dbc.Card(
                    [
                        dbc.CardHeader("Formal Normality Tests (α = 0.01)", style={"fontFamily": BODY_FONT, "fontWeight": "600"}),
                        dbc.CardBody(
                            [
                                html.Label("Test feature", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                                dcc.Dropdown(
                                    id="ds-norm-feature",
                                    options=[{"label": c, "value": c} for c in candidate_cols],
                                    value=candidate_cols[0],
                                    clearable=False,
                                    style={"color": "#111", "marginBottom": "10px"},
                                ),
                                html.Label("Test type", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                                dcc.RadioItems(
                                    id="ds-norm-test-type",
                                    options=[
                                        {"label": " K-S test (fit μ, σ)", "value": "ks"},
                                        {"label": " Shapiro-Wilk", "value": "shapiro"},
                                        {"label": " D'Agostino K²", "value": "dagostino"},
                                    ],
                                    value="ks",
                                    style={"color": TEXT_DARK, "fontFamily": BODY_FONT},
                                ),
                                html.Hr(style={"borderColor": GRID}),
                                html.Pre(id="ds-norm-result", style={"fontSize": "14px", "color": TEXT_DARK, "fontFamily": BODY_FONT}),
                                html.Small(
                                    "Shapiro can be slow for very large N; the app samples for responsiveness.",
                                    style={"color": LIGHTGREY, "fontFamily": BODY_FONT},
                                ),
                            ]
                        ),
                    ],
                    style={"border": "1px solid #dddddd", "backgroundColor": "#ffffff"},
                )
            ],
            md=6,
        )

        return card(dbc.Row([left, right], className="g-3"))

    if tab == "ds-num-explore":
        analysis_num, id_like = split_numeric_columns_for_analysis(df)
        num_cols = analysis_num
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist() + id_like

        if not num_cols:
            return card(html.Div("No numeric features available.", style={"fontFamily": BODY_FONT, "color": TEXT_DARK}))

        plot_options = [
            {"label": "Line plot", "value": "line"},
            {"label": "Histogram + KDE", "value": "hist_kde"},
            {"label": "Box plot", "value": "box"},
            {"label": "Violin plot", "value": "violin"},
            {"label": "Strip plot", "value": "strip"},
            {"label": "Swarm-like plot", "value": "swarm"},
            {"label": "Area plot", "value": "area"},
            {"label": "Hexbin / 2D density", "value": "hexbin"},
            {"label": "2D KDE / Contour", "value": "kde2d"},
            {"label": "QQ-plot", "value": "qq"},
            {"label": "Regression (scatter + line)", "value": "reg"},
        ]

        controls = dbc.Col(
            [
                html.H5("Numeric Plot Explorer", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                html.Label("X (numeric)", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                dcc.Dropdown(
                    id="num-explore-x",
                    options=[{"label": c, "value": c} for c in num_cols],
                    value=num_cols[0],
                    clearable=False,
                    style={"marginBottom": "8px", "color": "#111"},
                ),
                html.Label("Y (numeric – for bivariate plots)", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                dcc.Dropdown(
                    id="num-explore-y",
                    options=[{"label": c, "value": c} for c in num_cols],
                    value=num_cols[1] if len(num_cols) > 1 else None,
                    clearable=True,
                    style={"marginBottom": "8px", "color": "#111"},
                ),
                html.Label("Hue / Color (categorical)", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                dcc.Dropdown(
                    id="num-explore-hue",
                    options=[{"label": c, "value": c} for c in cat_cols],
                    value=cat_cols[0] if cat_cols else None,
                    clearable=True,
                    style={"marginBottom": "8px", "color": "#111"},
                ),
                html.Label("Plot type", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                dcc.Dropdown(
                    id="num-explore-plot-type",
                    options=plot_options,
                    value="hist_kde",
                    clearable=False,
                    style={"marginBottom": "8px", "color": "#111"},
                ),
            ],
            md=3,
        )

        graph_col = dbc.Col(
            [
                dbc.Card(
                    [
                        dbc.CardHeader("Numeric Explorer", style={"fontFamily": BODY_FONT}),
                        dbc.CardBody(
                            [
                                dcc.Graph(id="num-explore-fig"),
                                html.Br(),
                                html.P(id="num-explore-explain", style={"fontFamily": BODY_FONT, "color": LIGHTGREY}),
                            ]
                        ),
                    ],
                    style={"backgroundColor": "#ffffff", "border": "1px solid #dddddd"},
                )
            ],
            md=9,
        )

        return card(dbc.Row([controls, graph_col], className="g-3"))

    if tab == "ds-cat-explore":
        analysis_num, id_like = split_numeric_columns_for_analysis(df)
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist() + id_like
        num_cols = analysis_num

        if not cat_cols:
            return card(html.Div("No categorical features available.", style={"fontFamily": BODY_FONT, "color": TEXT_DARK}))

        plot_options = [
            {"label": "Count plot", "value": "count"},
            {"label": "Grouped bar plot", "value": "bar_group"},
            {"label": "Stacked bar plot", "value": "bar_stack"},
            {"label": "Pie chart", "value": "pie"},
            {"label": "Strip plot (numeric vs category)", "value": "strip"},
            {"label": "Box plot (numeric vs category)", "value": "box"},
            {"label": "Violin plot (numeric vs category)", "value": "violin"},
        ]

        controls = dbc.Col(
            [
                html.H5("Categorical Plot Explorer", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                html.Label("Category (x-axis)", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                dcc.Dropdown(
                    id="cat-explore-x",
                    options=[{"label": c, "value": c} for c in cat_cols],
                    value=cat_cols[0],
                    clearable=False,
                    style={"marginBottom": "8px", "color": "#111"},
                ),
                html.Label("Numeric value", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                dcc.Dropdown(
                    id="cat-explore-y",
                    options=[{"label": c, "value": c} for c in num_cols],
                    value=num_cols[0] if num_cols else None,
                    clearable=True,
                    style={"marginBottom": "8px", "color": "#111"},
                ),
                html.Label("Hue / Color (second category)", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                dcc.Dropdown(
                    id="cat-explore-hue",
                    options=[{"label": c, "value": c} for c in cat_cols],
                    value=cat_cols[1] if len(cat_cols) > 1 else None,
                    clearable=True,
                    style={"marginBottom": "8px", "color": "#111"},
                ),
                html.Label("Plot type", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                dcc.Dropdown(
                    id="cat-explore-plot-type",
                    options=plot_options,
                    value="count",
                    clearable=False,
                    style={"marginBottom": "8px", "color": "#111"},
                ),
            ],
            md=3,
        )

        graph_col = dbc.Col(
            [
                dbc.Card(
                    [
                        dbc.CardHeader("Categorical Explorer", style={"fontFamily": BODY_FONT}),
                        dbc.CardBody(
                            [
                                dcc.Graph(id="cat-explore-fig"),
                                html.Br(),
                                html.P(id="cat-explore-explain", style={"fontFamily": BODY_FONT, "color": LIGHTGREY}),
                            ]
                        ),
                    ],
                    style={"backgroundColor": "#ffffff", "border": "1px solid #dddddd"},
                )
            ],
            md=9,
        )

        return card(dbc.Row([controls, graph_col], className="g-3"))

    return html.Div()


@my_app.callback(Output("ds-qq-fig", "figure"), Output("ds-qq-explain", "children"), Input("ds-qq-feature", "value"))
def update_ds_qq(feature):
    if feature is None or feature not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Select a valid feature for QQ plot.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
        style_fig_app(fig)
        return fig, ""

    vals = df[feature].dropna().astype(float).values
    vals = np.sort(vals)
    n = len(vals)
    if n < 8:
        fig = go.Figure()
        fig.add_annotation(text="Not enough points for QQ plot (need ~8+).", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
        style_fig_app(fig)
        return fig, "Pick a feature with more non-missing values."

    q = (np.arange(1, n + 1) - 0.5) / n
    zq = np.sqrt(2) * erfinv(2 * q - 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=zq, y=vals, mode="markers", name=feature))

    mu, sd = vals.mean(), vals.std(ddof=1)
    if sd > 0 and not np.isnan(sd):
        line_x = np.array([zq.min(), zq.max()])
        fig.add_trace(go.Scatter(x=line_x, y=mu + sd * line_x, mode="lines", name="Reference Line", line=dict(color=ORANGE)))

    fig.update_layout(title=f"QQ Plot — {feature}")
    fig = style_fig_app(fig, x_title="Theoretical Normal Quantiles", y_title="Observed Values")
    return fig, "If points follow the line closely, the feature looks roughly normal. Curvature suggests skew/heavy tails."


@my_app.callback(Output("ds-norm-result", "children"), Input("ds-norm-feature", "value"), Input("ds-norm-test-type", "value"))
def update_ds_normality_test(feature, test_type):
    alpha = 0.01
    if feature is None or feature not in df.columns:
        return "Select a valid feature."

    series = df[feature].dropna().astype(float)
    n = len(series)
    if n < 8:
        return "Not enough data to run a reliable normality test."

    series_for_shapiro = series
    if n > 5000:
        series_for_shapiro = series.sample(5000, random_state=42)

    try:
        if test_type == "ks":
            mu = series.mean()
            sigma = series.std(ddof=1)
            if sigma == 0 or np.isnan(sigma):
                return "Std dev is 0 — K-S test not meaningful for a constant series."
            stat, pval = stats.kstest(series, "norm", args=(mu, sigma))
            test_name = "K-S test (with fitted μ, σ)"
            used_n = n
        elif test_type == "shapiro":
            stat, pval = stats.shapiro(series_for_shapiro)
            test_name = "Shapiro-Wilk"
            used_n = len(series_for_shapiro)
        else:
            stat, pval = stats.normaltest(series)
            test_name = "D'Agostino K²"
            used_n = n

        decision = "PASS (looks normal)" if pval >= alpha else "FAIL (not normal)"
        return "\n".join(
            [
                f"{test_name} on {feature}",
                f"N used = {used_n:,} (raw N={n:,})",
                f"Test statistic = {stat:.4f}",
                f"p-value = {pval:.6f}",
                f"α = {alpha} → {decision}",
                "",
                "Interpretation:",
                "If p ≥ α → do NOT reject normality.",
                "If p < α → reject normality (distribution deviates from normal).",
            ]
        )
    except Exception as e:
        return f"Error running test: {str(e)}"


@my_app.callback(Output("ds-transform-fig", "figure"), Input("ds-transform-feature", "value"), Input("ds-transform-view", "value"))
def update_ds_transform(feature, view):
    if feature is None:
        feature = "pts_tot"
    s = league_season[["season_start_year", feature]].dropna().copy()
    s = s.sort_values("season_start_year")
    x_vals = s["season_start_year"]
    y_raw = s[feature]
    y_z = z_transform(y_raw)

    fig = go.Figure()
    if view in ["raw", "both"]:
        fig.add_trace(go.Scatter(x=x_vals, y=y_raw, mode="lines+markers", name="Raw"))
    if view in ["z", "both"]:
        fig.add_trace(go.Scatter(x=x_vals, y=y_z, mode="lines+markers", name="Z-transformed"))

    y_label_map = {
        "pts_tot": "Total Points (Two-Team per Game)",
        "ast_tot": "Assists (Two-Team per Game)",
        "reb_tot": "Rebounds (Two-Team per Game)",
    }
    stat_label = y_label_map.get(feature, feature)
    fig.update_layout(title=f"Raw vs Z-Transformed — {stat_label}")
    fig = style_fig_app(fig, x_title="Season Start Year", y_title=stat_label if view != "z" else f"{stat_label} (Z-score)")
    fig.update_yaxes(tickformat=".2f")
    return fig


@my_app.callback(
    Output("num-explore-fig", "figure"),
    Output("num-explore-explain", "children"),
    Input("num-explore-plot-type", "value"),
    Input("num-explore-x", "value"),
    Input("num-explore-y", "value"),
    Input("num-explore-hue", "value"),
)
def update_num_explorer(plot_type, x, y, hue):
    try:
        if isinstance(x, (list, tuple)):
            x = x[0] if len(x) else None
        if isinstance(y, (list, tuple)):
            y = y[0] if len(y) else None
        if isinstance(hue, (list, tuple)):
            hue = hue[0] if len(hue) else None

        if x is None or x not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="Please select a valid X feature.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
            fig = style_fig_app(fig)
            return fig, ""

        needs_y = plot_type in ["hexbin", "kde2d", "reg"]
        if needs_y and (y is None or y not in df.columns):
            fig = go.Figure()
            fig.add_annotation(text="This plot needs BOTH X and Y.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
            fig = style_fig_app(fig)
            return fig, "Select a valid numeric Y variable."

        cols_needed = [x]
        if y is not None and plot_type in ["line", "hexbin", "kde2d", "reg"]:
            cols_needed.append(y)
        if hue is not None and hue in df.columns:
            cols_needed.append(hue)

        cols_needed = [c for c in cols_needed if c in df.columns]
        data = df.dropna(subset=cols_needed).copy()

        if data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available after dropping NaNs.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
            fig = style_fig_app(fig)
            return fig, "No valid rows for the selected variables."

        explain = ""

        if plot_type == "line":
            if "season_start_year" in data.columns:
                data = data.sort_values("season_start_year")
                y_col = x if y is None else y
                fig = px.line(data, x="season_start_year", y=y_col, color=hue, markers=True, title="Line Plot")
                fig = style_fig_app(fig, x_title="Season Start Year", y_title=y_col)
                explain = "Line plot shows change over time; add Hue to split groups."
            else:
                data = data.sort_values(x)
                y_col = y if y is not None else x
                fig = px.line(data, x=x, y=y_col, color=hue, markers=True, title="Line Plot")
                fig = style_fig_app(fig, x_title=x, y_title=y_col)
                explain = "Line plot shows relationship across X."

        elif plot_type == "hist_kde":
            fig = px.histogram(
                data, x=x, color=hue, nbins=40,
                histnorm="probability density", marginal="rug",
                title=f"Histogram + KDE: {x}"
            )
            fig.update_traces(opacity=0.7)
            fig = style_fig_app(fig, x_title=x, y_title="Density")
            explain = "Histogram shows distribution shape; KDE gives smooth density."

        elif plot_type == "box":
            if hue:
                fig = px.box(data, x=hue, y=x, points="all", title=f"Box Plot: {x} by {hue}")
                fig = style_fig_app(fig, x_title=hue, y_title=x)
                explain = "Box plot compares distributions across categories and highlights outliers."
            else:
                fig = px.box(data, y=x, points="all", title=f"Box Plot: {x}")
                fig = style_fig_app(fig, x_title="", y_title=x)
                explain = "Box plot summarizes distribution and outliers for the selected variable."

        elif plot_type == "violin":
            if hue:
                fig = px.violin(data, x=hue, y=x, color=hue, box=True, points="all", title=f"Violin Plot: {x} by {hue}")
                fig = style_fig_app(fig, x_title=hue, y_title=x)
                explain = "Violin plot shows the full distribution shape for each category."
            else:
                fig = px.violin(data, y=x, box=True, points="all", title=f"Violin Plot: {x}")
                fig = style_fig_app(fig, x_title="", y_title=x)
                explain = "Violin plot shows distribution shape + box summary."

        elif plot_type == "strip":
            if hue:
                fig = px.strip(data, x=hue, y=x, color=hue, title=f"Strip Plot: {x} by {hue}")
                fig = style_fig_app(fig, x_title=hue, y_title=x)
                explain = "Strip plot shows individual observations per category."
            else:
                data["_all_"] = "All"
                fig = px.strip(data, x="_all_", y=x, title=f"Strip Plot: {x}")
                fig = style_fig_app(fig, x_title="", y_title=x)
                explain = "Strip plot shows individual observations (all points in one group)."

        elif plot_type == "swarm":
            if hue:
                fig = px.strip(data, x=hue, y=x, color=hue, title=f"Swarm-like Plot: {x} by {hue}")
                fig.update_traces(jitter=0.35)
                fig = style_fig_app(fig, x_title=hue, y_title=x)
                explain = "Jitter reduces overlap so point density becomes visible across categories."
            else:
                data["_all_"] = "All"
                fig = px.strip(data, x="_all_", y=x, title=f"Swarm-like Plot: {x}")
                fig.update_traces(jitter=0.35)
                fig = style_fig_app(fig, x_title="", y_title=x)
                explain = "Swarm-like view using jitter to reduce overlap (single group)."

        elif plot_type == "area":
            if "season_start_year" not in data.columns:
                fig = go.Figure()
                fig.add_annotation(text="Area plot needs season_start_year.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
                fig = style_fig_app(fig)
                return fig, "Area plot needs a time axis."
            data = data.sort_values("season_start_year")
            fig = px.area(data, x="season_start_year", y=x, color=hue, title=f"Area Plot: {x}")
            fig = style_fig_app(fig, x_title="Season Start Year", y_title=x)
            explain = "Area plot emphasizes levels over time."

        elif plot_type == "hexbin":
            fig = px.density_heatmap(data, x=x, y=y, nbinsx=30, nbinsy=30, title=f"2D Density: {x} vs {y}")
            fig = style_fig_app(fig, x_title=x, y_title=y)
            explain = "2D density heatmap shows concentration of points."

        elif plot_type == "kde2d":
            use_cols = [x, y] + ([hue] if hue else [])
            d2 = data[use_cols].copy()
            d2[x] = pd.to_numeric(d2[x], errors="coerce")
            d2[y] = pd.to_numeric(d2[y], errors="coerce")
            d2 = d2.dropna(subset=[x, y])

            if d2.empty:
                fig = go.Figure()
                fig.add_annotation(text="No valid numeric data after cleaning.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
                fig = style_fig_app(fig)
                return fig, "Try different X/Y variables."

            ux = d2[x].nunique(dropna=True)
            uy = d2[y].nunique(dropna=True)
            if ux < 8 or uy < 8:
                fig = go.Figure()
                fig.add_annotation(text=f"Not enough variation for KDE contour.\nUnique({x})={ux}, Unique({y})={uy}", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
                fig = style_fig_app(fig)
                return fig, "Pick variables with more spread; avoid near-constant columns."

            if len(d2) > 20000:
                d2 = d2.sample(20000, random_state=42)

            title = f"2D KDE / Contour: {x} vs {y}"
            if hue:
                d2[hue] = d2[hue].astype(str).str.strip()
                d2 = d2[d2[hue] != ""]
                if d2.empty:
                    fig = go.Figure()
                    fig.add_annotation(text="Hue column became empty after cleaning.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
                    fig = style_fig_app(fig)
                    return fig, "Try a different Hue or remove Hue."
                fig = px.density_contour(d2, x=x, y=y, facet_col=hue, facet_col_wrap=2, title=title)
            else:
                fig = px.density_contour(d2, x=x, y=y, title=title)

            fig.update_traces(ncontours=25, contours_coloring="lines", line_width=2, showscale=False)

            if not hue:
                fig.add_trace(go.Scatter(x=d2[x], y=d2[y], mode="markers", marker=dict(size=3, opacity=0.20), name="Points"))

            fig = style_fig_app(fig, x_title=x, y_title=y)
            fig.update_xaxes(tickformat=".2f")
            fig.update_yaxes(tickformat=".2f")
            explain = "TRUE 2D KDE contour: tighter rings = higher density. If weird, X/Y may be near-constant."

        elif plot_type == "qq":
            vals = np.sort(pd.to_numeric(data[x], errors="coerce").dropna().values.astype(float))
            n = len(vals)
            if n < 8:
                fig = go.Figure()
                fig.add_annotation(text="Not enough data for QQ-plot.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
                fig = style_fig_app(fig)
                return fig, "QQ-plot requires more data."

            q = (np.arange(1, n + 1) - 0.5) / n
            zq = np.sqrt(2) * erfinv(2 * q - 1)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=zq, y=vals, mode="markers", name=x))

            mu, sd = vals.mean(), vals.std(ddof=1)
            if sd > 0 and not np.isnan(sd):
                line_x = np.array([zq.min(), zq.max()])
                fig.add_trace(go.Scatter(x=line_x, y=mu + sd * line_x, mode="lines", name="Reference", line=dict(color=ORANGE)))

            fig.update_layout(title=f"QQ-plot: {x}")
            fig = style_fig_app(fig, x_title="Theoretical Normal Quantiles", y_title=x)
            explain = "QQ-plot compares distribution to normal; curvature indicates non-normality."

        elif plot_type == "reg":
            d2 = data[[x, y] + ([hue] if hue else [])].copy()
            d2[x] = pd.to_numeric(d2[x], errors="coerce")
            d2[y] = pd.to_numeric(d2[y], errors="coerce")
            d2 = d2.dropna(subset=[x, y])

            if d2.empty:
                fig = go.Figure()
                fig.add_annotation(text="Not enough numeric data for regression after cleaning.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
                fig = style_fig_app(fig)
                return fig, "Pick different numeric variables."

            if len(d2) > 15000:
                d2 = d2.sample(15000, random_state=42)

            fig = px.scatter(d2, x=x, y=y, color=hue, title=f"Regression (scatter + line): {y} vs {x}", color_discrete_sequence=BRIGHT_QUAL)

            xs = d2[x].values.astype(float)
            ys = d2[y].values.astype(float)
            if len(xs) >= 2:
                m, b = np.polyfit(xs, ys, 1)
                x_line = np.linspace(xs.min(), xs.max(), 200)
                y_line = m * x_line + b
                fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Fit line", line=dict(width=3, color=ORANGE)))

            fig = style_fig_app(fig, x_title=x, y_title=y)
            fig.update_xaxes(tickformat=".2f")
            fig.update_yaxes(tickformat=".2f")
            explain = "Scatter shows relationship; line is a simple least-squares fit."

        else:
            fig = go.Figure()
            fig.add_annotation(text="Unknown plot type.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
            fig = style_fig_app(fig)
            explain = "Pick a different plot type."

        return fig, explain

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Numeric Explorer error: {type(e).__name__}: {str(e)}", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
        fig = style_fig_app(fig)
        return fig, "An error occurred. Try different variables/plot type."


@my_app.callback(
    Output("cat-explore-fig", "figure"),
    Output("cat-explore-explain", "children"),
    Input("cat-explore-plot-type", "value"),
    Input("cat-explore-x", "value"),
    Input("cat-explore-y", "value"),
    Input("cat-explore-hue", "value"),
)
def update_cat_explorer(plot_type, x, y, hue):
    if x is None:
        fig = go.Figure()
        fig.add_annotation(text="Please select a categorical feature.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
        style_fig_app(fig)
        return fig, ""

    cols_needed = [x]
    if y is not None and plot_type not in ["count", "pie"]:
        cols_needed.append(y)
    if hue is not None:
        cols_needed.append(hue)
    cols_needed = [c for c in cols_needed if c in df.columns]

    data = df.dropna(subset=cols_needed).copy()
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available after dropping NaNs.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
        style_fig_app(fig)
        return fig, "No valid rows for the selected combination."

    explain = ""

    if plot_type == "count":
        vc = data[x].value_counts().reset_index()
        vc.columns = [x, "count"]
        fig = px.bar(vc, x=x, y="count", text="count", title=f"Count: {x}", color_discrete_sequence=[ORANGE])
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig = style_fig_app(fig, x_title=x, y_title="Count")
        explain = "Counts per category."

    elif plot_type == "bar_group":
        if y is None:
            fig = go.Figure()
            fig.add_annotation(text="Grouped bar needs numeric Y.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
            style_fig_app(fig)
            return fig, "Select a numeric Y."

        group_cols = [x] + ([hue] if hue else [])
        agg = data.groupby(group_cols, as_index=False)[y].mean()

        fig = px.bar(agg, x=x, y=y, color=hue, barmode="group", title=f"Grouped Bar (mean): {y} by {x}", color_discrete_sequence=BRIGHT_QUAL)
        fig = style_fig_app(fig, x_title=x, y_title=y)
        fig.update_yaxes(tickformat=".2f")
        explain = "Uses MEAN aggregation so percentage metrics remain interpretable."

    elif plot_type == "bar_stack":
        if y is None:
            fig = go.Figure()
            fig.add_annotation(text="Stacked bar needs numeric Y.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
            style_fig_app(fig)
            return fig, "Select a numeric Y."

        group_cols = [x] + ([hue] if hue else [])
        agg = data.groupby(group_cols, as_index=False)[y].mean()

        fig = px.bar(agg, x=x, y=y, color=hue, barmode="stack", title=f"Stacked Bar (mean): {y} by {x}", color_discrete_sequence=BRIGHT_QUAL)
        fig = style_fig_app(fig, x_title=x, y_title=y)
        fig.update_yaxes(tickformat=".2f")
        explain = "Stacked bars show average levels by category; mean prevents inflated totals."

    elif plot_type == "pie":
        vc = data[x].value_counts().reset_index()
        vc.columns = [x, "count"]
        fig = px.pie(vc, names=x, values="count", title=f"Pie: {x}", color_discrete_sequence=BRIGHT_QUAL)
        fig.update_traces(texttemplate="%{percent:.2f}", textposition="inside")
        fig = style_fig_app(fig)
        explain = "Proportions by category."

    elif plot_type == "strip":
        if y is None:
            fig = go.Figure()
            fig.add_annotation(text="Strip plot needs numeric Y.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
            style_fig_app(fig)
            return fig, "Select a numeric Y."
        fig = px.strip(data, x=x, y=y, color=hue, title=f"Strip: {y} across {x}")
        fig = style_fig_app(fig, x_title=x, y_title=y)
        explain = "Shows individual values per category."

    elif plot_type == "box":
        if y is None:
            fig = go.Figure()
            fig.add_annotation(text="Box plot needs numeric Y.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
            style_fig_app(fig)
            return fig, "Select a numeric Y."
        fig = px.box(data, x=x, y=y, color=hue, points="all", title=f"Box: {y} by {x}")
        fig = style_fig_app(fig, x_title=x, y_title=y)
        explain = "Distribution + outliers per category."

    elif plot_type == "violin":
        if y is None:
            fig = go.Figure()
            fig.add_annotation(text="Violin plot needs numeric Y.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
            style_fig_app(fig)
            return fig, "Select a numeric Y."
        fig = px.violin(data, x=x, y=y, color=hue, box=True, points="all", title=f"Violin: {y} by {x}")
        fig = style_fig_app(fig, x_title=x, y_title=y)
        explain = "Density + box summary per category."

    else:
        fig = go.Figure()
        fig.add_annotation(text="Unknown plot type.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
        style_fig_app(fig)

    return fig, explain


@my_app.callback(
    Output("nba-tab-content", "children"),
    Input("nba-tabs", "value"),
    Input("year-range", "value"),
    Input("team-select", "value"),
    Input("chart-type", "value"),
)
def render_nba_tab(tab, yr, team_val, chart_type):
    leag = filter_years(league_season, yr)

    def line_or_bar(df_in, y_col, title, y_label):
        df_sorted = df_in.sort_values("season_start_year")
        if chart_type == "bar":
            fig_local = px.bar(df_sorted, x="season_start_year", y=y_col, title=title)
            style_fig_app(fig_local, x_title="Season Start Year", y_title=y_label)
            fig_local.update_traces(marker_color=ORANGE, name="League Average")
        else:
            fig_local = px.line(df_sorted, x="season_start_year", y=y_col, markers=True, title=title)
            style_fig_app(fig_local, x_title="Season Start Year", y_title=y_label)
            fig_local.update_traces(line=dict(color=ORANGE, width=3), marker=dict(color=ORANGE, size=6), name="League Average")

        team_list = (team_val or [])[:10]
        if team_season is not None and team_list:
            for t in team_list:
                tdf = filter_years(team_season[team_season["team"] == t], yr)
                if y_col in tdf.columns and not tdf.empty:
                    fig_local.add_trace(go.Scatter(x=tdf["season_start_year"], y=tdf[y_col], mode="lines+markers", name=t, line=dict(width=2), marker=dict(size=5)))
        return fig_local

    if tab == "nba-trends":
        metric_options = [{"label": col, "value": col} for col in ["pts_tot", "ast_tot", "reb_tot", "fg3a_tot", "fg3_pct", "share_3pa"] if col in leag.columns]
        if not metric_options:
            return card(html.Div("No trend metrics found.", style={"fontFamily": BODY_FONT, "color": TEXT_DARK}))

        return card(
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Choose metric", style={"color": TEXT_DARK, "fontFamily": BODY_FONT}),
                            dcc.Dropdown(
                                id="nba-trends-metric",
                                options=metric_options,
                                value=metric_options[0]["value"],
                                clearable=False,
                                style={"width": "320px", "color": "#111"},
                            ),
                        ],
                        style={"marginBottom": "10px"},
                    ),
                    dcc.Graph(id="nba-trends-fig"),
                    html.Small("Tip: Add teams from the top dropdown to overlay team lines on the league trend.", style={"fontFamily": BODY_FONT, "color": LIGHTGREY}),
                ]
            )
        )

    if tab == "nba-3pt":
        blocks = []
        if "fg3a_tot" in leag.columns:
            fig = line_or_bar(leag, "fg3a_tot", "3PA per Game (Two-Team)", "FG3A Per Game")
            fig.add_vline(x=1980, line_dash="dot", annotation_text="3PT Introduced", annotation_position="top left", line=dict(color=LIGHTGREY))
            fig.add_vline(x=2015, line_dash="dot", annotation_text="Curry MVP", annotation_position="top left", line=dict(color=LIGHTGREY))
            blocks.append(card(dcc.Graph(figure=fig)))

        if "fg3_pct" in leag.columns:
            fig = line_or_bar(leag, "fg3_pct", "League 3P% Over Time", "3P%")
            blocks.append(card(dcc.Graph(figure=fig)))

        if {"share_3pa", "share_2pa"}.issubset(leag.columns):
            mix = leag[["season_start_year", "share_3pa", "share_2pa"]].melt(id_vars="season_start_year", var_name="kind", value_name="share")
            mix["kind"] = mix["kind"].map({"share_3pa": "3PA Share", "share_2pa": "2PA Share"})
            fig = px.area(mix.sort_values("season_start_year"), x="season_start_year", y="share", color="kind", title="Shot Mix Over Time (Share of FGA)")
            style_fig_app(fig, x_title="Season Start Year", y_title="Share of FGA")
            blocks.append(card(dcc.Graph(figure=fig)))

        return blocks if blocks else card(html.Div("3PT fields not found.", style={"fontFamily": BODY_FONT, "color": TEXT_DARK}))

    if tab == "nba-context":
        if {"pts_home_team", "pts_away_team"}.issubset(leag.columns):
            s = leag.sort_values("season_start_year")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["season_start_year"], y=s["pts_home_team"], mode="lines+markers", name="Home PTS", line=dict(color=ORANGE, width=3), marker=dict(color=ORANGE)))
            fig.add_trace(go.Scatter(x=s["season_start_year"], y=s["pts_away_team"], mode="lines+markers", name="Away PTS", line=dict(color="#2aa3ff", width=2), marker=dict(color="#2aa3ff")))
            fig.update_layout(title="Home vs Away Scoring — League Averages (Per Team)")
            style_fig_app(fig, x_title="Season Start Year", y_title="PTS per Team")
            return card(dcc.Graph(figure=fig))
        return card(html.Div("Needs pts_home and pts_away.", style={"fontFamily": BODY_FONT, "color": TEXT_DARK}))

    if tab == "nba-outliers":
        if "pts_tot" not in df.columns:
            return card(html.Div("Outlier view needs pts_tot.", style={"fontFamily": BODY_FONT, "color": TEXT_DARK}))
        games = filter_years(df[["season_start_year", "era", "pts_tot"]].copy(), yr)
        thr = games["pts_tot"].quantile(0.99)
        base = px.scatter(games, x="season_start_year", y="pts_tot", color="era", opacity=0.25, title=f"All Games + Top 1% Scoring Games ≥ {thr:.0f} Points")
        style_fig_app(base, x_title="Season Start Year", y_title="Total Points (Two-Team)")
        base.update_traces(marker=dict(size=5))
        hi = games[games["pts_tot"] >= thr]
        base.add_trace(go.Scatter(x=hi["season_start_year"], y=hi["pts_tot"], mode="markers", name="Top 1%", marker=dict(size=9, color=ORANGE)))
        return card(dcc.Graph(figure=base))

    if tab == "nba-race":
        return card(
            html.Div(
                [
                    html.H4("Bar Chart Race — Top Teams Over Time", style={"fontFamily": "serif", "color": TEXT_DARK, "marginBottom": "8px"}),
                    html.P("Watch the top teams race over seasons based on average per-game stats.", style={"fontFamily": BODY_FONT, "color": LIGHTGREY, "fontSize": "13px"}),
                    html.Div(
                        [
                            html.Label("Statistic", style={"color": TEXT_DARK, "fontFamily": BODY_FONT, "marginRight": "8px"}),
                            dcc.Dropdown(
                                id="race-stat",
                                options=[
                                    {"label": "Points (PTS)", "value": "pts_tot"},
                                    {"label": "3PA (FG3A)", "value": "fg3a_tot"},
                                    {"label": "3PM (FG3M)", "value": "fg3m_tot"},
                                    {"label": "Assists (AST)", "value": "ast_tot"},
                                    {"label": "Rebounds (REB)", "value": "reb_tot"},
                                ],
                                value="pts_tot",
                                clearable=False,
                                style={"width": "260px", "color": "#111"},
                            ),
                        ],
                        style={"marginBottom": "10px"},
                    ),
                    dcc.Graph(id="race-graph"),
                    html.Small("Use the play button below the chart to animate seasons.", style={"fontFamily": BODY_FONT, "color": LIGHTGREY}),
                ]
            )
        )

    if tab == "nba-story":
        if team_season is None:
            return card(html.Div("Team-level aggregates not available in this dataset.",
                                 style={"fontFamily": BODY_FONT, "color": TEXT_DARK}))

        team_opts = [{"label": t, "value": t} for t in sorted(team_season["team"].unique())]

        return card(
            html.Div(
                [
                    html.H4("Team Evolution Story (Pick One Team)", style={"fontFamily": "serif", "color": TEXT_DARK}),
                    html.Div(
                        [
                            html.Label("Select team", style={"fontFamily": BODY_FONT, "color": TEXT_DARK}),
                            dcc.Dropdown(
                                id="story-team",
                                options=team_opts,
                                value=team_opts[0]["value"] if team_opts else None,
                                clearable=False,
                                style={"width": "360px", "color": "#111"},
                            ),
                        ],
                        style={"marginBottom": "10px"},
                    ),

                    # ✅ Download buttons row
                    html.Div(
                        [
                            dbc.Button("Download Story Figure (HTML)", id="btn-dl-story-html", color="primary",
                                       size="sm",
                                       style={"marginRight": "10px"}),
                            dbc.Button("Download Story Data (CSV)", id="btn-dl-story-csv", color="secondary",
                                       size="sm"),
                            dcc.Download(id="download-story-html"),
                            dcc.Download(id="download-story-csv"),
                        ],
                        style={"marginBottom": "10px"},
                    ),

                    dcc.Graph(id="story-fig"),
                    html.Div(id="story-notes"),
                ]
            )
        )
    return card(html.Div("Select an NBA sub-tab above.", style={"fontFamily": BODY_FONT, "color": TEXT_DARK}))


@my_app.callback(
    Output("download-story-html", "data"),
    Input("btn-dl-story-html", "n_clicks"),
    State("story-fig", "figure"),
    State("story-team", "value"),
    State("year-range", "value"),
    prevent_initial_call=True,
)
def download_story_html(n_clicks, fig, team, yr):
    if not fig or team is None or yr is None:
        return None

    lo, hi = yr
    filename = f"team_story_{team}_{lo}-{hi}.html"

    html_str = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
    return dict(content=html_str, filename=filename, type="text/html")

@my_app.callback(
    Output("download-story-csv", "data"),
    Input("btn-dl-story-csv", "n_clicks"),
    State("story-team", "value"),
    State("year-range", "value"),
    prevent_initial_call=True,
)
def download_story_csv(n_clicks, team, yr):
    if team is None or yr is None or team_season is None:
        return None

    lo, hi = yr
    s = team_season[team_season["team"] == team].copy()
    s = s[(s["season_start_year"] >= lo) & (s["season_start_year"] <= hi)].sort_values("season_start_year")

    # keep the key columns that appear in the story
    keep_cols = [c for c in ["team", "season_start_year", "pts_tot", "fg3a_tot", "fg3_pct", "share_3pa", "share_2pa"] if c in s.columns]
    out = s[keep_cols].copy()

    # enforce 2-decimal formatting in CSV (as strings)
    for c in out.columns:
        if c not in ["team", "season_start_year"] and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")

    filename = f"team_story_data_{team}_{lo}-{hi}.csv"
    return dcc.send_data_frame(out.to_csv, filename, index=False)


@my_app.callback(
    Output("nba-trends-fig", "figure"),
    Input("nba-trends-metric", "value"),
    Input("year-range", "value"),
    Input("team-select", "value"),
    Input("chart-type", "value"),
)
def update_nba_trends(metric, yr, team_val, chart_type):
    leag = filter_years(league_season, yr)
    if metric is None or metric not in leag.columns:
        fig = go.Figure()
        fig.add_annotation(text="Select a valid metric.", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_DARK))
        style_fig_app(fig)
        return fig

    title_map = {
        "pts_tot": "League Trend — Total Points (Two-Team)",
        "ast_tot": "League Trend — Assists (Two-Team)",
        "reb_tot": "League Trend — Rebounds (Two-Team)",
        "fg3a_tot": "League Trend — 3PA (Two-Team)",
        "fg3_pct": "League Trend — 3P%",
        "share_3pa": "League Trend — 3PA Share",
    }
    y_label_map = {
        "pts_tot": "Points",
        "ast_tot": "Assists",
        "reb_tot": "Rebounds",
        "fg3a_tot": "3PA",
        "fg3_pct": "3P%",
        "share_3pa": "3PA Share",
    }

    leag_sorted = leag.sort_values("season_start_year")

    if chart_type == "bar":
        fig = px.bar(leag_sorted, x="season_start_year", y=metric, title=title_map.get(metric, metric))
        style_fig_app(fig, x_title="Season Start Year", y_title=y_label_map.get(metric, metric))
        fig.update_traces(marker_color=ORANGE, name="League Average")
    else:
        fig = px.line(leag_sorted, x="season_start_year", y=metric, markers=True, title=title_map.get(metric, metric))
        style_fig_app(fig, x_title="Season Start Year", y_title=y_label_map.get(metric, metric))
        fig.update_traces(line=dict(color=ORANGE, width=3), marker=dict(color=ORANGE, size=6), name="League Average")

    team_list = (team_val or [])[:10]
    if team_season is not None and team_list:
        for t in team_list:
            tdf = filter_years(team_season[team_season["team"] == t], yr)
            if metric in tdf.columns and not tdf.empty:
                fig.add_trace(go.Scatter(x=tdf["season_start_year"], y=tdf[metric], mode="lines+markers", name=t, line=dict(width=2), marker=dict(size=5)))

    return fig

TEAM_COLOR_MAP = {}

def team_color(team: str) -> str:
    """
    Deterministic: each team always gets the same color across frames.
    Uses Plotly qualitative palette.
    """
    if team not in TEAM_COLOR_MAP:
        TEAM_COLOR_MAP[team] = BRIGHT_QUAL[len(TEAM_COLOR_MAP) % len(BRIGHT_QUAL)]
    return TEAM_COLOR_MAP[team]

@my_app.callback(
    Output("race-graph", "figure"),
    Input("race-stat", "value"),
    Input("year-range", "value"),
)
def update_bar_race(stat_col, yr_range):

    if team_season is None or stat_col not in team_season.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Team-season aggregates not available for this statistic.",
            x=0.5, y=0.5, showarrow=False
        )
        return style_fig_app(fig)

    # Use all teams in team_season (no hard-coded NBA list here)
    ts = team_season.copy()

    # Filter years only
    lo, hi = yr_range
    ts = ts[(ts["season_start_year"] >= lo) & (ts["season_start_year"] <= hi)].copy()

    ts = ts[["season_start_year", "team", stat_col]].copy()
    ts["team"] = ts["team"].astype(str).str.strip()
    ts[stat_col] = pd.to_numeric(ts[stat_col], errors="coerce")
    ts = ts.dropna(subset=["season_start_year", "team", stat_col])

    # Ensure 1 row per (season, team)
    ts = ts.groupby(["season_start_year", "team"], as_index=False)[stat_col].mean()

    if ts.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data in selected range.", x=0.5, y=0.5, showarrow=False)
        return style_fig_app(fig)

    seasons = sorted(ts["season_start_year"].unique())

    # Precompute top10 per season
    top_by_season = {}
    for s in seasons:
        g = ts[ts["season_start_year"] == s].sort_values(stat_col, ascending=False).head(10)
        top_by_season[s] = g

    # global x max for stable axis
    global_max = ts[stat_col].max()
    x_range = [0, float(global_max) * 1.15]

    def frame_trace(season):
        g = top_by_season[season].copy()

        # biggest on top
        g = g.sort_values(stat_col, ascending=True)

        colors = [team_color(t) for t in g["team"]]

        return go.Bar(
            x=g[stat_col],
            y=g["team"],
            orientation="h",
            text=[f"{v:.1f}" for v in g[stat_col]],
            textposition="outside",
            marker=dict(color=colors),  # ✅ one color per bar
        )

    # Initial figure = first season
    first = seasons[0]
    fig = go.Figure(data=[frame_trace(first)])

    # Frames
    frames = []
    for s in seasons:
        frames.append(go.Frame(data=[frame_trace(s)], name=str(s)))
    fig.frames = frames

    # Slider steps
    steps = []
    for s in seasons:
        steps.append({
            "label": str(s),
            "method": "animate",
            "args": [[str(s)], {"mode": "immediate",
                               "frame": {"duration": 450, "redraw": True},
                               "transition": {"duration": 150}}],
        })

    fig.update_layout(
        title=f"Bar Chart Race — Top 10 Teams by {stat_col.upper()}",
        xaxis=dict(range=x_range, title=stat_col.upper(), showgrid=True, gridcolor=GRID),
        yaxis=dict(title="Team", automargin=True),
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "x": 0.02,
            "y": -0.12,
            "buttons": [
                {"label": "Play",
                 "method": "animate",
                 "args": [None, {"fromcurrent": True,
                                "frame": {"duration": 450, "redraw": True},
                                "transition": {"duration": 150}}]},
                {"label": "Pause",
                 "method": "animate",
                 "args": [[None], {"mode": "immediate",
                                  "frame": {"duration": 0, "redraw": False},
                                  "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": 0,
            "y": -0.18,
            "x": 0.10,
            "len": 0.85,
            "steps": steps,
            "currentvalue": {"prefix": "season_start_year="},
        }],
        margin=dict(l=70, r=40, t=80, b=110),
    )

    return style_fig_app(fig, x_title=stat_col.upper(), y_title="Team")

@my_app.callback(
    Output("story-fig", "figure"),
    Output("story-notes", "children"),
    Input("story-team", "value"),
    Input("year-range", "value"),
)
def update_team_story(team, yr):
    fig = go.Figure()

    if team is None or team_season is None:
        fig.add_annotation(text="Select a team.", x=0.5, y=0.5, showarrow=False)
        fig = style_fig_app(fig)
        return fig, ""

    s = team_season[team_season["team"] == team].copy()
    s = filter_years(s, yr).sort_values("season_start_year")

    required = ["pts_tot", "fg3a_tot", "fg3_pct", "share_3pa", "share_2pa"]
    if s.empty or not all(c in s.columns for c in required):
        fig.add_annotation(text="Not enough data for this team in the selected range.", x=0.5, y=0.5, showarrow=False)
        fig = style_fig_app(fig)
        return fig, ""

    s["fg3_pct_roll"] = s["fg3_pct"].rolling(window=5, min_periods=1).mean()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"{team}: PTS per Game — Line",
            f"{team}: 3PA per Game — Bars",
            f"{team}: 3P% — Scatter + 5Y Avg",
            f"{team}: Shot Mix — Stacked Area",
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=s["season_start_year"], y=s["pts_tot"],
            mode="lines+markers",
            line=dict(width=3, color=ORANGE),
            marker=dict(size=5, color=ORANGE),
            name="PTS"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=s["season_start_year"], y=s["fg3a_tot"],
            marker=dict(color=ORANGE),
            opacity=0.9,
            name="3PA"
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=s["season_start_year"], y=s["fg3_pct"],
            mode="markers",
            marker=dict(size=6, opacity=0.6, color=ORANGE),
            name="3P% (year)"
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=s["season_start_year"], y=s["fg3_pct_roll"],
            mode="lines",
            line=dict(width=3),
            name="3P% (5Y avg)"
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=s["season_start_year"], y=s["share_2pa"],
            mode="lines",
            fill="tozeroy",
            line=dict(width=0.5),
            name="2PA Share",
            opacity=0.7
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=s["season_start_year"], y=s["share_3pa"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0.5),
            name="3PA Share",
            opacity=0.9
        ),
        row=2, col=2
    )

    fig.update_layout(title=f"Team Evolution Story — {team}", height=820, showlegend=True)
    fig = style_fig_app(fig)
    fig.update_xaxes(title_text="Season Start Year", tickformat=".0f", showgrid=True)
    fig.update_yaxes(showgrid=True)

    fig.update_yaxes(title_text="PTS", row=1, col=1)
    fig.update_yaxes(title_text="3PA", row=1, col=2)
    fig.update_yaxes(title_text="3P%", row=2, col=1, tickformat=".3f")
    fig.update_yaxes(title_text="Share of FGA", row=2, col=2, tickformat=".2f", range=[0, 1])

    first = s.head(5)
    last = s.tail(5)

    def mean_safe(x):
        return float(np.nanmean(x)) if len(x) else float("nan")

    pts_up = mean_safe(last["pts_tot"]) - mean_safe(first["pts_tot"])
    a3_up = mean_safe(last["fg3a_tot"]) - mean_safe(first["fg3a_tot"])
    p3_up = mean_safe(last["fg3_pct"]) - mean_safe(first["fg3_pct"])
    sh_up = mean_safe(last["share_3pa"]) - mean_safe(first["share_3pa"])

    notes = html.Ul(
        [
            html.Li(f"Scoring changed by about {pts_up:+.1f} PTS (last 5 seasons vs first 5 in your range)."),
            html.Li(f"3PA changed by about {a3_up:+.1f} attempts per game — shows how much the team leaned into threes."),
            html.Li(f"3P% changed by about {p3_up:+.3f} — efficiency trend (noisy year-to-year, smoother in 5Y line)."),
            html.Li(f"3PA share changed by about {sh_up:+.2f} (share of total shots) — confirms shot-mix shift."),
        ],
        style={"fontFamily": "serif", "color": TEXT_DARK, "lineHeight": "1.6"},
    )

    return fig, card(html.Div([html.H4("Auto Observations", style={"fontFamily": "serif", "color": TITLE_BLUE}), notes]))



if __name__ == "__main__":
    my_app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
