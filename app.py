import os
import time

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.feature_engineering import create_features


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GameSense AI",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

DATA_PATH = "data/raw/player-churn.csv"
MODEL_PATH = "models/gamesense_best_model.joblib"
RESULTS_PATH = "models/model_comparison.csv"


# ============================================================
# NEON GAMING CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL BACKGROUND
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(0, 229, 255, 0.08),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(138, 43, 226, 0.10),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(255, 0, 255, 0.05),
                transparent 35%
            ),
            #050509;
    }


    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background: #07070D;
        border-right: 1px solid rgba(0, 229, 255, 0.25);
    }


    /* =====================================================
       MAIN TITLE
       ===================================================== */

    .main-title {
        text-align: center;

        font-size: 4rem;

        font-weight: 900;

        letter-spacing: 6px;

        background: linear-gradient(
            90deg,
            #00E5FF,
            #8A2BE2,
            #FF00FF,
            #00E5FF
        );

        background-size: 300% 300%;

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        animation: neonGradient 5s ease infinite;
    }


    @keyframes neonGradient {

        0% {
            background-position: 0% 50%;
        }

        50% {
            background-position: 100% 50%;
        }

        100% {
            background-position: 0% 50%;
        }

    }


    /* =====================================================
       SUBTITLE
       ===================================================== */

    .subtitle {

        text-align: center;

        color: #8E97AE;

        letter-spacing: 4px;

        font-size: 0.9rem;

        margin-bottom: 20px;

    }


    /* =====================================================
       SECTION HEADERS
       ===================================================== */

    .section-header {

        color: #00E5FF;

        font-size: 1.45rem;

        font-weight: 800;

        letter-spacing: 2px;

        text-shadow:
            0 0 8px rgba(0, 229, 255, 0.75);

    }


    /* =====================================================
       NEON LINE
       ===================================================== */

    .neon-line {

        height: 1px;

        background: linear-gradient(
            90deg,
            transparent,
            #00E5FF,
            #8A2BE2,
            #FF00FF,
            transparent
        );

        box-shadow:
            0 0 10px rgba(0, 229, 255, 0.6);

        margin: 20px 0;

    }


    /* =====================================================
       KPI CARDS
       ===================================================== */

    div[data-testid="stMetric"] {

        background:
            linear-gradient(
                145deg,
                rgba(15, 17, 31, 0.96),
                rgba(7, 8, 16, 0.98)
            );

        border: 1px solid rgba(0, 229, 255, 0.35);

        border-radius: 18px;

        padding: 20px;

        min-height: 110px;

        box-shadow:
            0 0 12px rgba(0, 229, 255, 0.08),
            inset 0 0 20px rgba(0, 229, 255, 0.025);

        transition: all 0.25s ease;

    }


    div[data-testid="stMetric"]:hover {

        transform: translateY(-4px);

        border-color: rgba(0, 229, 255, 0.75);

        box-shadow:
            0 0 18px rgba(0, 229, 255, 0.30),
            0 0 35px rgba(138, 43, 226, 0.15);

    }


    div[data-testid="stMetricLabel"] {

        color: #7D879E !important;

        text-transform: uppercase;

        letter-spacing: 2px;

        font-size: 0.75rem !important;

    }


    div[data-testid="stMetricValue"] {

        color: #00E5FF !important;

        font-size: 2.3rem !important;

        font-weight: 900 !important;

        text-shadow:
            0 0 8px rgba(0, 229, 255, 0.7);

    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {

        width: 100%;

        background:
            linear-gradient(
                90deg,
                #00E5FF,
                #8A2BE2
            );

        color: white;

        border: none;

        border-radius: 12px;

        min-height: 50px;

        font-size: 1rem;

        font-weight: 900;

        letter-spacing: 1px;

        box-shadow:
            0 0 12px rgba(0, 229, 255, 0.3);

        transition: all 0.2s ease;

    }


    .stButton > button:hover {

        transform: scale(1.02);

        box-shadow:
            0 0 20px rgba(0, 229, 255, 0.6),
            0 0 35px rgba(138, 43, 226, 0.4);

    }


    /* =====================================================
       INPUTS
       ===================================================== */

    input {

        background-color: #0C0D16 !important;

        color: white !important;

    }


    div[data-baseweb="select"] > div {

        background-color: #0C0D16 !important;

        color: white !important;

    }


    /* =====================================================
       PROGRESS BAR
       ===================================================== */

    div[data-testid="stProgress"] > div > div {

        background:
            linear-gradient(
                90deg,
                #00E5FF,
                #8A2BE2,
                #FF00FF
            );

    }


    /* =====================================================
       DATAFRAME
       ===================================================== */

    div[data-testid="stDataFrame"] {

        border: 1px solid rgba(0, 229, 255, 0.25);

        border-radius: 12px;

    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer-text {

        text-align: center;

        color: #555D70;

        letter-spacing: 2px;

        margin-top: 50px;

        font-size: 0.8rem;

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        return None

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

@st.cache_data
def load_results():

    if not os.path.exists(RESULTS_PATH):

        return None

    return pd.read_csv(
        RESULTS_PATH,
        index_col=0
    )


# ============================================================
# INITIALIZE APPLICATION
# ============================================================

df = load_data()

model = load_model()

results = load_results()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="main-title">⚡ GAMESENSE AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">PLAYER INTELLIGENCE • CHURN PREDICTION • MACHINE LEARNING</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="neon-line"></div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎮 GameSense AI")

    st.caption(
        "PLAYER INTELLIGENCE SYSTEM"
    )

    st.divider()

    st.markdown(
        "### SYSTEM MODULES"
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔮 Churn Predictor",
            "🤖 Model Performance",
            "📈 Player Analytics"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        "### SYSTEM STATUS"
    )

    if model is not None:

        st.success(
            "● ML MODEL ONLINE"
        )

    else:

        st.error(
            "● ML MODEL OFFLINE"
        )

    st.caption(
        "GameSense AI v1.0"
    )


# ============================================================
# ============================================================
# DASHBOARD
# ============================================================
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<p class="section-header">⚡ GAME OVERVIEW</p>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total_players = len(df)

    churned_players = int(
        df["player_churn"].sum()
    )

    active_players = (
        total_players -
        churned_players
    )

    churn_rate = (
        churned_players /
        total_players
    ) * 100


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            label="TOTAL PLAYERS",
            value=f"{total_players:,}"
        )


    with col2:

        st.metric(
            label="ACTIVE PLAYERS",
            value=f"{active_players:,}"
        )


    with col3:

        st.metric(
            label="CHURNED PLAYERS",
            value=f"{churned_players:,}"
        )


    with col4:

        st.metric(
            label="CHURN RATE",
            value=f"{churn_rate:.1f}%"
        )


    st.markdown(
        '<div class="neon-line"></div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------------

    with col1:

        st.markdown(
            '<p class="section-header">🎯 CHURN DISTRIBUTION</p>',
            unsafe_allow_html=True
        )


        churn_data = pd.DataFrame(
            {
                "Status": [
                    "Active",
                    "Churned"
                ],

                "Players": [
                    active_players,
                    churned_players
                ]
            }
        )


        fig = px.pie(
            churn_data,
            names="Status",
            values="Players",
            hole=0.68
        )


        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            legend=dict(
                font=dict(
                    color="white"
                )
            ),
            margin=dict(
                t=20,
                b=20,
                l=20,
                r=20
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # PLAYER LIFETIME
    # --------------------------------------------------------

    with col2:

        st.markdown(
            '<p class="section-header">⏱️ PLAYER LIFETIME</p>',
            unsafe_allow_html=True
        )


        fig = px.histogram(
            df,
            x="player_lifetime",
            nbins=35
        )


        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis_title="Player Lifetime",
            yaxis_title="Players"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # QUICK INSIGHTS
    # --------------------------------------------------------

    st.markdown(
        '<p class="section-header">🧠 QUICK INSIGHTS</p>',
        unsafe_allow_html=True
    )


    insight1, insight2, insight3 = st.columns(3)


    with insight1:

        st.info(
            f"🎮 **{total_players:,}** players "
            "are represented in the dataset."
        )


    with insight2:

        st.warning(
            f"⚠️ **{churn_rate:.1f}%** "
            "of players are classified as churned."
        )


    with insight3:

        st.success(
            f"✓ **{active_players:,}** "
            "players remain active."
        )


# ============================================================
# ============================================================
# CHURN PREDICTOR
# ============================================================
# ============================================================

elif page == "🔮 Churn Predictor":

    st.markdown(
        '<p class="section-header">🔮 PLAYER CHURN PREDICTOR</p>',
        unsafe_allow_html=True
    )


    st.write(
        "Enter the player's basic behavioral information "
        "and let GameSense AI estimate the churn probability."
    )


    st.markdown(
        '<div class="neon-line"></div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PLAYER INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        player_lifetime = st.number_input(
            "Player Lifetime",
            min_value=0.0,
            max_value=float(
                df["player_lifetime"].max()
            ),
            value=float(
                df["player_lifetime"].median()
            ),
            step=1000.0
        )


        session_count = st.number_input(
            "Total Session Count",
            min_value=1,
            max_value=int(
                df["session_count"].max()
            ),
            value=int(
                df["session_count"].median()
            ),
            step=1
        )


    with col2:

        cohort_id = st.number_input(
            "Cohort ID",
            min_value=0,
            value=1,
            step=1
        )


        cohort_day = st.number_input(
            "Cohort Day of Week",
            min_value=0,
            max_value=6,
            value=2,
            step=1
        )


    st.write("")


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if st.button(
        "⚡ ANALYZE PLAYER"
    ):


        if model is None:

            st.error(
                "Trained model not found. "
                "Run python src/train.py first."
            )


        else:

            # ------------------------------------------------
            # SCANNING ANIMATION
            # ------------------------------------------------

            progress = st.progress(
                0
            )

            status = st.empty()


            steps = [
                "Scanning player telemetry...",
                "Analyzing session behavior...",
                "Analyzing player lifetime...",
                "Calculating engagement signals...",
                "Generating engineered features...",
                "Running Gradient Boosting model...",
                "Generating risk assessment..."
            ]


            for i, message in enumerate(
                steps
            ):

                status.info(
                    f"⚡ {message}"
                )


                progress.progress(
                    int(
                        (
                            (i + 1) /
                            len(steps)
                        ) * 100
                    )
                )


                time.sleep(
                    0.25
                )


            status.empty()

            progress.empty()


            # ------------------------------------------------
            # BUILD PLAYER INPUT
            # ------------------------------------------------
            #
            # Start with an existing row so that all raw
            # columns required by the trained pipeline exist.
            #
            # Then replace the features controlled by the user.
            # ------------------------------------------------

            player_row = df.drop(
                columns=[
                    "player_churn"
                ],
                errors="ignore"
            ).iloc[
                [0]
            ].copy()


            # ------------------------------------------------
            # UPDATE USER INPUTS
            # ------------------------------------------------

            if "cohort_id" in player_row.columns:

                player_row[
                    "cohort_id"
                ] = cohort_id


            if "cohort_day_of_week" in player_row.columns:

                player_row[
                    "cohort_day_of_week"
                ] = cohort_day


            if "player_lifetime" in player_row.columns:

                player_row[
                    "player_lifetime"
                ] = player_lifetime


            if "session_count" in player_row.columns:

                player_row[
                    "session_count"
                ] = session_count


            # ------------------------------------------------
            # APPLY THE EXACT SAME FEATURE ENGINEERING
            # USED DURING MODEL TRAINING
            # ------------------------------------------------

            player_features = create_features(
                player_row
            )


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            try:

                probability = model.predict_proba(
                    player_features
                )[0][1]


                # ------------------------------------------------
                # RISK LEVEL
                # ------------------------------------------------

                if probability >= 0.75:

                    risk = "HIGH"

                elif probability >= 0.50:

                    risk = "MEDIUM"

                else:

                    risk = "LOW"


                # ------------------------------------------------
                # RESULT HEADER
                # ------------------------------------------------

                st.markdown(
                    '<div class="neon-line"></div>',
                    unsafe_allow_html=True
                )


                st.markdown(
                    '<p class="section-header">🎯 PREDICTION RESULT</p>',
                    unsafe_allow_html=True
                )


                # ------------------------------------------------
                # RESULT CARDS
                # ------------------------------------------------

                result_col1, result_col2, result_col3 = st.columns(3)


                with result_col1:

                    st.metric(
                        "CHURN PROBABILITY",
                        f"{probability * 100:.1f}%"
                    )


                with result_col2:

                    st.metric(
                        "RISK LEVEL",
                        risk
                    )


                with result_col3:

                    prediction = (
                        "Likely to Churn"
                        if probability >= 0.50
                        else "Likely to Stay"
                    )


                    st.metric(
                        "PREDICTION",
                        prediction
                    )


                st.write("")


                # ------------------------------------------------
                # RISK BAR
                # ------------------------------------------------

                st.markdown(
                    "### CHURN RISK LEVEL"
                )


                st.progress(
                    float(probability)
                )


                if risk == "HIGH":

                    st.error(
                        "🔴 HIGH CHURN RISK\n\n"
                        "This player may require a retention strategy."
                    )


                elif risk == "MEDIUM":

                    st.warning(
                        "🟡 MEDIUM CHURN RISK\n\n"
                        "Consider monitoring this player's engagement."
                    )


                else:

                    st.success(
                        "🟢 LOW CHURN RISK\n\n"
                        "This player appears relatively engaged."
                    )


                # ------------------------------------------------
                # PLAYER PROFILE
                # ------------------------------------------------

                st.markdown(
                    '<p class="section-header">📋 PLAYER PROFILE</p>',
                    unsafe_allow_html=True
                )


                p1, p2, p3 = st.columns(3)


                with p1:

                    st.metric(
                        "PLAYER LIFETIME",
                        f"{player_lifetime:,.0f}"
                    )


                with p2:

                    st.metric(
                        "SESSION COUNT",
                        session_count
                    )


                with p3:

                    st.metric(
                        "COHORT",
                        cohort_id
                    )


                # ------------------------------------------------
                # ENGINEERED FEATURES
                # ------------------------------------------------

                st.markdown(
                    '<p class="section-header">🧠 ENGINEERED FEATURES</p>',
                    unsafe_allow_html=True
                )


                f1, f2, f3 = st.columns(3)


                with f1:

                    if "total_session_activity" in player_features.columns:

                        st.metric(
                            "TOTAL SESSION ACTIVITY",
                            f"{player_features['total_session_activity'].iloc[0]:,.0f}"
                        )


                with f2:

                    if "total_stage_activity" in player_features.columns:

                        st.metric(
                            "TOTAL STAGE ACTIVITY",
                            f"{player_features['total_stage_activity'].iloc[0]:,.0f}"
                        )


                with f3:

                    if "sessions_per_lifetime" in player_features.columns:

                        st.metric(
                            "SESSIONS / LIFETIME",
                            f"{player_features['sessions_per_lifetime'].iloc[0]:.6f}"
                        )


            except Exception as error:

                st.error(
                    "Prediction could not be generated."
                )

                st.exception(
                    error
                )


# ============================================================
# ============================================================
# MODEL PERFORMANCE
# ============================================================
# ============================================================

elif page == "🤖 Model Performance":

    st.markdown(
        '<p class="section-header">🤖 MODEL PERFORMANCE</p>',
        unsafe_allow_html=True
    )


    if results is None:

        st.warning(
            "Model comparison results were not found. "
            "Run python src/train.py first."
        )


    else:

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        best_model = results[
            "f1"
        ].idxmax()


        best_f1 = results.loc[
            best_model,
            "f1"
        ]


        st.success(
            f"🏆 BEST MODEL: {best_model}  |  "
            f"F1 SCORE: {best_f1:.4f}"
        )


        st.markdown(
            '<div class="neon-line"></div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # METRIC SELECTOR
        # ----------------------------------------------------

        metric = st.selectbox(
            "SELECT EVALUATION METRIC",
            [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc"
            ]
        )


        # ----------------------------------------------------
        # PREPARE CHART DATA
        # ----------------------------------------------------

        chart_data = (
            results
            .reset_index()
        )


        chart_data = chart_data.rename(
            columns={
                "index": "Model"
            }
        )


        # ----------------------------------------------------
        # MODEL COMPARISON CHART
        # ----------------------------------------------------

        fig = px.bar(
            chart_data,
            x="Model",
            y=metric,
            text_auto=".2%",
            title=f"{metric.upper()} COMPARISON"
        )


        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            yaxis=dict(
                range=[
                    0,
                    1.05
                ]
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # DETAILED RESULTS
        # ----------------------------------------------------

        st.markdown(
            '<p class="section-header">📊 DETAILED RESULTS</p>',
            unsafe_allow_html=True
        )


        st.dataframe(
            results.style.format(
                "{:.4f}"
            ),
            use_container_width=True
        )


        # ----------------------------------------------------
        # MODEL SCOREBOARD
        # ----------------------------------------------------

        st.markdown(
            '<p class="section-header">⚡ MODEL SCOREBOARD</p>',
            unsafe_allow_html=True
        )


        for model_name, row in results.iterrows():

            st.write(
                f"**{model_name}**"
            )


            st.progress(
                float(
                    row["f1"]
                )
            )


            st.caption(
                f"F1: {row['f1']:.4f}  |  "
                f"Accuracy: {row['accuracy']:.4f}  |  "
                f"Precision: {row['precision']:.4f}  |  "
                f"Recall: {row['recall']:.4f}  |  "
                f"ROC-AUC: {row['roc_auc']:.4f}"
            )


# ============================================================
# ============================================================
# PLAYER ANALYTICS
# ============================================================
# ============================================================

elif page == "📈 Player Analytics":

    st.markdown(
        '<p class="section-header">📈 PLAYER ANALYTICS</p>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # SESSION DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            '<p class="section-header">🎮 SESSION DISTRIBUTION</p>',
            unsafe_allow_html=True
        )


        fig = px.histogram(
            df,
            x="session_count",
            color="player_churn",
            nbins=15,
            title="Session Count Distribution"
        )


        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # PLAYER LIFETIME VS SESSION COUNT
    # --------------------------------------------------------

    with col2:

        st.markdown(
            '<p class="section-header">⏱️ ENGAGEMENT ANALYSIS</p>',
            unsafe_allow_html=True
        )


        fig = px.scatter(
            df,
            x="player_lifetime",
            y="session_count",
            color="player_churn",
            title="Player Lifetime vs Session Count",
            opacity=0.7
        )


        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # PLAYER TYPE ANALYSIS
    # --------------------------------------------------------

    if "player_type" in df.columns:

        st.markdown(
            '<p class="section-header">🎮 PLAYER TYPE ANALYSIS</p>',
            unsafe_allow_html=True
        )


        type_data = pd.crosstab(
            df["player_type"],
            df["player_churn"],
            normalize="index"
        ) * 100


        type_data = (
            type_data
            .reset_index()
        )


        if len(type_data.columns) == 3:

            type_data.columns = [
                "Player Type",
                "Not Churned",
                "Churned"
            ]


        fig = px.bar(
            type_data,
            x="Player Type",
            y=[
                "Not Churned",
                "Churned"
            ],
            barmode="group",
            title="Churn Rate by Player Type"
        )


        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # DATASET EXPLORER
    # --------------------------------------------------------

    with st.expander(
        "🔍 VIEW DATASET"
    ):

        st.dataframe(
            df.head(100),
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="neon-line"></div>',
    unsafe_allow_html=True
)


st.caption(
    "⚡ GAMESENSE AI • PLAYER INTELLIGENCE SYSTEM • "
    "PYTHON • SCIKIT-LEARN • STREAMLIT • PLOTLY"
)