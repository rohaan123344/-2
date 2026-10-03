# =========================================================
# AICADEMY 360
# MACHINE LEARNING ASSIGNMENT #2
# DARAZ STYLE DAILY ORDERS PREDICTOR
# =========================================================


# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================

import os

import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# 2. PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Aicademy 360 | Daily Orders Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 3. CUSTOM DESIGN
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       MAIN PAGE
       ===================================================== */

    .stApp {
        background: linear-gradient(
            135deg,
            #f7fbff 0%,
            #eef7ff 45%,
            #f7f2ff 100%
        );
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }


    /* =====================================================
       MAIN HEADINGS
       ===================================================== */

    h1 {
        color: #075985 !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #0c4a6e !important;
        font-weight: 800 !important;
    }

    h3 {
        color: #164e63 !important;
        font-weight: 750 !important;
    }


    /* =====================================================
       NORMAL TEXT
       ===================================================== */

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #172033 !important;
    }


    /* =====================================================
       CAPTIONS
       ===================================================== */

    [data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }


    /* =====================================================
       METRIC CARDS
       ===================================================== */

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    }

    div[data-testid="stMetric"] label {
        color: #64748b !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #075985 !important;
        font-weight: 800 !important;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #071a35 0%,
            #0b2d59 55%,
            #123b66 100%
        );
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: white !important;
    }


    /* =====================================================
       TABS
       ===================================================== */

    button[data-baseweb="tab"] {
        color: #334155 !important;
        font-size: 15px;
        font-weight: 700;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #075985 !important;
    }


    /* =====================================================
       FORM / INPUT LABELS
       ===================================================== */

    label {
        color: #172033 !important;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        background: linear-gradient(
            90deg,
            #0284c7,
            #4f46e5
        );

        color: white !important;
        border: none;
        border-radius: 14px;
        min-height: 48px;
        font-weight: 800;
    }

    button[kind="primaryFormSubmit"] {
        background: linear-gradient(
            90deg,
            #059669,
            #0284c7
        ) !important;

        color: white !important;
        border-radius: 14px !important;
        font-weight: 800 !important;
    }


    /* =====================================================
       DOWNLOAD BUTTON
       ===================================================== */

    [data-testid="stDownloadButton"] button {
        background: linear-gradient(
            90deg,
            #0284c7,
            #4f46e5
        ) !important;

        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 800 !important;
    }


    /* =====================================================
       EXPANDERS
       ===================================================== */

    [data-testid="stExpander"] summary {
        color: #172033 !important;
        font-weight: 700;
    }


    /* =====================================================
       MOBILE
       ===================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        h1 {
            font-size: 31px !important;
        }

        h2 {
            font-size: 25px !important;
        }

        h3 {
            font-size: 20px !important;
        }

        button[data-baseweb="tab"] {
            font-size: 12px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. LOAD DATASET
# =========================================================

DATA_FILE = "daraz_daily_orders_synthetic.csv"


if not os.path.exists(DATA_FILE):

    st.error(
        "❌ Dataset not found."
    )

    st.write(
        f"Please make sure `{DATA_FILE}` is in the same "
        "folder as `app.py`."
    )

    st.stop()


df = pd.read_csv(DATA_FILE)


# =========================================================
# 5. CHECK REQUIRED COLUMNS
# =========================================================

feature_columns = [
    "Website_Visitors",
    "Ad_Spend_PKR",
    "Discount_Percent",
    "Weekend",
]

target_column = "Daily_Orders"


required_columns = feature_columns + [target_column]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "❌ Some required columns are missing:"
    )

    st.write(
        ", ".join(missing_columns)
    )

    st.stop()


# =========================================================
# 6. REMOVE MISSING VALUES
# =========================================================

df = df.dropna(
    subset=required_columns
).copy()


# =========================================================
# 7. CHOOSE X AND Y
# =========================================================

X = df[feature_columns]

y = df[target_column]


# =========================================================
# 8. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# =========================================================
# 9. TRAIN LINEAR REGRESSION
# =========================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train,
)


# =========================================================
# 10. TEST MODEL
# =========================================================

test_predictions = model.predict(
    X_test
)


# =========================================================
# 11. MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(
    y_test,
    test_predictions
)


r2 = r2_score(
    y_test,
    test_predictions
)


# =========================================================
# 12. TEST RESULTS
# =========================================================

test_results = X_test.copy()

test_results["Actual_Orders"] = y_test.values

test_results["Predicted_Orders"] = (
    test_predictions.round(1)
)

test_results["Residual"] = (
    test_results["Actual_Orders"]
    - test_results["Predicted_Orders"]
).round(1)


# =========================================================
# 13. COEFFICIENT TABLE
# =========================================================

coefficient_table = pd.DataFrame(
    {
        "Feature": feature_columns,
        "Coefficient": model.coef_.round(4),
    }
)


# =========================================================
# 14. MODEL FORMULA
# =========================================================

formula = (
    f"Daily Orders = {model.intercept_:.2f}"
)

for feature, coefficient in zip(
    feature_columns,
    model.coef_
):

    if coefficient >= 0:

        formula += (
            f" + {coefficient:.4f} × {feature}"
        )

    else:

        formula += (
            f" - {abs(coefficient):.4f} × {feature}"
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title(
        "📦 Daily Orders"
    )

    st.caption(
        "Aicademy 360"
    )

    st.divider()

    st.subheader(
        "🚀 Project Information"
    )

    st.write(
        "🤖 Model: Linear Regression"
    )

    st.write(
        "📚 Type: Supervised Regression"
    )

    st.write(
        "🎯 Target: Daily Orders"
    )

    st.write(
        "📊 Features: 4"
    )

    st.write(
        f"📉 MAE: {mae:.2f}"
    )

    st.write(
        f"🎯 R²: {r2:.3f}"
    )

    st.divider()

    st.subheader(
        "🎓 Assignment"
    )

    st.info(
        "Machine Learning Assignment #2"
    )

    st.divider()

    st.subheader(
        "🛠️ Technologies"
    )

    st.write("🐍 Python")
    st.write("🐼 Pandas")
    st.write("📐 Linear Regression")
    st.write("📊 Scikit-Learn")
    st.write("🎈 Streamlit")

    st.divider()

    st.caption(
        "Learn • Practice • Build"
    )


# =========================================================
# MAIN TITLE
# =========================================================

st.title(
    "📦 Daraz Style Daily Orders Predictor"
)

st.subheader(
    "Predict Daily Orders Using Linear Regression"
)

st.write(
    "Use website visitors, advertising spend, discount percentage, "
    "and weekend information to estimate the number of daily orders."
)

st.caption(
    f"Aicademy 360 • Machine Learning Assignment #2 • "
    f"{len(df)} dataset records"
)


# =========================================================
# INTRODUCTION
# =========================================================

st.info(
    "💡 Regression predicts a number. In this project, "
    "our target is Daily_Orders."
)


# =========================================================
# PROJECT SUMMARY
# =========================================================

metric_1, metric_2, metric_3, metric_4 = st.columns(4)


with metric_1:

    st.metric(
        "📚 Data Rows",
        len(df)
    )


with metric_2:

    st.metric(
        "🏋️ Training Rows",
        len(X_train)
    )


with metric_3:

    st.metric(
        "📉 Test MAE",
        f"{mae:.2f}"
    )


with metric_4:

    st.metric(
        "🎯 Test R²",
        f"{r2:.3f}"
    )


st.divider()


# =========================================================
# TABS
# =========================================================

predict_tab, insights_tab, data_tab = st.tabs(
    [
        "📦 Predict Orders",
        "📊 Model Insights",
        "🧾 Explore Data",
    ]
)


# =========================================================
# TAB 1 — PREDICT ORDERS
# =========================================================

with predict_tab:

    st.header(
        "📦 Create a New Business Scenario"
    )

    st.write(
        "Choose the conditions for a new business day "
        "and let the trained model estimate daily orders."
    )

    st.info(
        "🎯 For this classroom project, keeping your inputs "
        "inside the dataset's range gives a more reasonable estimate."
    )


    # =====================================================
    # PREDICTION FORM
    # =====================================================

    with st.form(
        "prediction_form"
    ):

        st.subheader(
            "🌐 Website & Advertising"
        )

        left_col, right_col = st.columns(2)


        # -------------------------------------------------
        # WEBSITE VISITORS
        # -------------------------------------------------

        with left_col:

            website_visitors = st.slider(
                "🌐 Website Visitors",

                min_value=int(
                    df["Website_Visitors"].min()
                ),

                max_value=int(
                    df["Website_Visitors"].max()
                ),

                value=int(
                    df["Website_Visitors"].median()
                ),

                step=50,

                help=(
                    "Number of people who visited "
                    "the website."
                ),
            )


        # -------------------------------------------------
        # AD SPEND
        # -------------------------------------------------

        with right_col:

            ad_spend = st.slider(
                "💰 Ad Spend (PKR)",

                min_value=int(
                    df["Ad_Spend_PKR"].min()
                ),

                max_value=int(
                    df["Ad_Spend_PKR"].max()
                ),

                value=int(
                    df["Ad_Spend_PKR"].median()
                ),

                step=500,

                help=(
                    "Amount spent on advertising."
                ),
            )


        st.subheader(
            "🏷️ Discount & Day Type"
        )


        left_col_2, right_col_2 = st.columns(2)


        # -------------------------------------------------
        # DISCOUNT
        # -------------------------------------------------

        with left_col_2:

            discount_percent = st.slider(
                "🏷️ Discount (%)",

                min_value=int(
                    df["Discount_Percent"].min()
                ),

                max_value=int(
                    df["Discount_Percent"].max()
                ),

                value=int(
                    df["Discount_Percent"].median()
                ),

                step=1,

                help=(
                    "Percentage discount offered."
                ),
            )


        # -------------------------------------------------
        # DAY TYPE
        # -------------------------------------------------

        with right_col_2:

            day_type = st.radio(
                "📅 Day Type",

                options=[
                    "Weekday",
                    "Weekend",
                ],

                horizontal=True,
            )


        st.divider()


        # -------------------------------------------------
        # PREDICTION BUTTON
        # -------------------------------------------------

        predict_button = st.form_submit_button(
            "🚀 Predict Daily Orders",

            type="primary",

            use_container_width=True,
        )


    # =====================================================
    # PREDICT
    # =====================================================

    if predict_button:

        weekend = (
            1
            if day_type == "Weekend"
            else 0
        )


        new_day = pd.DataFrame(
            {
                "Website_Visitors": [
                    website_visitors
                ],

                "Ad_Spend_PKR": [
                    ad_spend
                ],

                "Discount_Percent": [
                    discount_percent
                ],

                "Weekend": [
                    weekend
                ],
            }
        )


        predicted_orders = model.predict(
            new_day
        )[0]


        predicted_orders = max(
            0,
            predicted_orders
        )


        st.divider()


        st.success(
            "✅ Prediction Completed Successfully!"
        )


        result_1, result_2, result_3 = st.columns(3)


        with result_1:

            st.metric(
                "📦 Estimated Daily Orders",
                f"{round(predicted_orders):,}"
            )


        with result_2:

            st.metric(
                "🌐 Website Visitors",
                f"{website_visitors:,}"
            )


        with result_3:

            st.metric(
                "💰 Ad Spend",
                f"PKR {ad_spend:,}"
            )


        st.subheader(
            "📋 Business Scenario"
        )


        st.write(
            f"🌐 **Website Visitors:** {website_visitors:,}"
        )

        st.write(
            f"💰 **Advertising Spend:** PKR {ad_spend:,}"
        )

        st.write(
            f"🏷️ **Discount:** {discount_percent}%"
        )

        st.write(
            f"📅 **Day Type:** {day_type}"
        )

        st.write(
            f"📦 **Estimated Orders:** "
            f"{round(predicted_orders):,}"
        )


        st.caption(
            f"Raw model estimate: "
            f"{predicted_orders:.2f} orders"
        )


        with st.expander(
            "🔎 See the exact input sent to the model"
        ):

            st.dataframe(
                new_day,

                use_container_width=True,

                hide_index=True,
            )


        st.warning(
            "⚠️ This is a model estimate based on "
            "synthetic classroom data. It is not a "
            "guaranteed future sales number."
        )


# =========================================================
# TAB 2 — MODEL INSIGHTS
# =========================================================

with insights_tab:

    st.header(
        "📊 Model Insights"
    )

    st.write(
        "Here we evaluate the Linear Regression model "
        "using the unseen test data."
    )


    # =====================================================
    # PERFORMANCE
    # =====================================================

    st.subheader(
        "🎯 Model Performance"
    )


    score_1, score_2 = st.columns(2)


    with score_1:

        st.metric(
            "📉 Mean Absolute Error",
            f"{mae:.2f} orders"
        )


    with score_2:

        st.metric(
            "🎯 R² Score",
            f"{r2:.3f}"
        )


    st.write(
        f"**MAE:** The model's predictions are about "
        f"**{mae:.2f} orders** away from the actual "
        f"value on average."
    )


    st.write(
        f"**R²:** The model has an R² score of "
        f"**{r2:.3f}**."
    )


    st.divider()


    # =====================================================
    # MODEL FORMULA
    # =====================================================

    st.subheader(
        "🧮 What Did Linear Regression Learn?"
    )


    st.write(
        "The model creates a mathematical relationship "
        "between the four input features and Daily Orders."
    )


    st.code(
        formula,
        language="text"
    )


    st.metric(
        "📍 Intercept",
        f"{model.intercept_:.2f}"
    )


    st.divider()


    # =====================================================
    # COEFFICIENTS
    # =====================================================

    st.subheader(
        "📐 Feature Coefficients"
    )


    st.write(
        "The coefficient table shows the fitted relationship "
        "between each feature and Daily Orders while the "
        "other features are held constant."
    )


    st.dataframe(
        coefficient_table,

        use_container_width=True,

        hide_index=True,
    )


    st.bar_chart(
        coefficient_table.set_index(
            "Feature"
        )
    )


    st.caption(
        "A coefficient describes the fitted relationship "
        "in the model. It does not automatically prove causation."
    )


    st.divider()


    # =====================================================
    # ACTUAL VS PREDICTED
    # =====================================================

    st.subheader(
        "🎯 Actual vs Predicted Orders"
    )


    st.write(
        "This chart compares the actual test-set orders "
        "with the orders predicted by the model."
    )


    st.scatter_chart(
        test_results,

        x="Actual_Orders",

        y="Predicted_Orders",
    )


    with st.expander(
        "📋 Open the Test-Set Prediction Table"
    ):

        st.dataframe(
            test_results,

            use_container_width=True,

            hide_index=True,
        )


# =========================================================
# TAB 3 — EXPLORE DATA
# =========================================================

with data_tab:

    st.header(
        "🧾 Explore the Synthetic Dataset"
    )

    st.write(
        "Each row represents one historical classroom "
        "business day. Daily_Orders is the target."
    )


    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    data_1, data_2, data_3, data_4 = st.columns(4)


    with data_1:

        st.metric(
            "📚 Rows",
            len(df)
        )


    with data_2:

        st.metric(
            "📊 Columns",
            len(df.columns)
        )


    with data_3:

        st.metric(
            "🎯 Target",
            "Daily Orders"
        )


    with data_4:

        st.metric(
            "❓ Missing Values",
            int(
                df.isnull().sum().sum()
            )
        )


    st.divider()


    # =====================================================
    # FEATURE EXPLORER
    # =====================================================

    st.subheader(
        "🔍 Feature Explorer"
    )


    selected_feature = st.selectbox(
        "Choose a feature to compare with Daily Orders",

        options=feature_columns,
    )


    st.write(
        f"📈 **{selected_feature} vs Daily Orders**"
    )


    st.scatter_chart(
        df,

        x=selected_feature,

        y="Daily_Orders",
    )


    st.divider()


    # =====================================================
    # FULL DATASET
    # =====================================================

    show_full_data = st.checkbox(
        "📄 Show the complete dataset"
    )


    if show_full_data:

        st.dataframe(
            df,

            use_container_width=True,

            hide_index=True,
        )


    # =====================================================
    # SUMMARY STATISTICS
    # =====================================================

    with st.expander(
        "📊 View Summary Statistics"
    ):

        st.dataframe(
            df.describe().round(2),

            use_container_width=True,
        )


    # =====================================================
    # DATASET INFORMATION
    # =====================================================

    with st.expander(
        "🧠 View Dataset Column Information"
    ):

        column_info = pd.DataFrame(
            {
                "Column": df.columns,

                "Data Type": [
                    str(dtype)
                    for dtype in df.dtypes
                ],

                "Missing Values": [
                    int(
                        df[column].isnull().sum()
                    )
                    for column in df.columns
                ],
            }
        )


        st.dataframe(
            column_info,

            use_container_width=True,

            hide_index=True,
        )


    # =====================================================
    # DOWNLOAD DATASET
    # =====================================================

    st.subheader(
        "📥 Download Dataset"
    )


    csv_data = df.to_csv(
        index=False
    )


    st.download_button(
        label="📥 Download CSV Dataset",

        data=csv_data,

        file_name="daraz_daily_orders_synthetic.csv",

        mime="text/csv",

        use_container_width=True,
    )


# =========================================================
# FINAL FOOTER
# =========================================================

st.divider()

st.caption(
    "📦 Aicademy 360 | Daily Orders Predictor | "
    "Machine Learning Assignment #2"
)

st.caption(
    "🐍 Python • 🐼 Pandas • 📐 Linear Regression • "
    "📊 Scikit-Learn • 🎈 Streamlit"
)

st.caption(
    "👨‍💻 Built by Rohaan Asif"
)