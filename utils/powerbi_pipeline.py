import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import shap

def powerbi_pipeline(df):
    st.title("📊 Power BI Style Dashboard (Advanced)")
    if df.empty:
        st.warning("⚠️ Uploaded dataset is empty. Please upload a valid file.")
        return

    overview_tab, transform_tab, dashboard_tab, insights_tab, ml_tab, adv_tab = st.tabs([
        "📁 Data Overview", "🛠️ Data Transformation", "📈 Custom Dashboard", "📌 Auto Insights", "🤖 ML Predictions", "🔬 Advanced Analytics"
    ])

    with overview_tab:
        show_data_overview(df)
    with transform_tab:
        df = transform_data(df)
    with dashboard_tab:
        create_custom_dashboard(df)
    with insights_tab:
        auto_insights(df)
    with ml_tab:
        run_advanced_ml(df)
    with adv_tab:
        adv_analytics(df)

    return df

# ------------------ Helper Functions ------------------

def show_data_overview(df):
    st.subheader("📁 Data Summary")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Data Types:", df.dtypes)

    # Advanced statistics
    st.markdown("### 📈 Statistical Summary")
    st.dataframe(df.describe(include='all').T)

    # Correlation heatmap
    st.markdown("### 🔥 Correlation Heatmap")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    # Outlier detection
    st.markdown("### 🚨 Outlier Detection")
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
        st.write(f"{col}: {len(outliers)} outliers")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("### 📊 Histograms")
    for col in num_cols:
        fig = px.histogram(df, x=col, title=f"Histogram of {col}")
        st.plotly_chart(fig, use_container_width=True)

def transform_data(df):
    st.subheader("🛠️ Data Cleaning & Normalization")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # Missing value imputation
    st.markdown("#### 🧹 Missing Value Imputation")
    impute_strategy = st.selectbox("Impute numeric columns using:", ["None", "Mean", "Median", "Zero"])
    if impute_strategy != "None":
        for col in num_cols:
            if impute_strategy == "Mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif impute_strategy == "Median":
                df[col].fillna(df[col].median(), inplace=True)
            elif impute_strategy == "Zero":
                df[col].fillna(0, inplace=True)
        st.success("✅ Missing values imputed.")

    # Categorical encoding
    st.markdown("#### 🏷️ Encode Categorical Columns")
    enc_cols = st.multiselect("Select columns to encode (Label Encoding)", cat_cols)
    if st.button("Encode Selected Columns"):
        for col in enc_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        st.success("✅ Label encoding applied.")

    # Normalization
    st.markdown("#### ⚖️ Normalize Numeric Columns")
    norm_cols = st.multiselect("Select columns to normalize (StandardScaler)", num_cols)
    if st.button("Standardize Selected Columns"):
        scaler = StandardScaler()
        df[norm_cols] = scaler.fit_transform(df[norm_cols])
        st.success("✅ Standardization applied.")

    st.subheader("📏 KPI Metrics")
    kpi_cols = st.multiselect("KPI metric columns", num_cols, default=num_cols)
    for col in kpi_cols:
        st.metric(f"Mean of {col}", round(df[col].mean(), 2))
        st.metric(f"Max of {col}", round(df[col].max(), 2))
        st.metric(f"Min of {col}", round(df[col].min(), 2))

    # Download cleaned data
    st.download_button("⬇️ Download Cleaned Data", df.to_csv(index=False), file_name="cleaned_data.csv", mime="text/csv")
    return df

def create_custom_dashboard(df):
    st.subheader("🎯 Create Custom Dashboard")
    filter_col = st.selectbox("🔍 Select column to filter", ["None"] + list(df.columns))
    selected_vals = None
    if filter_col != "None":
        unique_vals = df[filter_col].dropna().unique()
        selected_vals = st.multiselect(f"Filter values for `{filter_col}`", unique_vals, default=list(unique_vals))
        df = df[df[filter_col].isin(selected_vals)]

    if df.empty:
        st.warning("No data to plot after filtering.")
        return

    x_axis = st.selectbox("Select X-axis Column", df.columns)
    y_axis = st.selectbox("Select Y-axis Column", df.select_dtypes(include=np.number).columns)
    chart_type = st.radio("Select Chart Type", ["Bar", "Line", "Scatter", "Area", "Pie", "Box"])

    fig = generate_chart(df, x_axis, y_axis, chart_type)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        export_chart(fig)
        save_load_config(x_axis, y_axis, chart_type, filter_col, selected_vals if filter_col != "None" else None)

def generate_chart(df, x_axis, y_axis, chart_type):
    if chart_type == "Bar":
        return px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
    elif chart_type == "Line":
        return px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
    elif chart_type == "Scatter":
        return px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
    elif chart_type == "Area":
        return px.area(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
    elif chart_type == "Pie":
        return px.pie(df, names=x_axis, title=f"Pie Chart of {x_axis}")
    elif chart_type == "Box":
        return px.box(df, x=x_axis, y=y_axis, title=f"Box Plot of {y_axis} by {x_axis}")
    return None

def export_chart(fig):
    st.markdown("### 📤 Export Chart")
    img_bytes = fig.to_image(format="png")
    st.download_button("📸 Download as PNG", data=img_bytes, file_name="chart.png", mime="image/png")
    html = fig.to_html()
    st.download_button("📄 Download as HTML", data=html.encode(), file_name="chart.html", mime="text/html")

def save_load_config(x, y, chart_type, filter_col, selected_vals):
    config = {
        "x_axis": x,
        "y_axis": y,
        "chart_type": chart_type,
        "filter_column": filter_col,
        "selected_values": selected_vals if filter_col != "None" else None
    }
    config_json = json.dumps(config, indent=4)
    st.download_button("💾 Download Chart Config", config_json, file_name="chart_config.json", mime="application/json")
    uploaded_file = st.file_uploader("📤 Upload Config File", type="json")
    if uploaded_file:
        try:
            config_loaded = json.load(uploaded_file)
            st.success("✅ Config loaded:")
            st.json(config_loaded)
        except Exception as e:
            st.error(f"Failed to load config: {e}")

def auto_insights(df):
    st.subheader("📌 Auto Insights")
    num_cols = df.select_dtypes(include=np.number).columns
    if num_cols.empty:
        st.warning("No numeric columns available.")
        return
    col = st.selectbox("Select a numeric column", num_cols)
    if col:
        st.write(df[col].describe())
        st.markdown("#### Distribution Plot")
        st.plotly_chart(px.histogram(df, x=col), use_container_width=True)
        st.markdown("#### Value Counts")
        st.write(df[col].value_counts().head(10))

        # Outlier analysis
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
        st.write(f"Outliers detected: {len(outliers)}")

def run_advanced_ml(df):
    st.subheader("🤖 ML Classifier: Random Forest (Advanced)")
    cols = df.select_dtypes(include=[np.number, 'object']).columns.tolist()
    if len(cols) < 2:
        st.warning("Insufficient columns for ML.")
        return

    label_col = st.selectbox("🎯 Target Column", cols)
    feature_cols = st.multiselect("🧩 Feature Columns", [c for c in cols if c != label_col])

    smote_on = st.checkbox("Apply SMOTE for class imbalance (classification only)", value=True)
    crossval_on = st.checkbox("Use Stratified 5-Fold Cross-Validation", value=True)
    gridsearch_on = st.checkbox("Run GridSearchCV for hyperparameter tuning", value=False)

    if st.button("🚀 Train Model"):
        try:
            X = pd.get_dummies(df[feature_cols])
            y = df[label_col]
            if y.dtype == 'O':
                y = pd.factorize(y)[0]

            if smote_on:
                try:
                    X, y = SMOTE().fit_resample(X, y)
                    st.success("✅ SMOTE applied for class balance.")
                except Exception as e:
                    st.warning(f"SMOTE failed: {e}")

            model = RandomForestClassifier()
            params = {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}
            if gridsearch_on:
                model = GridSearchCV(model, params, cv=3)
                st.write("GridSearchCV enabled.")

            if crossval_on:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
                st.write(f"Cross-Validation Scores: {scores}")
                st.write(f"Mean Accuracy: {scores.mean():.3f}")
                # Fit on full data for reporting
                model.fit(X, y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())
                score = model.score(X_test, y_test)
                st.metric("Test Accuracy", f"{score:.3f}")

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                st.markdown("### Confusion Matrix")
                cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
                st.plotly_chart(cm_fig, use_container_width=True)

                # ROC AUC if binary
                if len(np.unique(y)) == 2:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                    st.metric("ROC-AUC", f"{auc:.3f}")

            # Feature importance
            st.markdown("### 🔍 Feature Importance")
            importances = model.best_estimator_.feature_importances_ if hasattr(model, "best_estimator_") else model.feature_importances_
            importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            st.dataframe(importance_df.head(10))

            # SHAP values (explainability)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X[:100])
                st.markdown("### 🧠 SHAP Summary Plot")
                shap.summary_plot(shap_values, X[:100], plot_type="bar", show=False)
                st.pyplot(bbox_inches='tight')
            except Exception as e:
                st.info(f"SHAP visualization not available: {e}")

        except Exception as e:
            st.error(f"🚫 Error training model: {e}")

def adv_analytics(df):
    st.subheader("🔬 Advanced Analytics Playground")
    st.write("💡 Use Python code (pandas, numpy, plotly, etc.) to analyze the dataframe below:")
    code = st.text_area("Python code (your variable is 'df')", height=150)
    if st.button("Run"):
        try:
            # Provide a local dictionary for execution context
            local_vars = {"df": df, "pd": pd, "np": np, "px": px, "st": st}
            exec(code, {}, local_vars)
        except Exception as e:
            st.error(f"Error in code: {e}")
