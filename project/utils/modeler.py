import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import importlib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.semi_supervised import SelfTrainingClassifier
from imblearn.over_sampling import SMOTE

# Try optional libraries (XGBoost, LightGBM, CatBoost, SHAP)
xgb = importlib.util.find_spec("xgboost")
lgb = importlib.util.find_spec("lightgbm")
catb = importlib.util.find_spec("catboost")
shap_spec = importlib.util.find_spec("shap")

if xgb:
    import xgboost as xgboost
if lgb:
    import lightgbm as lightgbm
if catb:
    from catboost import CatBoostClassifier, CatBoostRegressor
if shap_spec:
    import shap


def interpret_classification_results(acc, f1, prec, rec):
    avg_score = np.mean([acc, f1, prec, rec])
    if avg_score > 0.90:
        verdict = "🏆 **Excellent Model!** High accuracy and balanced metrics."
    elif avg_score > 0.75:
        verdict = "✅ **Good Model.** Reliable for many tasks."
    elif avg_score > 0.60:
        verdict = "⚠️ **Moderate Performance.** Consider tuning/engineering."
    else:
        verdict = "❌ **Poor Model.** Try different algorithms or more data."
    tips = [
        "- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV).",
        "- Feature engineering / remove noisy features.",
        "- Collect more data for underrepresented classes.",
        "- Try ensemble methods (XGBoost/LightGBM/CatBoost)."
    ]
    return verdict, tips


def interpret_regression_results(r2, rmse, mae):
    if r2 > 0.90:
        verdict = "🏆 **Excellent Model!** Explains most variance."
    elif r2 > 0.75:
        verdict = "✅ **Good Model.** Strong predictive power."
    elif r2 > 0.50:
        verdict = "⚠️ **Moderate Model.** Improve with features/tuning."
    else:
        verdict = "❌ **Weak Model.** Consider non-linear or ensemble models."
    tips = [
        "- Try polynomial/interactions or tree-based models.",
        "- Tune hyperparameters for tree-based learners.",
        "- Remove outliers or transform skewed features.",
        "- Use Gradient Boosting / XGBoost / LightGBM / CatBoost."
    ]
    return verdict, tips


def safe_create_tree_explainer(model, X_sample):
    """Return a SHAP explainer if possible, else None"""
    if not shap_spec:
        return None, "shap not installed"
    try:
        # TreeExplainer supports tree models & many boosters
        explainer = shap.TreeExplainer(model)
        # compute shap values on a reasonable sample (or full if small)
        shap_values = explainer.shap_values(X_sample)
        return (explainer, shap_values), None
    except Exception as e:
        # fallback to KernelExplainer for small datasets could be expensive; skip
        return None, str(e)


def run_modeling(df):
    """
    AutoML Modeler with XGBoost/LightGBM/CatBoost + SHAP explainability (optional).
    - Automatically adds optional models when libraries available.
    - Option to enable GPU acceleration (if supported by installed libs).
    """
    st.subheader("🤖 AutoML + XGBoost/LightGBM/CatBoost + SHAP")

    if df is None or df.empty:
        st.error("❌ Empty dataset — please upload a valid dataset.")
        return

    # --- encode categorical columns if needed ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        st.info(f"🔤 Encoding {len(cat_cols)} categorical columns automatically...")
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # --- target selection ---
    target = st.selectbox("🎯 Select Target Column", df.columns)
    if not target:
        st.warning("⚠️ Select a target column.")
        return

    X = df.drop(columns=[target])
    y = df[target]

    if X.empty:
        st.error("❌ No feature columns available.")
        return

    # --- detect task ---
    task_type = "regression" if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15 else "classification"
    st.markdown(f"### 🔎 Detected task type: **{task_type.title()}**")

    # --- optional scaling ---
    if st.checkbox("⚙️ Standardize numeric features (Z-score)"):
        scaler = StandardScaler()
        X[X.columns] = scaler.fit_transform(X[X.columns])
        st.success("✅ Features standardized")

    # --- split dataset safely ---
    test_size = st.slider("🔀 Test size (%)", 10, 50, 30)
    stratify_opt = y if (task_type == "classification" and y.nunique() > 1) else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, stratify=stratify_opt, random_state=42
        )
    except Exception as e:
        st.warning(f"⚠️ Stratified split failed: {e} — falling back to random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=42
        )

    # --- classification-only: SMOTE balancing ---
    if task_type == "classification":
        st.markdown("### ⚖️ Balancing with SMOTE (classification only)")
        try:
            min_count = y_train.value_counts().min()
            k_neighbors = max(1, min(5, min_count - 1))
            sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            st.success("✅ SMOTE applied")
        except Exception as e:
            st.warning(f"⚠️ SMOTE skipped: {e}")

    # --- Model selection UI for optional libraries & GPU ---
    st.markdown("### ⚙️ Optional Boosted Models")
    enable_xgb = False
    enable_lgb = False
    enable_catb = False
    use_gpu = False

    if xgb:
        enable_xgb = st.checkbox("Include XGBoost", value=False)
    else:
        st.info("XGBoost not installed — skip. (pip install xgboost)")

    if lgb:
        enable_lgb = st.checkbox("Include LightGBM", value=False)
    else:
        st.info("LightGBM not installed — skip. (pip install lightgbm)")

    if catb:
        enable_catb = st.checkbox("Include CatBoost", value=False)
    else:
        st.info("CatBoost not installed — skip. (pip install catboost)")

    if any([enable_xgb, enable_lgb, enable_catb]):
        use_gpu = st.checkbox("Attempt GPU acceleration for supported boosters (if available)", value=False)

    # --- base model dict ---
    models = {}
    # Classic models (always available)
    if task_type == "classification":
        models.update({
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
            "SVM (RBF)": SVC(probability=True),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        })
    else:
        models.update({
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "SVR (RBF)": SVR(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
        })

    # Optional boosted models (add safely)
    try:
        if enable_xgb:
            if task_type == "classification":
                if use_gpu:
                    # try GPU tree_method
                    xgb_params = {"tree_method": "gpu_hist"}
                else:
                    xgb_params = {"tree_method": "hist"}
                models["XGBoost"] = xgboost.XGBClassifier(use_label_encoder=False, eval_metric="logloss", **xgb_params, random_state=42)
            else:
                models["XGBoost Regressor"] = xgboost.XGBRegressor(**({"tree_method": "gpu_hist"} if use_gpu else {}), random_state=42)
    except Exception as e:
        st.warning(f"⚠️ Could not create XGBoost model: {e}")

    try:
        if enable_lgb:
            if task_type == "classification":
                if use_gpu:
                    models["LightGBM"] = lightgbm.LGBMClassifier(device="gpu", random_state=42)
                else:
                    models["LightGBM"] = lightgbm.LGBMClassifier(random_state=42)
            else:
                if use_gpu:
                    models["LightGBM Regressor"] = lightgbm.LGBMRegressor(device="gpu", random_state=42)
                else:
                    models["LightGBM Regressor"] = lightgbm.LGBMRegressor(random_state=42)
    except Exception as e:
        st.warning(f"⚠️ Could not create LightGBM model: {e}")

    try:
        if enable_catb:
            if task_type == "classification":
                models["CatBoost"] = CatBoostClassifier(task_type="GPU" if use_gpu else "CPU", verbose=0, random_state=42)
            else:
                models["CatBoost Regressor"] = CatBoostRegressor(task_type="GPU" if use_gpu else "CPU", verbose=0, random_state=42)
    except Exception as e:
        st.warning(f"⚠️ Could not create CatBoost model: {e}")

    # Optionally include self-training for classification
    if task_type == "classification" and st.checkbox("Include Self-Training (semi-supervised)", value=True):
        models["Self-Training (RF)"] = SelfTrainingClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

    # --- Train & evaluate models ---
    st.markdown("## ▶️ Training models (this may take a while depending on dataset & models)...")
    results = []
    progress = st.progress(0)
    total = len(models)
    i = 0

    for name, model in list(models.items()):
        i += 1
        try:
            with st.spinner(f"Training {name}..."):
                # fit
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                if task_type == "classification":
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="weighted")
                    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
                    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
                    results.append([name, acc, f1, prec, rec])
                else:
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    results.append([name, r2, mae, rmse])
        except Exception as e:
            st.warning(f"⚠️ {name} failed: {e}")
        progress.progress(i / total)

    # --- results table ---
    if task_type == "classification":
        results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1-Score", "Precision", "Recall"])
        if results_df.empty:
            st.error("❌ No successful models to display.")
            return
        best_idx = results_df["F1-Score"].idxmax()
    else:
        results_df = pd.DataFrame(results, columns=["Model", "R²", "MAE", "RMSE"])
        if results_df.empty:
            st.error("❌ No successful models to display.")
            return
        best_idx = results_df["R²"].idxmax()

    results_df = results_df.sort_values(by=results_df.columns[1], ascending=False)
    st.success("✅ Model training & evaluation complete!")
    st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"))

    best_model_name = results_df.iloc[0]["Model"]
    st.markdown(f"### 🏆 Best model: **{best_model_name}**")

    # retrieve best model object
    best_model = None
    for n, m in models.items():
        if n == best_model_name:
            best_model = m
            break

    # retrain on full data option
    if st.checkbox("🔁 Retrain best model on full dataset", value=False):
        try:
            best_model.fit(X, y)
            st.success("✅ Best model retrained on full dataset.")
        except Exception as e:
            st.warning(f"⚠️ Retrain failed: {e}")

    # --- Visuals: confusion/regression plot ---
    if task_type == "classification":
        preds = best_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        st.markdown("### 🔲 Confusion Matrix (best model)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # classification metrics & interpretation
        best_row = results_df[results_df["Model"] == best_model_name].iloc[0]
        verdict, tips = interpret_classification_results(best_row["Accuracy"], best_row["F1-Score"], best_row["Precision"], best_row["Recall"])
        st.markdown(verdict)
        with st.expander("💡 Improvement suggestions"):
            for t in tips:
                st.write(t)

    else:
        preds = best_model.predict(X_test)
        st.markdown("### 📈 Predictions vs Actual (best model)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=y_test, y=preds, alpha=0.6, ax=ax)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        verdict, tips = interpret_regression_results(r2, rmse, mae)
        st.markdown(verdict)
        with st.expander("💡 Improvement suggestions"):
            for t in tips:
                st.write(t)
        st.info(f"R² = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}")

    # --- SHAP explainability (best-effort) ---
    if shap_spec:
        st.markdown("### 🔍 SHAP Explainability (Best Model) — best-effort")
        try:
            # choose a reasonable sample for SHAP (to keep compute moderate)
            sample_size = min(200, X_test.shape[0])
            X_sample = X_test.sample(sample_size, random_state=42)
            explainer_data, err = safe_create_tree_explainer(best_model, X_sample)
            if explainer_data is None:
                st.info(f"SHAP unavailable for this model: {err}")
            else:
                explainer, shap_values = explainer_data
                st.markdown("#### 🌟 SHAP feature importance (mean absolute value)")
                # shap_values may be a list (multi-class) or array (binary/regression)
                try:
                    if isinstance(shap_values, list):
                        # multi-class: sum mean abs across classes
                        mean_abs = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
                    else:
                        mean_abs = np.abs(shap_values).mean(axis=0)
                    imp_df = pd.DataFrame({"feature": X_sample.columns, "shap_value": mean_abs})
                    imp_df = imp_df.sort_values("shap_value", ascending=False)
                    st.dataframe(imp_df.head(20).reset_index(drop=True))
                    fig, ax = plt.subplots(figsize=(6, min(0.4 * len(imp_df.head(20)), 6)))
                    sns.barplot(data=imp_df.head(20), x="shap_value", y="feature", ax=ax)
                    ax.set_title("Top SHAP feature importance")
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"⚠️ SHAP importance failed: {e}")

                # dependence plot for a top feature
                try:
                    top_feat = imp_df.iloc[0]["feature"]
                    st.markdown(f"#### Dependence plot for top feature: `{top_feat}`")
                    # shap.dependence_plot draws to matplotlib by default if show=False
                    plt.figure()
                    shap.dependence_plot(top_feat, shap_values, X_sample, show=False)
                    st.pyplot(plt)
                except Exception as e:
                    st.warning(f"⚠️ SHAP dependence plot failed: {e}")
        except Exception as e:
            st.warning(f"⚠️ SHAP explainability failed: {e}")
    else:
        st.info("Install `shap` to enable SHAP explainability (pip install shap).")

    # --- Predict on new uploaded data ---
    st.markdown("### 🔮 Make predictions on new data (CSV)")
    uploaded_file = st.file_uploader("Upload new CSV for prediction (must contain training features)", type=["csv"])
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        missing = [c for c in X.columns if c not in new_df.columns]
        if missing:
            st.error(f"❌ Missing columns in uploaded file: {missing}")
        else:
            try:
                preds = best_model.predict(new_df[X.columns])
                st.dataframe(pd.DataFrame({"Prediction": preds}))
                st.success("✅ Predictions complete.")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

    # final note
    st.markdown("---")
    st.caption("Tip: If you enable XGBoost/LightGBM/CatBoost with GPU, ensure proper drivers & binaries are installed on the host.")

