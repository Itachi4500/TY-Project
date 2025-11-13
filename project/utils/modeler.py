import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.semi_supervised import SelfTrainingClassifier
from imblearn.over_sampling import SMOTE


def run_modeling(df):
    """
    🤖 AI AutoML Model Builder with Automatic Classification/Regression Mode
    + SMOTE balancing (for classification)
    + Self-Training support (semi-supervised)
    + Auto feature scaling, feature importance, confusion/regression charts
    """
    st.subheader("🧠 AI AutoML Model Builder (Classification + Regression)")

    if df is None or df.empty:
        st.error("❌ The dataset is empty. Please upload a valid dataset.")
        return

    # --- Encode categorical variables ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        st.info(f"🔤 Encoding {len(cat_cols)} categorical columns automatically...")
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # --- Select Target Column ---
    target = st.selectbox("🎯 Select Target Column", df.columns)
    if not target:
        st.warning("⚠️ Please select a target column.")
        return

    X = df.drop(columns=[target])
    y = df[target]

    if X.empty:
        st.error("❌ No feature columns available for modeling.")
        return

    # --- Detect Task Type ---
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        task_type = "regression"
    else:
        task_type = "classification"

    st.markdown(f"### 🔍 Detected Task Type: **{task_type.title()}**")

    # --- Optional Scaling ---
    if st.checkbox("⚙️ Standardize Numeric Features (Z-Score Scaling)"):
        scaler = StandardScaler()
        X[X.columns] = scaler.fit_transform(X[X.columns])
        st.success("✅ Features standardized successfully.")

    # --- Split Dataset ---
    test_size = st.slider("🔀 Test Size (%)", 10, 100, 30)
    stratify_opt = y if (task_type == "classification" and y.nunique() > 1) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, stratify=stratify_opt, random_state=42
    )

    # --- Apply SMOTE only for Classification ---
    if task_type == "classification":
        st.markdown("### ⚖️ Applying SMOTE Balancing on Training Data")
        try:
            sm = SMOTE(random_state=42, k_neighbors=min(5, y_train.value_counts().min() - 1))
            X_train, y_train = sm.fit_resample(X_train, y_train)
            st.success("✅ SMOTE applied successfully (classes balanced).")
        except Exception as e:
            st.warning(f"⚠️ SMOTE skipped: {e}")

    # --- Define Models Dynamically ---
    if task_type == "classification":
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
            "SVM (RBF Kernel)": SVC(probability=True, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
        if st.checkbox("🧠 Include Self-Training (Semi-Supervised)", True):
            models["Self-Training (Random Forest)"] = SelfTrainingClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    else:
        models = {
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "SVR (RBF Kernel)": SVR(kernel="rbf"),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        }

    # --- Train and Evaluate Models ---
    st.markdown("## 🧠 Training Models & Comparing Performance...")
    results = []
    progress_bar = st.progress(0)

    for i, (name, model) in enumerate(models.items(), 1):
        try:
            with st.spinner(f"Training {name}..."):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                if task_type == "classification":
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="weighted")
                    prec = precision_score(y_test, preds, average="weighted")
                    rec = recall_score(y_test, preds, average="weighted")
                    results.append([name, acc, f1, prec, rec])
                else:
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    results.append([name, r2, mae, rmse])
        except Exception as e:
            st.warning(f"⚠️ {name} failed: {e}")
        progress_bar.progress(i / len(models))

    # --- Results Table ---
    if task_type == "classification":
        results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1-Score", "Precision", "Recall"])
        best_model_name = results_df.iloc[results_df["F1-Score"].idxmax()]["Model"]
    else:
        results_df = pd.DataFrame(results, columns=["Model", "R²", "MAE", "RMSE"])
        best_model_name = results_df.iloc[results_df["R²"].idxmax()]["Model"]

    st.success("✅ Model Comparison Complete!")
    st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"))

    # --- Best Model ---
    best_model = models[best_model_name]
    st.markdown(f"🏆 **Best Model:** `{best_model_name}`")
    st.balloons()

    # --- Retrain on Full Data ---
    if st.checkbox("🔁 Retrain Best Model on Full Dataset"):
        best_model.fit(X, y)
        st.success(f"✅ {best_model_name} retrained on full dataset.")

    # --- Visualization ---
    if task_type == "classification":
        st.markdown("### 🔲 Confusion Matrix (Best Model)")
        preds = best_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {best_model_name}")
        st.pyplot(fig)
    else:
        st.markdown("### 📈 Regression Performance")
        preds = best_model.predict(X_test)
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=preds, alpha=0.7)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{best_model_name} Predictions vs Actuals")
        st.pyplot(fig)
        st.info(f"R² = {r2_score(y_test, preds):.4f}, RMSE = {np.sqrt(mean_squared_error(y_test, preds)):.4f}")

    # --- Feature Importance ---
    if hasattr(best_model, "feature_importances_"):
        st.markdown("### 🌟 Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": best_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.dataframe(importance_df)
        fig, ax = plt.subplots()
        sns.barplot(data=importance_df.head(15), x="Importance", y="Feature", ax=ax)
        plt.title("Top Feature Importances")
        st.pyplot(fig)

    # --- New Data Prediction ---
    st.markdown("### 🔮 Predict on New Data")
    uploaded_file = st.file_uploader("Upload New Data for Prediction (CSV)", type=["csv"])
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        missing = [c for c in X.columns if c not in new_df.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}")
        else:
            preds = best_model.predict(new_df[X.columns])
            st.dataframe(pd.DataFrame({"Prediction": preds}))
            st.success("✅ Predictions completed!")

