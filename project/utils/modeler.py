import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from imblearn.over_sampling import SMOTE


def run_modeling(df):
    """
    🤖 AI AutoML Model Builder with Self-Training & SMOTE
    - Auto-encodes categorical columns
    - Automatically balances data using SMOTE
    - Trains & compares multiple algorithms
    - Displays leaderboard of model performance
    - Shows feature importance and confusion matrix
    """
    st.subheader("🤖 AI AutoML Model Builder + Self-Training + SMOTE")

    if df is None or df.empty:
        st.error("❌ The dataset is empty. Please upload a valid dataset.")
        return

    # --- Auto Encode Categoricals ---
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

    try:
        X = df.drop(columns=[target])
        y = df[target]

        if X.empty:
            st.error("❌ No feature columns available for modeling.")
            return

        # --- Encode target if categorical ---
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            y = LabelEncoder().fit_transform(y)

        # --- Optional Feature Scaling ---
        if st.checkbox("⚙️ Standardize Numeric Features (Z-Score Scaling)"):
            scaler = StandardScaler()
            X[X.columns] = scaler.fit_transform(X[X.columns])
            st.success("✅ Numeric features standardized.")

        # --- Show Class Distribution ---
        st.markdown("### 🎯 Target Label Distribution (Before Split)")
        st.dataframe(pd.Series(y).value_counts().rename_axis("Label").reset_index(name="Count"))

        # --- Safe Split ---
        test_size = st.slider("🔀 Test Size (%)", 10, 50, 30)
        unique_classes, counts = np.unique(y, return_counts=True)
        stratify_opt = y if min(counts) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, stratify=stratify_opt, random_state=42
        )

        # --- Apply SMOTE Balancing ---
        st.markdown("### ⚖️ Applying SMOTE to Balance Classes")
        try:
            sm = SMOTE(random_state=42, k_neighbors=min(5, max(1, min(counts) - 1)))
            X_train, y_train = sm.fit_resample(X_train, y_train)
            st.success("✅ SMOTE applied successfully.")
        except Exception as e:
            st.warning(f"⚠️ SMOTE skipped: {e}")

        # --- Semi-supervised Unlabeled Simulation ---
        unlabeled_ratio = st.slider("❓ % of Unlabeled Training Data (for Self-Training)", 10, 90, 30)
        unlabeled_mask = np.random.rand(len(y_train)) < (unlabeled_ratio / 100)
        y_train_partial = np.copy(y_train)
        y_train_partial[~unlabeled_mask] = -1

        st.markdown("### 🧩 Label Distribution (After SMOTE & Semi-supervised Split)")
        label_dist = pd.Series(y_train_partial).replace(-1, "Unlabeled").value_counts()
        st.dataframe(label_dist.rename_axis("Label").reset_index(name="Count"))

        # --- Model List for AutoML Comparison ---
        st.markdown("## ⚙️ Model Configuration")

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "Logistic Regression": LogisticRegression(max_iter=500, solver="lbfgs"),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
            "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        include_self_training = st.checkbox("🧠 Include Self-Training (Semi-Supervised Mode)", True)
        if include_self_training:
            base_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            models["Self-Training (Random Forest)"] = SelfTrainingClassifier(base_rf)

        # --- Train & Evaluate Models ---
        st.markdown("## 🧠 Training Models & Comparing Performance...")
        results = []
        progress_bar = st.progress(0)
        total_models = len(models)

        for i, (name, model) in enumerate(models.items(), 1):
            try:
                with st.spinner(f"Training {name}..."):
                    model.fit(X_train, y_train_partial)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="weighted")
                    prec = precision_score(y_test, preds, average="weighted")
                    rec = recall_score(y_test, preds, average="weighted")
                    results.append([name, acc, f1, prec, rec])
                progress_bar.progress(i / total_models)
            except Exception as e:
                st.warning(f"⚠️ {name} failed: {e}")

        # --- Show Leaderboard ---
        results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1-Score", "Precision", "Recall"])
        results_df = results_df.sort_values(by="F1-Score", ascending=False)
        st.success("✅ Model Comparison Complete!")

        st.markdown("### 🏆 Model Performance Leaderboard")
        st.dataframe(results_df.reset_index(drop=True).style.highlight_max(axis=0, color="lightgreen"))

        # --- Highlight Best Model ---
        best_model_name = results_df.iloc[0]["Model"]
        best_acc = results_df.iloc[0]["Accuracy"]
        st.balloons()
        st.markdown(f"🎉 **Best Model:** `{best_model_name}` with Accuracy = {best_acc:.4f}")

        # --- Optional: Re-train Best Model on Full Data ---
        if st.checkbox("🔁 Retrain Best Model on Full Data"):
            best_model = models[best_model_name]
            best_model.fit(X, y)
            st.success(f"✅ {best_model_name} retrained on full dataset.")

        # --- Confusion Matrix for Best Model ---
        st.markdown("### 🔲 Confusion Matrix (Best Model)")
        best_model = models[best_model_name]
        preds = best_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {best_model_name}")
        st.pyplot(fig)

        # --- Feature Importance (if supported) ---
        st.markdown("### 🌟 Feature Importance")
        if hasattr(best_model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": best_model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            st.dataframe(importance_df)
            fig, ax = plt.subplots()
            sns.barplot(
                data=importance_df.head(15),
                x="Importance",
                y="Feature",
                ax=ax
            )
            plt.title("Top Feature Importances")
            st.pyplot(fig)
        else:
            st.info(f"ℹ️ Feature importances not available for {best_model_name}")

        # --- Prediction on New Data ---
        st.markdown("### 🔮 Make Predictions on New Data")
        uploaded_file = st.file_uploader("Upload New Data for Prediction (CSV)", type=["csv"])
        if uploaded_file:
            new_df = pd.read_csv(uploaded_file)
            missing_cols = [c for c in X.columns if c not in new_df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns in uploaded file: {missing_cols}")
            else:
                preds = best_model.predict(new_df[X.columns])
                st.dataframe(pd.DataFrame({"Prediction": preds}))
                st.success("✅ Predictions completed!")

    except Exception as e:
        st.error(f"❌ Error during modeling: {e}")
