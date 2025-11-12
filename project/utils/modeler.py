import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt

def run_modeling(df):
    """
    🧠 AI-Powered Self-Training Model Builder
    Enhanced version with:
    - Auto preprocessing
    - Feature importance visualization
    - Confusion matrix plot
    - Smart label handling
    """
    st.subheader("🧠 AI-Powered Self-Training Model Builder")

    # --- Basic Validation ---
    if df is None or df.empty:
        st.error("❌ The dataset is empty. Please upload a valid dataset.")
        return

    # --- Handle categorical variables ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols and not cat_cols:
        st.error("⚠️ No usable columns found in dataset (need numeric or categorical).")
        return

    df_processed = df.copy()

    # --- Label Encoding for categorical features ---
    if cat_cols:
        st.info(f"🔤 Encoding {len(cat_cols)} categorical columns automatically...")
        for col in cat_cols:
            df_processed[col] = LabelEncoder().fit_transform(df_processed[col].astype(str))

    # --- Select Target Column ---
    target = st.selectbox("🎯 Select Target Column", df_processed.columns)
    if not target:
        st.warning("⚠️ Please select a target column.")
        return

    try:
        # --- Feature / Target Split ---
        X = df_processed.drop(columns=[target])
        y = df_processed[target]

        if X.empty:
            st.error("❌ No features available for modeling.")
            return

        # --- Optional Scaling ---
        if st.checkbox("⚙️ Standardize Numeric Features (Z-Score Scaling)"):
            scaler = StandardScaler()
            X[X.columns] = scaler.fit_transform(X[X.columns])
            st.success("✅ Numeric features standardized successfully.")

        # --- Encode Target if needed ---
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            y = LabelEncoder().fit_transform(y)

        # --- Split Dataset ---
        test_size = st.slider("🔀 Test Size (%)", 10, 50, 30)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, stratify=y, random_state=42
        )

        # --- Handle Class Imbalance ---
        unique_classes = np.unique(y_train)
        if len(unique_classes) > 1:
            class_weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
            weights = dict(zip(unique_classes, class_weights))
            st.write("⚖️ Computed Class Weights:", weights)
        else:
            st.warning("⚠️ Only one class present in target. Class weights skipped.")
            weights = None

        # --- Simulate Unlabeled Data ---
        unlabeled_ratio = st.slider("❓ % of Unlabeled Training Data", 10, 90, 40)
        unlabeled_mask = np.random.rand(len(y_train)) < (unlabeled_ratio / 100)
        y_train_partial = np.copy(y_train)
        y_train_partial[~unlabeled_mask] = -1

        st.markdown("#### 🧩 Label Distribution in Training Data")
        label_dist = pd.Series(y_train_partial).replace(-1, "Unlabeled").value_counts()
        st.dataframe(label_dist.rename_axis("Label").reset_index(name="Count"))

        # --- Model Configuration ---
        n_estimators = st.number_input("🌲 Number of Trees (Random Forest)", 10, 500, 100)
        max_depth = st.slider("🌳 Max Depth", 2, 50, 10)
        min_samples_split = st.slider("🔹 Min Samples Split", 2, 10, 2)
        bootstrap = st.checkbox("🪵 Use Bootstrap Sampling", True)

        base_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=weights,
            bootstrap=bootstrap,
            random_state=42,
            n_jobs=-1
        )

        model = SelfTrainingClassifier(base_model)

        # --- Model Training ---
        with st.spinner("🧠 Training Self-Training Model..."):
            model.fit(X_train, y_train_partial)

        # --- Predictions ---
        y_pred = model.predict(X_test)

        # --- Evaluation ---
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"🎯 Model Accuracy: {accuracy:.4f}")

        st.markdown("### 📄 Classification Report")
        st.code(classification_report(y_test, y_pred), language="text")

        # --- Confusion Matrix ---
        st.markdown("### 🔲 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        # --- Feature Importance ---
        st.markdown("### 🌟 Feature Importance")
        try:
            feature_importances = model.base_estimator_.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": feature_importances
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
        except Exception as e:
            st.warning(f"⚠️ Could not compute feature importances: {e}")

        # --- Predict New Data ---
        st.markdown("### 🔮 Make Predictions on New Data")
        uploaded_new_data = st.file_uploader("Upload New Data for Prediction (CSV)", type=["csv"])
        if uploaded_new_data:
            new_df = pd.read_csv(uploaded_new_data)
            if set(X.columns).issubset(new_df.columns):
                preds = model.predict(new_df[X.columns])
                st.write("✅ Predictions:")
                st.dataframe(pd.DataFrame({"Prediction": preds}))
            else:
                st.error("❌ Uploaded file does not match training features.")

    except Exception as e:
        st.error(f"❌ Error during modeling: {e}")
