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
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt


def run_modeling(df):
    """
    🧠 AI-Powered Self-Training Model Builder (with Auto SMOTE balancing)
    - Auto categorical encoding
    - Safe splitting with fallback for rare classes
    - Automatic SMOTE resampling for imbalanced data
    - Feature importance + confusion matrix visualization
    - Prediction on new uploaded data
    """

    st.subheader("🧠 AI-Powered Self-Training Model Builder with Auto SMOTE")

    if df is None or df.empty:
        st.error("❌ The dataset is empty. Please upload a valid dataset.")
        return

    # --- Step 1: Handle categorical variables ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols and not cat_cols:
        st.error("⚠️ No usable columns found (need numeric or categorical).")
        return

    df_processed = df.copy()

    # --- Encode categorical columns ---
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
        X = df_processed.drop(columns=[target])
        y = df_processed[target]

        if X.empty:
            st.error("❌ No features available for modeling.")
            return

        # --- Step 2: Optional Scaling ---
        if st.checkbox("⚙️ Standardize Numeric Features (Z-Score Scaling)"):
            scaler = StandardScaler()
            X[X.columns] = scaler.fit_transform(X[X.columns])
            st.success("✅ Numeric features standardized successfully.")

        # --- Encode target if needed ---
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            y = LabelEncoder().fit_transform(y)

        # --- Step 3: Display Target Distribution ---
        st.markdown("### 🎯 Target Label Distribution (Before Split)")
        st.dataframe(pd.Series(y).value_counts().rename_axis("Label").reset_index(name="Count"))

        # --- Step 4: Safe Split Handling ---
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = class_counts.min()

        if len(unique_classes) < 2:
            st.error("❌ Target must have at least two unique classes.")
            return

        if min_class_count < 2:
            st.warning(
                f"⚠️ The least populated class has only {min_class_count} sample(s). "
                "Using a random (non-stratified) split instead."
            )
            stratify_option = None
        else:
            stratify_option = y

        test_size = st.slider("🔀 Test Size (%)", 10, 50, 30)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, stratify=stratify_option, random_state=42
        )

        # --- Step 5: Auto SMOTE Balancing ---
        st.markdown("### ⚖️ Applying SMOTE Balancing on Training Data")
        try:
            sm = SMOTE(random_state=42, k_neighbors=min(5, min_class_count - 1))
            X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
            st.success("✅ SMOTE applied successfully! Classes are now balanced.")
            before = pd.Series(y_train).value_counts()
            after = pd.Series(y_train_bal).value_counts()
            st.write("**Before SMOTE:**")
            st.dataframe(before.rename_axis("Class").reset_index(name="Count"))
            st.write("**After SMOTE:**")
            st.dataframe(after.rename_axis("Class").reset_index(name="Count"))
        except Exception as e:
            st.warning(f"⚠️ SMOTE could not be applied: {e}")
            X_train_bal, y_train_bal = X_train, y_train

        # --- Step 6: Simulate Unlabeled Data (Semi-supervised) ---
        unlabeled_ratio = st.slider("❓ % of Unlabeled Training Data", 10, 90, 40)
        unlabeled_mask = np.random.rand(len(y_train_bal)) < (unlabeled_ratio / 100)
        y_train_partial = np.copy(y_train_bal)
        y_train_partial[~unlabeled_mask] = -1

        st.markdown("#### 🧩 Label Distribution in Training Data (After SMOTE)")
        label_dist = pd.Series(y_train_partial).replace(-1, "Unlabeled").value_counts()
        st.dataframe(label_dist.rename_axis("Label").reset_index(name="Count"))

        # --- Step 7: Configure Model ---
        n_estimators = st.number_input("🌲 Number of Trees (Random Forest)", 10, 500, 100)
        max_depth = st.slider("🌳 Max Depth", 2, 50, 10)
        min_samples_split = st.slider("🔹 Min Samples Split", 2, 10, 2)
        bootstrap = st.checkbox("🪵 Use Bootstrap Sampling", True)

        base_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            random_state=42,
            n_jobs=-1
        )

        model = SelfTrainingClassifier(base_model)

        # --- Step 8: Train Model ---
        with st.spinner("🧠 Training Self-Training Model..."):
            model.fit(X_train_bal, y_train_partial)

        # --- Step 9: Predictions ---
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"🎯 Model Accuracy: {accuracy:.4f}")

        # --- Step 10: Evaluation Report ---
        st.markdown("### 📄 Classification Report")
        st.code(classification_report(y_test, y_pred), language="text")

        # --- Step 11: Confusion Matrix ---
        st.markdown("### 🔲 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        # --- Step 12: Feature Importance ---
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

        # --- Step 13: Predict New Data ---
        st.markdown("### 🔮 Make Predictions on New Data")
        uploaded_new_data = st.file_uploader("Upload New Data for Prediction (CSV)", type=["csv"])
        if uploaded_new_data:
            new_df = pd.read_csv(uploaded_new_data)
            missing_cols = [c for c in X.columns if c not in new_df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns in uploaded file: {missing_cols}")
            else:
                preds = model.predict(new_df[X.columns])
                st.write("✅ Predictions:")
                st.dataframe(pd.DataFrame({"Prediction": preds}))

    except Exception as e:
        st.error(f"❌ Error during modeling: {e}")
