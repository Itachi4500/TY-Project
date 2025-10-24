import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.preprocessing import LabelEncoder

def run_modeling(df):
    st.subheader("🧠 Self-Training Model Builder")

    if df.empty:
        st.error("❌ The dataset is empty.")
        return

    # Keep only numeric columns (including target)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.error("❌ No numeric columns found in the dataset.")
        return

    # Target column selection
    target = st.selectbox("🎯 Select Target Column", numeric_df.columns)
    if not target:
        st.warning("⚠️ Please select a target column.")
        return

    # Split features and target
    try:
        features = numeric_df.drop(columns=[target])
        target_series = numeric_df[target]

        if features.empty:
            st.error("❌ No numeric feature columns available for modeling.")
            return

        # No need for get_dummies, as only numeric columns are used
        X = features
        y = LabelEncoder().fit_transform(target_series)

        # Train-test split
        test_size = st.slider("🔀 Test Size (%)", 10, 50, 30)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, stratify=y, random_state=42
        )

        # Simulate unlabeled data (semi-supervised learning)
        unlabeled_ratio = st.slider("❓ % of Unlabeled Training Data", 10, 90, 50)
        mask = np.random.rand(len(y_train)) < (unlabeled_ratio / 100)
        y_train_partial = np.copy(y_train)
        y_train_partial[~mask] = -1

        st.markdown("🔎 Label Distribution (Training)")
        label_dist = pd.Series(y_train_partial).replace(-1, 'Unlabeled').value_counts()
        st.dataframe(label_dist.rename_axis("Label").reset_index(name="Count"))

        # Model settings
        n_estimators = st.number_input("🌲 Number of Trees (Random Forest)", 10, 500, 100)
        base_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self_training_model = SelfTrainingClassifier(base_model)

        # Train model
        with st.spinner("Training Self-Training Classifier..."):
            self_training_model.fit(X_train, y_train_partial)

        # Predict & Evaluate
        y_pred = self_training_model.predict(X_test)

        st.markdown("### ✅ Classification Report")
        st.code(classification_report(y_test, y_pred), language="text")

        st.success(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    except Exception as e:
        st.error(f"❌ Error during modeling: {e}")
