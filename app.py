import streamlit as st
import pandas as pd

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("📊 Student Performance Predictor + AI Explainer")

uploaded_file = st.file_uploader("Upload student data (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} student records.")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Couldn't read this file: {e}")

    st.subheader("Model Training")

    feature_cols = ["attendance_pct", "avg_marks", "study_hours_per_week", "assignments_submitted", "previous_grade"]
    target_col = "outcome"

    if all(col in df.columns for col in feature_cols + [target_col]):
        from sklearn.model_selection import train_test_split

        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"Training on {len(X_train)} records, testing on {len(X_test)}.")

        # Step 3 — Train the model + show accuracy
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.metric("Model Accuracy", f"{acc*100:.1f}%")

        # Step 4 — Predict on the full dataset + results table
        st.subheader("Predictions")

        df["predicted_outcome"] = model.predict(X)
        df["confidence"] = model.predict_proba(X).max(axis=1)

        st.dataframe(df[["student_id", "predicted_outcome", "confidence"] + feature_cols])

        # Step 5 — Feature importance extraction
        st.subheader("What Drives These Predictions?")

        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

        import plotly.express as px
        fig = px.bar(importances, orientation="h", title="Feature Importance (Overall Model)")
        st.plotly_chart(fig, use_container_width=True)

        top_features = importances.index[:3].tolist()

        # Step 6 — LLM explanation layer
        st.subheader("Individual Student Explanation")

        selected_id = st.selectbox("Select a student", df["student_id"])
        student_row = df[df["student_id"] == selected_id].iloc[0]

        if st.button("Generate AI Explanation"):
            import ollama

            prompt = f"""A student has these stats:
Attendance: {student_row['attendance_pct']}%
Average marks: {student_row['avg_marks']}
Study hours/week: {student_row['study_hours_per_week']}
Assignments submitted: {student_row['assignments_submitted']}
Previous grade: {student_row['previous_grade']}

A machine learning model predicted this student's outcome as: {student_row['predicted_outcome']}
(confidence: {student_row['confidence']:.0%})

The top factors the model weighs most heavily overall are: {', '.join(top_features)}.

Explain in 2-3 plain-language sentences, for a non-technical teacher, why this student likely got this prediction. Be specific to their numbers, not generic."""

            with st.spinner("Generating explanation..."):
                response = ollama.chat(model="phi4-mini", messages=[{"role": "user", "content": prompt}])
                explanation = response["message"]["content"]

            st.write(explanation)

        # Step 7 — Class-level overview chart
        st.subheader("Class Overview")

        outcome_counts = df["predicted_outcome"].value_counts()
        fig2 = px.pie(values=outcome_counts.values, names=outcome_counts.index, title="Predicted Outcome Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning(f"CSV must contain columns: {feature_cols + [target_col]}")
else:
    st.info("Upload a CSV to get started.")