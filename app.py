import streamlit as st
import pandas as pd

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("📊 Student Performance Predictor + AI Explainer")

with st.sidebar:
    st.header("About")
    st.write("Upload a CSV of student data to predict outcomes and get AI-generated explanations for each prediction.")
    st.markdown("---")
    st.caption("Built with Streamlit, scikit-learn, and Ollama (local LLM)")

uploaded_file = st.file_uploader("Upload student data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = None
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} student records.")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Couldn't read this file: {e}")

    if df is not None and not df.empty:
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

            # Step 4 — Predict on the full dataset + styled results table
            st.subheader("Predictions")

            df["predicted_outcome"] = model.predict(X)
            df["confidence"] = model.predict_proba(X).max(axis=1)

            def highlight_outcome(val):
                color = "#2ecc71" if val == "Pass" else "#e74c3c"
                return f"color: {color}; font-weight: bold"

            styled_df = df[["student_id", "predicted_outcome", "confidence"] + feature_cols].style.map(
                highlight_outcome, subset=["predicted_outcome"]
            )
            st.dataframe(styled_df)

            # Step 5 — Feature importance extraction (refined chart)
            st.subheader("What Drives These Predictions?")

            importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)

            import plotly.graph_objects as go

            fig = go.Figure(go.Bar(
                x=importances.values,
                y=importances.index,
                orientation="h",
                marker=dict(
                    color=importances.values,
                    colorscale=[[0, "#6EC1E4"], [1, "#1F77B4"]],
                    line=dict(width=0)
                ),
                text=[f"{v:.1%}" for v in importances.values],
                textposition="outside",
            ))

            fig.update_layout(
                title="Feature Importance (Overall Model)",
                xaxis_title="Relative Importance",
                yaxis_title=None,
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=14),
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", tickformat=".0%"),
            )

            st.plotly_chart(fig, width='stretch')

            top_features = importances.sort_values(ascending=False).index[:3].tolist()

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

            import plotly.express as px
            fig2 = px.pie(values=outcome_counts.values, names=outcome_counts.index, title="Predicted Outcome Distribution")
            st.plotly_chart(fig2, width='stretch')

        else:
            st.warning(f"CSV must contain columns: {feature_cols + [target_col]}")
else:
    st.info("Upload a CSV to get started.")