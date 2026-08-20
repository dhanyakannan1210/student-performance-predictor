# Student Performance Predictor + AI Explainer

**Team 14 — Track 5: Machine Learning–Based Applications**
**Problem Statement 5.2**

## Objective
An app that predicts student outcomes (Pass/Fail) from structured academic data using a trained machine learning model, and explains each prediction in plain language using a locally-run LLM — turning a black-box model into something a non-technical teacher can trust and act on.

## Features
- CSV upload with validation and error handling
- Random Forest classifier trained live on uploaded data, with accuracy displayed
- Per-student predictions with confidence scores
- Feature importance visualization (what drives predictions overall)
- AI-generated plain-language explanation per student, via local LLM (Ollama)
- Class-level outcome distribution overview

## Tech Stack
- Python, pandas, scikit-learn
- Streamlit (UI)
- Plotly (visualizations)
- Ollama running Phi-4-mini / Llama 3.2 3B (local LLM, no API key, no cloud)

## Setup & Run Instructions

1. Clone this repo:https://github.com/dhanyakannan1210/student-performance-predictor.git 
cd student-performance-predictor

2. Create and activate a virtual environment:python3 -m venv venv
source venv/bin/activate

3. Install dependencies:pip install streamlit pandas scikit-learn plotly ollama

4. Install Ollama and pull the model:ollama pull phi4-mini

5. Run the app:
streamlit run app.py

6. Upload the included `sample_students_large.csv` to try it out.

## CSV Format Required
Columns: `student_id, attendance_pct, avg_marks, study_hours_per_week, assignments_submitted, previous_grade, outcome`

## Team Members & Roles
- [Roshan] — [LLM integration + prompt engineering]
- [Dhanya shree] — [UI/UX + Streamlit]
- [Helina ivanovic] — [ML model]
- [Mridhula ] — [PPT,github]
## AI Tools Used
- Ollama (Phi-4-mini / Llama 3.2 3B) for generating plain-language prediction explanations
- Claude used for code development and debugging assistance during the hackathon
## Notes
- All processing happens locally — no data leaves the machine, no external API calls
- Designed and tested within the 7-hour hackathon window as part of Hack & Fest, SRM Institute of Science and Technology