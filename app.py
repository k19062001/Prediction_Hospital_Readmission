# app.py
# Full application with separate Doctor/Admin login pages, home landing,
# session-based auth, role protection, ML prediction, PDF generation, LLM, emails, and dashboard.

import os
import io
import re
import csv
import json
import smtplib
import joblib
import datetime
import pandas as pd
import numpy as np
import requests
import traceback
import random
from flask import (
    Flask, Response, render_template, request, redirect, url_for, flash, jsonify, send_file, session
)
from email.mime.text import MIMEText
from email.message import EmailMessage
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash

# charts
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# pdf
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY

# optional simple user storage using sqlite for admin to add doctors
from flask_sqlalchemy import SQLAlchemy
class DictObj:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

app = Flask(__name__)
load_dotenv()
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key")

# -------------------------
# Database for users (optional)
# -------------------------
DB_URL = os.getenv("DATABASE_URL", "sqlite:///users.db")
app.config["SQLALCHEMY_DATABASE_URI"] = DB_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class User(db.Model):
    """Simple User model for admin-managed accounts (username unique, hashed password)"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'doctor' or 'admin'

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

# create DB and default accounts if missing
with app.app_context():
    try:
        db.create_all()
        default_admin = os.getenv("DEFAULT_ADMIN_USER", "admin")
        default_admin_pass = os.getenv("DEFAULT_ADMIN_PASS", "admin123")
        default_doctor = os.getenv("DEFAULT_DOCTOR_USER", "doctor")
        default_doctor_pass = os.getenv("DEFAULT_DOCTOR_PASS", "doctor123")

        if not User.query.filter_by(username=default_admin).first():
            u = User(username=default_admin, role="admin")
            u.set_password(default_admin_pass)
            db.session.add(u)
        if not User.query.filter_by(username=default_doctor).first():
            d = User(username=default_doctor, role="doctor")
            d.set_password(default_doctor_pass)
            db.session.add(d)
        db.session.commit()
    except Exception:
        traceback.print_exc()

# -------------------------
# Load models / encoders (strictly rely on feature_names.pkl)
# -------------------------
MODEL_DIR = "models"
model = None
scaler = None
feature_names = []
num_cols = []
le_gender = None
le_discharge = None
le_bmi = None

try:
    model = joblib.load(os.path.join(MODEL_DIR, "lightgbm_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    num_cols = joblib.load(os.path.join(MODEL_DIR, "num_cols.pkl"))
    le_gender = joblib.load(os.path.join(MODEL_DIR, "le_gender.pkl"))
    le_discharge = joblib.load(os.path.join(MODEL_DIR, "le_discharge.pkl"))
    le_bmi = joblib.load(os.path.join(MODEL_DIR, "le_bmi.pkl"))
    print("✅ Model and preprocessing artifacts loaded.")
    print("Model expects", len(feature_names), "features.")
except Exception as e:
    print("❌ Error loading model artifacts:", e)
    traceback.print_exc()

# -------------------------
# Environment / Email / LLM
# -------------------------
FROM_EMAIL = os.getenv("EMAIL_USER", "")
EMAIL_PASS = os.getenv("EMAIL_PASS", "")
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY", "")
TOGETHER_AI_ENDPOINT = os.getenv("TOGETHER_AI_ENDPOINT", "https://api.together.xyz/v1/chat/completions")
TOGETHER_AI_MODEL = os.getenv("TOGETHER_AI_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

# -------------------------
# Settings / Files
# -------------------------
COLOR_MAP = {"High": "#ff4757", "Medium": "#ffa502", "Low": "#2ed573"}

LOG_FILE = "prediction_log.csv"
FOLLOWUP_FILE = "followups.csv"

LOG_HEADER = [
    "timestamp","patient_id","doctor_name","doctor_email",
    "age","gender","cholesterol","bmi","diabetes","hypertension",
    "medication_count","length_of_stay","discharge_destination",
    "systolic_bp","diastolic_bp","pulse_pressure","high_bp_flag",
    "comorbidity_index","bmi_category","discharge_risk",
    "meds_per_day","meds_per_comorbidity",
    "probability","risk","override_applied","override_reasons","email_status",
    "summary"
]

# -------------------------
# Helpers
# -------------------------
def require_role(roles):
    """Decorator to require a role (string or list) for a view. Uses session."""
    if isinstance(roles, str):
        roles_list = [roles]
    else:
        roles_list = list(roles)
    def decorator(f):
        def wrapped(*args, **kwargs):
            if "role" not in session:
                flash("Please login to continue.", "warning")
                return redirect(url_for("root"))
            if session.get("role") not in roles_list:
                flash("Unauthorized access.", "danger")
                return redirect(url_for("root"))
            return f(*args, **kwargs)
        wrapped.__name__ = f.__name__
        return wrapped
    return decorator

def log_prediction_row(row: list):
    newfile = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        if newfile:
            w.writerow(LOG_HEADER)
        w.writerow(row)

def log_followup(patient_id, followup_date, notes,status="Pending"):
    newfile = not os.path.exists(FOLLOWUP_FILE)
    with open(FOLLOWUP_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        if newfile:
            w.writerow(["patient_id","followup_date","notes","status"])
        w.writerow([patient_id, followup_date, notes])

def safe_transform(le, value, default=0):
    try:
        if le is None:
            return default
        if isinstance(value, (np.str_, np.string_)):
            value = str(value)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return le.transform([value])[0]
    except Exception:
        return default

def adjust_probability(probability, override_flag=False):
    prob_percent = probability * 100 + 10
    if override_flag:
        prob_percent += 20
    return min(prob_percent, 95)

def classify_risk(probability, override_flag=False):
    p = adjust_probability(probability, override_flag)
    if p < 30:
        return "Low"
    elif p < 40:
        return "Medium"
    else:
        return "High"

# LLM
def call_together_llm(patient_id, doctor_name, features_dict, prob_percent, risk, audience):
    try:
        features_json = json.dumps(features_dict, ensure_ascii=False)
    except Exception:
        features_json = str(features_dict)

    if audience == "doctor":
        prompt = (
            f"You are a clinical AI assistant for hospital staff. Patient ID: {patient_id}.\n"
            f"The prediction indicates a {risk} risk of 30-day readmission with an adjusted probability of {prob_percent:.2f}%.\n"
            f"Primary patient features: {features_json}\n\n"
            "Provide a single, very concise clinical summary (one sentence) of the main risk factors. "
            "Then list 3–4 brief, actionable clinical measures as bullet points to mitigate risk. "
            "Bold key risk factors (e.g., age, comorbidity, length of stay, discharge destination) but don't use ** \n"
            "Format strictly as:\n"
            "Summary: [one sentence]\n"
            "Measures:\n"
            ". [Measure 1]\n"
            ". [Measure 2]\n"
            ". [Measure 3]\n"
        )
    else:
        prompt = (
            f"You are an AI assistant for hospital administration. Patient ID: {patient_id}.\n"
            f"The prediction indicates a {risk} risk of readmission with an adjusted probability of {prob_percent:.2f}%.\n\n"
            "Provide a single-sentence executive summary on why this case needs attention. "
            "Then list 3–4 strategic, non-clinical follow-ups (coordination, resources, outreach), as brief bullet points.\n"
            "Format strictly as:\n"
            "Summary: [one sentence]\n"
            "Measures:\n"
            ". [Measure 1]\n"
            ". [Measure 2]\n"
            ". [Measure 3]\n"
        )

    if not TOGETHER_AI_API_KEY or not TOGETHER_AI_ENDPOINT:
        return None

    payload = {
        "model": TOGETHER_AI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.15
    }
    headers = {"Authorization": f"Bearer {TOGETHER_AI_API_KEY}", "Content-Type": "application/json"}
    try:
        resp = requests.post(TOGETHER_AI_ENDPOINT, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        rj = resp.json()
        try:
            return rj["choices"][0]["message"]["content"].strip()
        except Exception:
            return rj.get("result") or rj.get("output") or json.dumps(rj)[:2000]
    except Exception as e:
        print("LLM call failed:", e)
        return None

# PDF helpers
def process_text_for_pdf(text, highlight_color="#3498db"):
    if not text:
        return ""
    t = re.sub(r'\*\*(.*?)\*\*', rf'<b><font color="{highlight_color}">\1</font></b>', text)
    return t.replace('*', '').replace('`', '')

# Highlight colors for PDF summaries
HIGHLIGHT_SUMMARY_COLOR = "#3498db"
HIGHLIGHT_MEASURES_COLOR = "#e67e22"

def generate_patient_report(patient_id, doctor_name, doctor_email, features_dict, prob_percent, risk, summary_text=""):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=36, rightMargin=36)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    
    story = []

    # Title
    story.append(Paragraph("Clinical Readmission Risk Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # Patient Info
    pt_tbl = Table([
        ["Patient ID", patient_id],
        ["Doctor", doctor_name],
        ["Doctor Email", doctor_email],
        ["Report Generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    ], colWidths=[150, 350])
    pt_tbl.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ]))
    story.append(pt_tbl)
    story.append(Spacer(1, 16))

    # Risk
    story.append(Paragraph("Predicted Risk Summary", styles["Heading2"]))
    risk_color = {"Low": colors.green, "Medium": colors.orange, "High": colors.red}.get(risk, colors.black)
    risk_tbl = Table([["Risk Level", risk], ["Adjusted Probability (%)", f"{prob_percent:.2f}%"]], colWidths=[200, 280])
    risk_tbl.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
        ("TEXTCOLOR", (1,0), (1,0), risk_color),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ]))
    story.append(risk_tbl)
    story.append(Spacer(1, 16))

    # Key Features
    story.append(Paragraph("Key Patient Features", styles["Heading2"]))
    rows = [["Feature", "Value"]]
    essential_features = ["age", "bmi", "cholesterol", "comorbidity_index", "medication_count", "length_of_stay"]
    for k in essential_features:
        if k in features_dict:
            rows.append([k.replace("_", " ").title(), str(features_dict[k])])
    feat_tbl = Table(rows, colWidths=[200, 280])
    feat_tbl.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ]))
    story.append(feat_tbl)
    story.append(Spacer(1, 16))

    # LLM summary
    story.append(Paragraph("Clinical Overview & Actionable Plan", styles["Heading2"]))
    if summary_text:
        try:
            summary_section = summary_text.split("Measures:")[0].replace("Summary:", "").strip()
            measures_section = summary_text.split("Measures:")[1].strip()
        except IndexError:
            summary_section = "Could not parse summary from LLM."
            measures_section = summary_text
        
        # Summary paragraph
        processed_summary = process_text_for_pdf(summary_section.replace("*", ""), HIGHLIGHT_SUMMARY_COLOR)
        story.append(Paragraph(f"<b>Overview:</b> {processed_summary}", styles["Justify"]))
        story.append(Spacer(1, 6))

        # Measures list
        story.append(Paragraph("<b>Actionable Plan:</b>", styles["Normal"]))
        for line in measures_section.split("\n"):
            if line.strip():
                processed_line = process_text_for_pdf(line.replace("*", "").strip(), HIGHLIGHT_MEASURES_COLOR)
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&bull;&nbsp;{processed_line}", styles["Normal"]))

    else:
        story.append(Paragraph("No AI-generated summary available.", styles["Normal"]))

    story.append(Spacer(1, 16))
    doc.build(story)
    buf.seek(0)
    return buf.read()

def generate_admin_pdf(patient_id, risk, summary_text, features_dict):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=36, rightMargin=36)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], alignment=TA_CENTER)
    section = ParagraphStyle("section", parent=styles["Heading2"])
    normal = styles["Normal"]

    story = []
    story.append(Paragraph("MedPredict AI – Administrative Alert", title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Patient ID: <b>{patient_id}</b>", normal))
    story.append(Paragraph(f"Risk Level: <b>{risk}</b>", normal))
    if features_dict and "prob_percent" in features_dict:
        story.append(Paragraph(f"Probability: <b>{features_dict['prob_percent']:.2f}%</b>", normal))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Executive Summary & Follow-up", section))
    if summary_text:
        try:
            summary_section = summary_text.split("Measures:")[0].replace("Summary:", "").strip()
            measures_section = summary_text.split("Measures:")[1].strip()
        except IndexError:
            summary_section = "Could not parse summary from LLM."
            measures_section = summary_text

        story.append(Paragraph(f"<b>Overview:</b> {process_text_for_pdf(summary_section)}", normal))
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Recommendations:</b>", normal))
        for line in measures_section.split("\n"):
            line = line.strip()
            if line:
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&bull;&nbsp;{process_text_for_pdf(line, '#e74c3c')}", normal))
    else:
        story.append(Paragraph("No AI-generated summary available.", normal))

    doc.build(story)
    buf.seek(0)
    return buf.read()
def append_log_row(row):
    newfile = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        if newfile:
            header = [
                "timestamp", "patient_id", "doctor_name", "doctor_email",
                "age", "gender", "cholesterol", "bmi", "diabetes", "hypertension",
                "medication_count", "length_of_stay", "discharge_destination",
                "systolic_bp", "diastolic_bp", "pulse_pressure", "high_bp_flag",
                "comorbidity_index", "bmi_category", "discharge_risk",
                "meds_per_day", "meds_per_comorbidity", "probability", "risk",
                "override_applied", "override_reasons", "email_status", "summary",
            ]
            w.writerow(header)
        w.writerow(row)

# Email helpers
def send_email(to_email: str, subject: str, body: str) -> str:
    if not FROM_EMAIL or not EMAIL_PASS:
        return "Email disabled: set EMAIL_USER and EMAIL_PASS in .env"
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = FROM_EMAIL
        msg["To"] = to_email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(FROM_EMAIL, EMAIL_PASS)
            server.send_message(msg)
        return "Sent"
    except Exception as e:
        return f"Failed: {e}"
def send_admin_email(patient_id, risk, summary_text, features_dict):
    if not all([SMTP_USER, SMTP_PASS, ADMIN_EMAIL, SMTP_SERVER, SMTP_PORT]):
        return "Disabled"
    try:
        admin_pdf = generate_admin_pdf(patient_id, risk, summary_text, features_dict)

        email_summary = "A new readmission risk alert has been generated."
        if summary_text:
            try:
                email_summary = summary_text.split("Measures:")[0].replace("Summary:", "").strip()
            except Exception:
                email_summary = summary_text

        msg = EmailMessage()
        msg["Subject"] = f"MedPredict AI Alert: Patient {patient_id} - {risk.upper()}"
        msg["From"] = SMTP_USER
        msg["To"] = ADMIN_EMAIL
        msg.set_content(
            f"MedPredict AI Alert\n\n"
            f"Patient ID: {patient_id}\n"
            f"Risk Level: {risk.upper()}\n"
            f"Summary: {email_summary}\n\n"
            f"Please review the attached PDF for administrative follow-up."
        )
        msg.add_attachment(admin_pdf, maintype="application", subtype="pdf",
                           filename=f"AdminAlert_{patient_id}.pdf")

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        return "Sent"
    except Exception as e:
        return f"Failed: {e}"

# Dashboard helpers
def get_dashboard_stats():
    if not os.path.exists(LOG_FILE):
        return {"total": 0, "high": 0, "medium": 0, "low": 0, "recent_predictions": []}
    df = pd.read_csv(LOG_FILE)
    return {
        "total": len(df),
        "high": int((df["risk"] == "High").sum()),
        "medium": int((df["risk"] == "Medium").sum()),
        "low": int((df["risk"] == "Low").sum()),
        "recent_predictions": df.tail(5).to_dict("records") if len(df) > 0 else [],
    }

def create_risk_distribution_chart():
    if not os.path.exists(LOG_FILE): return json.dumps({}, cls=PlotlyJSONEncoder)
    df = pd.read_csv(LOG_FILE)
    risk_counts = df["risk"].value_counts()
    labels = [r for r in ["High","Medium","Low"] if r in risk_counts]
    values = [risk_counts[r] for r in labels]
    colors = [COLOR_MAP[r] for r in labels]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4, marker=dict(colors=colors))])
    fig.update_layout(title="Risk Distribution", font=dict(size=14), showlegend=True)
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def create_age_risk_chart():
    if not os.path.exists(LOG_FILE): return json.dumps({}, cls=PlotlyJSONEncoder)
    df = pd.read_csv(LOG_FILE)
    fig = px.scatter(df, x="age", y="probability", color="risk", color_discrete_map=COLOR_MAP,
                     title="Age vs Risk Probability", labels={"probability": "Risk Probability", "age": "Age"})
    fig.update_layout(font=dict(size=12))
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def create_predictions_over_time_chart():
    if not os.path.exists(LOG_FILE): return json.dumps({}, cls=PlotlyJSONEncoder)
    df = pd.read_csv(LOG_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    recent_df = df[df["timestamp"] > (pd.Timestamp.now() - pd.Timedelta(days=30))]
    fig = px.histogram(recent_df, x="timestamp", color="risk", color_discrete_map=COLOR_MAP, nbins=30, title="Predictions Over Time")
    fig.update_layout(font=dict(size=12), barmode="stack")
    return json.dumps(fig, cls=PlotlyJSONEncoder)

# -------------------------
# Routes: Home & Logins
# -------------------------
@app.route("/",endpoint='root')
@app.route("/home")
def home():
    """Home landing page with background & buttons to doctor/admin login."""
    return render_template("home.html")


@app.route("/doctor_login", methods=["GET", "POST"])
def doctor_login():
    """Separate doctor login board."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        # First try DB users
        user = None
        try:
            user = User.query.filter_by(username=username, role="doctor").first()
        except Exception:
            traceback.print_exc()

        # If DB user exists, check hash; otherwise optionally allow default hardcoded fallback
        if user and user.check_password(password):
            session["username"] = user.username
            session["role"] = "doctor"
            flash("Logged in as Doctor.", "success")
            return redirect(url_for("predict"))
        else:
            flash("Invalid doctor credentials.", "danger")
            return redirect(url_for("doctor_login"))

    return render_template("doctor_login.html")


@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    """Separate admin login board."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = None
        try:
            user = User.query.filter_by(username=username, role="admin").first()
        except Exception:
            traceback.print_exc()

        if user and user.check_password(password):
            session["username"] = user.username
            session["role"] = "admin"
            flash("Logged in as Admin.", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid admin credentials.", "danger")
            return redirect(url_for("admin_login"))

    return render_template("admin_login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out. Returning to home.", "info")
    return redirect(url_for("home"))

def get_patient_data(patient_id):
    df = pd.read_csv("data/hospital_readmissions_30k.csv")  # your raw CSV
    patient_row = df[df['patient_id'] == patient_id]
    if patient_row.empty:
        return None
    return patient_row.iloc[0].to_dict()
@app.route('/patient_details/<patient_id>', methods=['GET'])
def patient_details(patient_id):
    patient = get_patient_data(patient_id)
    if not patient:
        return "Patient not found", 404
    return render_template('patient_details.html', patient=patient)
    # On POST: process form input and make prediction
    if request.method == 'POST':
        form_data = request.form.to_dict()
        # Call your trained model here
        # Example:
        # prediction, probability = predict_patient(form_data)
        prediction = "High Risk"  # placeholder
        probability = 0.87        # placeholder
        return render_template('patient_details.html', patient=DictObj(form_data),
                               prediction=prediction, probability=probability)
def get_patient_info(patient_id):
    """
    Fetch patient info, prioritizing prediction_log.csv over the base dataset.
    This function normalizes data from both sources to be compatible with the form.
    """
    try:
        # 1. Check the log file first for the most recent data
        if os.path.exists(LOG_FILE):
            df_log = pd.read_csv(LOG_FILE, dtype={'patient_id': str})
            patient_records = df_log[df_log["patient_id"].astype(str) == str(patient_id)]
            
            if not patient_records.empty:
                # .iloc[-1] gets the last row, which is the most recent prediction
                record = patient_records.iloc[-1].to_dict()
                record = {k.lower(): v for k, v in record.items()} # Normalize keys
                
                # Combine BP fields if they exist
                if pd.notna(record.get("systolic_bp")) and pd.notna(record.get("diastolic_bp")):
                    record["blood_pressure"] = f"{int(record['systolic_bp'])}/{int(record['diastolic_bp'])}"
                
                print(f"✅ Found patient {patient_id} in prediction_log.csv")
                return record, "log"

        # 2. If not in log, check the base dataset
        base_dataset_path = "data/hospital_readmissions_30k.csv"
        if os.path.exists(base_dataset_path):
            df_base = pd.read_csv(base_dataset_path, dtype={'patient_id': str})
            patient_row = df_base[df_base["patient_id"].astype(str) == str(patient_id)]
            
            if not patient_row.empty:
                record = patient_row.iloc[0].to_dict()
                record = {k.lower(): v for k, v in record.items()} # Normalize keys

                # Combine BP fields
                if pd.notna(record.get("systolic_bp")) and pd.notna(record.get("diastolic_bp")):
                    record["blood_pressure"] = f"{int(record['systolic_bp'])}/{int(record['diastolic_bp'])}"

                # IMPORTANT: Normalize 'discharge_destination' value
                if 'discharge_destination' in record:
                    record['discharge_destination'] = str(record['discharge_destination']).replace(' ', '_')
                
                print(f"✅ Found patient {patient_id} in hospital_readmissions_30k.csv")
                return record, "base"

    except Exception as e:
        print(f"❌ Error loading patient info for ID {patient_id}: {e}")
        traceback.print_exc()

    print(f"⚠️ Patient {patient_id} not found in any data source.")
    return None, None
  
# Predict route (doctor only)
# -------------------------
@app.route("/predict", methods=["GET","POST"])
@require_role("doctor")
def predict():
    if request.method == "GET":
        # 🔹 Handle GET → prefill form
        patient_id = request.args.get("patient_id", "").strip()
        patient_data = None
        if patient_id:
            row, source = get_patient_info(patient_id)
            if row:
                patient_data = DictObj(row)
        return render_template("predict.html", patient=patient_data)    
    if request.method == "POST":
        try:
            form = request.form.to_dict()
            # Read form fields
            patient_id = form.get("patient_id", "").strip() or f"PID_{int(datetime.datetime.now().timestamp())}"
            age = int(form.get("age", 0))
            gender = form.get("gender", "")
            bp = form.get("blood_pressure", "0/0")
            cholesterol = float(form.get("cholesterol", 0.0))
            bmi = float(form.get("bmi", 0.0))
            diabetes = 1 if form.get("diabetes", "No") == "Yes" else 0
            hypertension = 1 if form.get("hypertension", "No") == "Yes" else 0
            medication_count = int(form.get("medication_count", 0))
            length_of_stay = int(form.get("length_of_stay", 1))
            discharge = form.get("discharge_destination", "Home").replace(" ", "_")
            doctor_name = form.get("doctor_name", session.get("username", "Unknown"))
            doctor_email = form.get("doctor_email", "")

            # Parse BP
            try:
                systolic_bp, diastolic_bp = map(float, bp.split("/"))
            except Exception:
                flash("⚠️ Invalid BP format, use systolic/diastolic", "danger")
                return redirect(url_for("predict"))

            # Base features for model input (safe transforms)
            data = {
                "age": age,
                "gender": safe_transform(le_gender, gender),
                "cholesterol": cholesterol,
                "bmi": bmi,
                "diabetes": diabetes,
                "hypertension": hypertension,
                "medication_count": medication_count,
                "length_of_stay": length_of_stay,
                "discharge_destination": safe_transform(le_discharge, discharge),
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
            }

            # Engineered fields for reporting only
            data["pulse_pressure"] = systolic_bp - diastolic_bp
            data["high_bp_flag"] = 1 if (systolic_bp > 140 or diastolic_bp > 90) else 0
            data["comorbidity_index"] = diabetes + hypertension

            # BMI category safe transform
            if bmi < 18.5:
                bmi_cat = "Underweight"
            elif bmi < 25:
                bmi_cat = "Normal"
            elif bmi < 30:
                bmi_cat = "Overweight"
            else:
                bmi_cat = "Obese"
            data["bmi_category"] = safe_transform(le_bmi, bmi_cat)

            data["discharge_risk"] = data["discharge_destination"]
            data["meds_per_day"] = medication_count / (length_of_stay + 1)
            data["meds_per_comorbidity"] = medication_count / (data["comorbidity_index"] + 1)

            # Build input strictly from feature_names.pkl
            input_df = pd.DataFrame([data])
            input_df = input_df.reindex(columns=feature_names, fill_value=0)

            # Only scale numeric columns that exist
            try:
                if scaler is not None and isinstance(num_cols, (list, tuple)):
                    numeric_to_scale = [c for c in num_cols if c in input_df.columns]
                    if numeric_to_scale:
                        input_df[numeric_to_scale] = scaler.transform(input_df[numeric_to_scale])
                else:
                    print("⚠️ Skipping scaler.transform: scaler missing or num_cols mismatch.")
            except Exception as e:
                print("⚠️ scaler.transform failed:", e)
                traceback.print_exc()

            # Predict
            try:
                probas = model.predict_proba(input_df)[0]
                if probas.shape[0] == 2:
                    prediction_proba = float(probas[1])  # binary classification
                    risk_level = "High" if prediction_proba > 0.5 else "Low"
                else:
                    # multiclass: map to risk buckets
                    risk_idx = probas.argmax()
                    risk_map = {0: "Low", 1: "Medium", 2: "High"}
                    prediction_proba = float(probas[risk_idx])
                    risk_level = risk_map.get(risk_idx, "Unknown")
            except Exception as e:
                print("❌ model.predict_proba error:", e)
                print("Model type:", type(model))
                print("feature_names length:", len(feature_names))
                print("input_df shape:", input_df.shape)
                print("input_df columns:", input_df.columns.tolist())
                raise
            # Overrides
            override_flag, reasons = False, []
            if age >= 85:
                override_flag = True; reasons.append("Age ≥ 85")
            if age >= 75 and diabetes:
                override_flag = True; reasons.append("Elderly with diabetes")
            if data["comorbidity_index"] >= 2 and age >= 65:
                override_flag = True; reasons.append("Multiple comorbidities")
            if bmi >= 35 and diabetes:
                override_flag = True; reasons.append("Severe obesity with diabetes")
            if discharge in ["Nursing_Facility", "Rehab"]:
                override_flag = True; reasons.append("Discharge to nursing/rehab facility")

            adjusted_percent = adjust_probability(prediction_proba, override_flag)
            risk_text = classify_risk(prediction_proba, override_flag)

            # LLM summaries (non-blocking: may return None)
            display_features = {
                "age": age, "bmi": bmi, "cholesterol": cholesterol,
                "medication_count": medication_count, "length_of_stay": length_of_stay,
                "diabetes": "Yes" if diabetes else "No",
                "hypertension": "Yes" if hypertension else "No",
                "pulse_pressure": round(data["pulse_pressure"], 2)
            }
            doctor_summary_text = call_together_llm(patient_id, doctor_name, display_features, adjusted_percent, risk_text, "doctor")
            admin_summary_text  = call_together_llm(patient_id, doctor_name, {"risk": risk_text, **display_features}, adjusted_percent, risk_text, "admin")

            # Emails
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            email_status = "Skipped"
            if doctor_email and risk_text == "High":
                email_body = (
                    f"Patient ID: {patient_id}\n"
                    f"Adjusted Probability: {adjusted_percent:.2f}%\n"
                    f"Risk Level: {risk_text}\n"
                    f"Override Applied: {override_flag}\n"
                    f"Reasons: {', '.join(reasons) if reasons else 'None'}\n\n"
                    f"Clinical Summary:\n{doctor_summary_text or 'N/A'}\n"
                )
                email_status = send_email(doctor_email, f"Patient {patient_id} Risk Report", email_body)

            admin_send_status = "Skipped"
            if risk_text == "High":
                admin_features = {"risk": risk_text, "prob_percent": adjusted_percent}
                admin_send_status = send_admin_email(patient_id, risk_text, admin_summary_text, admin_features)

            # Log prediction
            log_row = [
                timestamp, patient_id, doctor_name, doctor_email,
                age, gender, cholesterol, bmi, diabetes, hypertension,
                medication_count, length_of_stay, discharge,
                systolic_bp, diastolic_bp, data["pulse_pressure"], data["high_bp_flag"],
                data["comorbidity_index"], data["bmi_category"], data["discharge_risk"],
                round(data["meds_per_day"],6), round(data["meds_per_comorbidity"],6),
                round(adjusted_percent, 6), risk_text, int(override_flag), "|".join(reasons),
                f"Doctor:{email_status}; Admin:{admin_send_status}", (doctor_summary_text or "")
            ]
            log_prediction_row(log_row)

            # Generate patient PDF in memory (not auto-sent)
            pdf_features = {
                "age": age, "bmi": bmi, "cholesterol": cholesterol,
                "diabetes": diabetes, "hypertension": hypertension,
                "length_of_stay": length_of_stay, "pulse_pressure": data["pulse_pressure"]
            }
            _ = generate_patient_report(patient_id, doctor_name, doctor_email, pdf_features, adjusted_percent, risk_text, doctor_summary_text or "")

            # Auto follow-up for High risk
            if risk_text == "High":
                try:
                    log_followup(patient_id, datetime.datetime.now().strftime("%Y-%m-%d"), "Auto follow-up scheduled due to High Risk")
                except Exception as e:
                    print("⚠️ Could not write followup:", e)

            return render_template(
                "result.html",
                patient_id=patient_id,
                probability=round(adjusted_percent, 2),
                risk=risk_text,
                override_flag=override_flag,
                reasons=reasons,
                timestamp=timestamp,
                summary=(doctor_summary_text or ""),
                doctor_name=doctor_name,
                doctor_email=doctor_email
            )

        except Exception as e:
            print("❌ ERROR in /predict:", e)
            traceback.print_exc()
            flash(f"Error during prediction: {e}", "danger")
            return redirect(url_for("predict"))

    return render_template("predict.html",patient=DictObj(form_data),prediction=prediction)
def get_all_patient_records():
    """Return all patient records from the prediction log CSV."""
    if not os.path.exists(LOG_FILE):
        return []
    df = pd.read_csv(LOG_FILE)
    return df.to_dict("records")
def calculate_dashboard_stats():
    patients = get_all_patient_records()
    total = len(patients)
    high = len([p for p in patients if p.get("risk") == "High"])
    medium = len([p for p in patients if p.get("risk") == "Medium"])
    low = len([p for p in patients if p.get("risk") == "Low"])
    avg_age = sum([p.get("age", 0) for p in patients if p.get("age") is not None])/total if total else 0
    recent_predictions = sorted(patients, key=lambda x: x.get("timestamp", ""), reverse=True)[:5]
    return {
        "total_predictions": total,
        "high_risk_count": high,
        "medium_risk_count": medium,
        "low_risk_count": low,
        "high_risk_percentage": (high/total)*100 if total else 0,
        "avg_age": avg_age,
        "recent_predictions": recent_predictions
    }
# -----------------------------
@app.route("/export_risk")
def export_risk_csv():
    records = get_all_patient_records()
    si = io.StringIO()
    cw = csv.writer(si)

    cw.writerow(["Patient ID", "Age", "Risk", "Probability", "Doctor", "Timestamp"])
    for r in records:
        cw.writerow([
    r.get("patient_id", r.get("id", "")),
    r.get("age", ""),
    r.get("risk", ""),
    r.get("probability", ""),
    r.get("doctor_name", ""),
    r.get("timestamp", "")
])

    output = io.BytesIO()
    output.write(si.getvalue().encode("utf-8"))
    output.seek(0)
    return send_file(output, mimetype="text/csv", download_name="patients.csv", as_attachment=True)

# -----------------------------
# Followups & patients pages
# -------------------------
@app.route("/update_followup/<patient_id>", methods=["GET","POST"])
@require_role("doctor")
def update_followup(patient_id):
    import pandas as pd
    df = pd.read_csv(FOLLOWUP_FILE)
    if "status" not in df.columns:
        df["status"] = "Pending"
    # Update status for the correct row
    row_id = int(request.form.get("row_id", -1))  # Get row_id from form data, default to -1 if missing
    new_status = request.form.get("new_status", "Pending")  # Get new_status from form data

    if 0 <= row_id < len(df):
        df.loc[row_id, "status"] = new_status
        df.to_csv(FOLLOWUP_FILE, index=False)

        flash(f"Follow-up for patient {patient_id} marked as {new_status}", "success")
    else:
        flash("Invalid follow-up update request.", "danger")
    return redirect(url_for("followups"))
@app.route("/followups")
@require_role(["admin", "doctor"])
def followups():
    records = []
    if os.path.exists(FOLLOWUP_FILE):
        import pandas as pd
        from datetime import date
        df = pd.read_csv(FOLLOWUP_FILE)
        if "status" not in df.columns:
            df["status"] = "Pending"

        today = datetime.date.today()

        # 👉 Mark overdue followups
        df.loc[
            (pd.to_datetime(df["followup_date"]).dt.date < today) &
            (df["status"] == "Pending"),
            "status"
        ] = "Overdue"

        df.to_csv(FOLLOWUP_FILE, index=False, quoting=csv.QUOTE_ALL)
        records = df.to_dict(orient="records")
    return render_template("followups.html", records=records)
# app.py

...

@app.route("/patients")
@require_role(["admin", "doctor"])
def patients():
    log_file = "prediction_log.csv"
    patient_list = [] # Use a different name to avoid conflict with the function name

    if os.path.exists(log_file):
        try:
            df = pd.read_csv(log_file, dtype=str)
            # Normalize column names to lowercase for consistent access
            df.columns = [c.lower() for c in df.columns]

            # Sort by timestamp descending to show most recent first
            df = df.sort_values(by="timestamp", ascending=False)

            for _, row in df.iterrows():
                patient_list.append({
                    "patient_id": row.get("patient_id", "N/A"),
                    "age": row.get("age", "N/A"),
                    "risk": row.get("risk", "None"),
                    # Ensure probability is a float for formatting in the template
                    "probability": float(row.get("probability", 0.0)),
                    "doctor": row.get("doctor_name", "N/A"), # Corrected key from "doctor" to "doctor_name"
                    "timestamp": row.get("timestamp", "N/A")  # Corrected key for the template
                })
        except pd.errors.EmptyDataError:
            print("Warning: prediction_log.csv is empty.")
        except Exception as e:
            print(f"Error reading or processing prediction_log.csv: {e}")

    # Pass the list with the key 'patients' as expected by the corrected template
    return render_template("patients.html", patients=patient_list)
# -------------------------
# Dashboard & analytics
# -------------------------
@app.route("/dashboard")
@require_role("admin")
def dashboard():
    # A doctor accessing /dashboard will be handled by the template logic
    if session.get('user_role') == 'doctor':
        # Can add doctor specific dashboard logic here if needed in future
        pass

    stats = { "total_predictions": 0, "high_risk_count": 0, "medium_risk_count": 0, "low_risk_count": 0,"avg_age":0,"recent_predictions": [] }
    if not os.path.exists(LOG_FILE):
        return render_template("dashboard.html", stats=stats)
    
    try:
        df = pd.read_csv(LOG_FILE, dtype={'patient_id': str})
    except pd.errors.EmptyDataError:
        return render_template("dashboard.html", stats=stats)
    except Exception as e:
        flash(f"Error reading log file: {e}", "danger")
        return render_template("dashboard.html", stats=stats)

    # Filter for doctor's own patients if they are logged in
    if session.get('user_role') == 'doctor':
        df = df[df['doctor_name'] == session.get('user_name')].copy()

    if not df.empty:
        stats.update({
            "total_predictions": len(df),
            "high_risk_count": int((df["risk"] == "High").sum()),
            "medium_risk_count": int((df["risk"] == "Medium").sum()),
            "low_risk_count": int((df["risk"] == "Low").sum()),
            "avg_age": df["age"].mean() if "age" in df.columns and not df.empty else 0,
            "recent_predictions": df.sort_values('timestamp', ascending=False).head(5).to_dict("records")
        })
    return render_template("dashboard.html", stats=stats)
@app.route("/analytics")
@require_role(["admin", "doctor"])
def analytics():
    return render_template("analytics.html",
                           risk_chart=create_risk_distribution_chart(),
                           age_risk_chart=create_age_risk_chart(),
                           timeline_chart=create_predictions_over_time_chart())

# -------------------------
# Download PDF (admin/doctor)
# -------------------------
@app.route("/download_pdf/<patient_id>/<timestamp>")
@require_role(["admin", "doctor"])
def download_pdf(patient_id, timestamp):
    if not os.path.exists(LOG_FILE):
        flash("No logs found.", "danger")
        return redirect(url_for("dashboard"))

    df = pd.read_csv(LOG_FILE)
    record = df[(df["patient_id"] == patient_id) & (df["timestamp"] == timestamp)]
    if record.empty:
        flash("Record not found.", "danger")
        return redirect(url_for("dashboard"))

    row = record.iloc[0].to_dict()
    features = {
        "age": row.get("age"),
        "bmi": row.get("bmi"),
        "cholesterol": row.get("cholesterol"),
        "diabetes": row.get("diabetes"),
        "hypertension": row.get("hypertension"),
        "length_of_stay": row.get("length_of_stay"),
        "pulse_pressure": row.get("pulse_pressure"),
    }

    pdf_bytes = generate_patient_report(
        patient_id=row["patient_id"],
        doctor_name=row["doctor_name"],
        doctor_email=row["doctor_email"],
        features_dict=features,
        prob_percent=row["probability"],
        risk=row["risk"],
        summary_text=row.get("summary", "")
    )

    return send_file(
        io.BytesIO(pdf_bytes),
        as_attachment=True,
        download_name=f"PatientReport_{patient_id}_{timestamp}.pdf",
        mimetype="application/pdf"
    )

# -------------------------
# Admin: manage users (optional)
# -------------------------
@app.route("/admin/manage_users", methods=["GET", "POST"])
@require_role("admin")
def manage_users():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        role = request.form.get("role", "doctor")
        if not username or not password:
            flash("Username and password required.", "warning")
            return redirect(url_for("manage_users"))
        existing = User.query.filter_by(username=username).first()
        if existing:
            flash("Username already exists.", "warning")
            return redirect(url_for("manage_users"))
        new_user = User(username=username, role=role)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash(f"User '{username}' created.", "success")
        return redirect(url_for("manage_users"))
    users = User.query.order_by(User.id.asc()).all()
    return render_template("manage_users.html", users=users)

@app.route("/admin/delete_user/<int:user_id>", methods=["POST","GET"])
@require_role("admin")
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.role == "admin":
        flash("Cannot delete an admin account.", "danger")
        return redirect(url_for("manage_users"))
    db.session.delete(user)
    db.session.commit()
    flash(f"User {user.username} deleted.", "success")
    return redirect(url_for("manage_users"))

# -------------------------
# JSON APIs used by frontend charts (optional)
@app.route("/api/risk-distribution-summary")
@require_role(["admin", "doctor"])
def api_risk_distribution_summary():
    if not os.path.exists(LOG_FILE):
        return jsonify({"data": []})
    try:
        df = pd.read_csv(LOG_FILE)
        risk_counts = df["risk"].value_counts()
        
        # Ensure order is always High, Medium, Low for consistency
        labels = [r for r in ["High", "Medium", "Low"] if r in risk_counts.index]
        values = [int(risk_counts[label]) for label in labels]

        # The JS expects data in a list of traces
        trace = [{"labels": labels, "values": values}]
        return jsonify({"data": trace})

    except Exception as e:
        print(f"Error in /api/risk-distribution-summary: {e}")
        return jsonify({"data": []}), 500

# -------------------------
@app.route("/api/dashboard/risk_distribution", methods=["GET"])
@require_role(["admin", "doctor"])
def get_risk_distribution():
    # Load data from the prediction log
    try:
        df = pd.read_csv("prediction_log.csv")
    except FileNotFoundError:
        return jsonify({"data": []})

    # Ensure timestamp column is datetime and handle potential missing values
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.index = df.index.tz_localize(None) # Remove timezone info for consistency

    # Filter by date range if provided
    from_date_str = request.args.get("from")
    to_date_str = request.args.get("to")
    if from_date_str:
        df = df[df.index >= pd.to_datetime(from_date_str)]
    if to_date_str:
        df = df[df.index <= pd.to_datetime(to_date_str)]

    # Group and count risk levels per day
    grouped = df.groupby([df.index.date, "risk"]).size().unstack(fill_value=0)

    # Prepare data for the plot
    COLOR_MAP = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#198754"}
    traces = []
    # Loop through all possible risk labels, not just those in the current data
    for label in ["High", "Medium", "Low"]:
        if label in grouped.columns:
            traces.append({
                "x": grouped.index.astype(str).tolist(),
                "y": grouped[label].tolist(),
                "type": "bar",
                "name": label,
                "marker": {"color": COLOR_MAP[label]}
            })

    return jsonify({"data": traces})

def get_timeline_data():
    # Example timeline data: replace with DB query
    return [
        {"date": "2025-09-01", "High": 5, "Medium": 2, "Low": 3},
        {"date": "2025-09-02", "High": 3, "Medium": 4, "Low": 6},
        {"date": "2025-09-03", "High": 4, "Medium": 1, "Low": 5},
    ]

@app.route("/export_data")
def export_data():
    # Replace this with your actual patient records retrieval
    records = get_all_patient_records()  # Implement this function properly

    if not records:
        return "No patient records available", 404

    # Create CSV in memory
    si = io.StringIO()
    writer = csv.writer(si)

    # Write header
    headers = list(records[0].keys())
    writer.writerow(headers)

    # Write rows
    for row in records:
        writer.writerow([row.get(h, "") for h in headers])

    output = io.BytesIO()
    output.write(si.getvalue().encode("utf-8"))
    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        download_name="patient_records.csv",
        as_attachment=True
    )


@app.route("/export_timeline_csv")
def export_timeline_csv():
    # Replace this with your actual timeline data retrieval
    timeline_data = get_timeline_data()  # Implement this function

    if not timeline_data:
        return "No timeline data available", 404

    # Create CSV in memory
    si = io.StringIO()
    writer = csv.writer(si)

    # Write header
    headers = list(timeline_data[0].keys())
    writer.writerow(headers)

    # Write data
    for row in timeline_data:
        writer.writerow([row.get(h, "") for h in headers])

    output = io.BytesIO()
    output.write(si.getvalue().encode("utf-8"))
    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        download_name="timeline_data.csv",
        as_attachment=True
    )

@app.route("/api/age-risk")
@require_role(["admin", "doctor"])
def api_age_risk():
    if not os.path.exists(LOG_FILE):
        return jsonify({"data": []})
    df = pd.read_csv(LOG_FILE)
    data = [{
        "x": df["age"].tolist(),
        "y": df["probability"].tolist(),
        "mode": "markers",
        "type": "scatter",
        "name": "Age vs Risk"
    }]
    return jsonify({"data": data})

@app.route("/api/predictions-timeline")
@require_role(["admin", "doctor"])
def api_predictions_timeline():
    if not os.path.exists(LOG_FILE):
        return jsonify({"data": []})
    df = pd.read_csv(LOG_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d_%H-%M-%S", errors="coerce")
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    grouped = df.groupby(["date", "risk"]).size().unstack(fill_value=0)
    traces = []
    for label in ["High","Medium","Low"]:
        if label in grouped.columns:
            traces.append({
                "x": grouped.index.tolist(),
                "y": grouped[label].tolist(),
                "type": "bar",
                "name": label,
                "marker": {"color": COLOR_MAP[label]}
            })
    return jsonify({"data": traces})

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # Ensure users DB exists and default users present
    with app.app_context():
        try:
            db.create_all()
            default_admin = os.getenv("DEFAULT_ADMIN_USER", "admin")
            default_admin_pass = os.getenv("DEFAULT_ADMIN_PASS", "admin123")
            default_doctor = os.getenv("DEFAULT_DOCTOR_USER", "doctor")
            default_doctor_pass = os.getenv("DEFAULT_DOCTOR_PASS", "doctor123")

            if not User.query.filter_by(username=default_admin).first():
                u = User(username=default_admin, role="admin")
                u.set_password(default_admin_pass)
                db.session.add(u)
            if not User.query.filter_by(username=default_doctor).first():
                d = User(username=default_doctor, role="doctor")
                d.set_password(default_doctor_pass)
                db.session.add(d)
            db.session.commit()
        except Exception:
            traceback.print_exc()

    app.run(host="0.0.0.0",port=int(os.getenv("PORT",8000)))

