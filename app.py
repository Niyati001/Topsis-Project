from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from email.message import EmailMessage
import pandas as pd
import numpy as np
import io
import re
import os
import smtplib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ─── SMTP CONFIG ──────────────────────────────────────────────
SMTP_HOST = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
SMTP_USERNAME = os.environ.get('SMTP_USERNAME', '')
SMTP_APP_PASSWORD = os.environ.get('SMTP_APP_PASSWORD', '').replace(' ', '')
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', SMTP_USERNAME)


# ─── EMAIL FUNCTION ───────────────────────────────────────────
def send_result_email(to_email, csv_bytes, result):
    try:
        # Prepare preview
        top_results = result.sort_values(by="Rank").head(3)

        if top_results.empty:
            preview_html = "<p>No results available</p>"
            top_choice_name = "N/A"
        else:
            top_choice = top_results.iloc[0]
            top_choice_name = top_choice.iloc[0]  # first column value
            preview_html = top_results.drop(columns=["Topsis Score"]).to_html(index=False)

        # Build email
        message = EmailMessage()
        message['From'] = f"TOPSIS Analyzer <{SENDER_EMAIL}>"
        message['To'] = to_email
        message['Subject'] = "📊 Your TOPSIS Results Are Ready"

        message.set_content(
            "Your TOPSIS analysis is complete. See attachment for full results."
        )

        message.add_alternative(f"""
        <div style="font-family: Arial, sans-serif; max-width:600px; margin:auto; padding:20px;">
            <h2>📊 Your TOPSIS Results</h2>

            <p>Your decision analysis has been successfully completed.</p>

            <p><strong>Top-ranked alternatives:</strong></p>

            {preview_html}

            <p><strong>🏆 Top Choice:</strong> {top_choice_name}</p>

            <p>The complete results are attached as a CSV file.</p>
        </div>
        """, subtype='html')

        # Attach CSV
        message.add_attachment(
            csv_bytes,
            maintype='text',
            subtype='csv',
            filename='topsis_result.csv',
        )

        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_APP_PASSWORD)
            server.send_message(message)

        print("Email sent to:", to_email)
        return "ok"

    except Exception as e:
        print(f"SMTP EMAIL EXCEPTION: {type(e).__name__}: {e}")
        return f"error: {e}"


# ─── VALIDATION ───────────────────────────────────────────────
def validate_inputs(data, weights, impacts):
    if data.shape[1] < 3:
        raise ValueError("Input file must contain at least 3 columns")

    matrix = data.iloc[:, 1:]

    if not all(np.issubdtype(dtype, np.number) for dtype in matrix.dtypes):
        raise ValueError("All columns except the first must contain numeric values only")

    if len(weights) != matrix.shape[1] or len(impacts) != matrix.shape[1]:
        raise ValueError("Weights and impacts must match number of criteria columns")

    if not all(i in ['+', '-'] for i in impacts):
        raise ValueError("Impacts must be '+' or '-' only")

    if not all(w > 0 for w in weights):
        raise ValueError("Weights must be positive")


# ─── TOPSIS ───────────────────────────────────────────────────
def topsis(data, weights, impacts):
    validate_inputs(data, weights, impacts)

    matrix = data.iloc[:, 1:].astype(float)

    norm = matrix / np.sqrt((matrix ** 2).sum())
    weighted = norm * weights

    ideal_best, ideal_worst = [], []

    for i, impact in enumerate(impacts):
        col = weighted.iloc[:, i]
        if impact == '+':
            ideal_best.append(col.max())
            ideal_worst.append(col.min())
        else:
            ideal_best.append(col.min())
            ideal_worst.append(col.max())

    d_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    scores = d_worst / (d_best + d_worst)

    result = data.copy()
    result['Topsis Score'] = scores.round(6)
    result['Rank'] = scores.rank(ascending=False, method='max').astype(int)

    return result


# ─── ROUTES ───────────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({"status": "running"})


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


@app.route('/topsis', methods=['POST'])
def run_topsis():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    weights_str = request.form.get('weights', '').strip()
    impacts_str = request.form.get('impacts', '').strip()
    email = request.form.get('email', '').strip()

    if not email or not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
        return jsonify({'error': 'Invalid email'}), 400

    try:
        weights = [float(w.strip()) for w in weights_str.split(',')]
        impacts = [i.strip() for i in impacts_str.split(',')]

        data = pd.read_csv(file)

        result = topsis(data, weights, impacts)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Convert to CSV
    output = io.StringIO()
    result.to_csv(output, index=False)
    csv_bytes = output.getvalue().encode()

    # ✅ Send email (NO THREAD)
    send_result_email(email, csv_bytes, result)

    # Return file download
    return send_file(
        io.BytesIO(csv_bytes),
        mimetype='text/csv',
        as_attachment=True,
        download_name='topsis_result.csv'
    )


if __name__ == '__main__':
    app.run(debug=True)