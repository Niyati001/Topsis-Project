from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import re
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

SENDER_EMAIL    = os.environ.get('SENDER_EMAIL', '')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')


def send_email_background(to_email, csv_bytes):
    """Runs in background thread — never blocks or crashes the main response."""
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("No email credentials set — skipping email.")
        return
    try:
        msg = MIMEMultipart()
        msg['From']    = SENDER_EMAIL
        msg['To']      = to_email
        msg['Subject'] = 'Your TOPSIS Analysis Results'
        msg.attach(MIMEText("""Hello,

Your TOPSIS analysis is complete. Results are attached as a CSV file.

Columns added:
- Topsis Score: 0 to 1 (higher = better)
- Rank: 1 = best alternative

Thank you for using TOPSIS Decision Maker.
""", 'plain'))

        part = MIMEBase('application', 'octet-stream')
        part.set_payload(csv_bytes)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="topsis_result.csv"')
        msg.attach(part)

        # Port 465 with SSL (works on Render free tier)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        print(f"Email sent to {to_email}")

    except Exception as e:
        print(f"Email failed: {e}")


def validate_inputs(data, weights, impacts):
    if data.shape[1] < 3:
        raise ValueError("Input file must contain at least 3 columns")
    matrix = data.iloc[:, 1:]
    if not all(np.issubdtype(dtype, np.number) for dtype in matrix.dtypes):
        raise ValueError("All columns except the first must contain numeric values only")
    if len(weights) != matrix.shape[1] or len(impacts) != matrix.shape[1]:
        raise ValueError(
            f"Number of weights ({len(weights)}) and impacts ({len(impacts)}) "
            f"must equal number of criteria columns ({matrix.shape[1]})"
        )
    if not all(i in ['+', '-'] for i in impacts):
        raise ValueError("Impacts must be '+' or '-' only")
    if not all(w > 0 for w in weights):
        raise ValueError("All weights must be positive numbers")


def topsis(data, weights, impacts):
    validate_inputs(data, weights, impacts)
    matrix = data.iloc[:, 1:].astype(float)

    norm     = matrix / np.sqrt((matrix ** 2).sum())
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

    d_best  = np.sqrt(((weighted - ideal_best)  ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    scores  = d_worst / (d_best + d_worst)

    result = data.copy()
    result['Topsis Score'] = scores.round(6)
    result['Rank'] = scores.rank(ascending=False, method='max').astype(int)
    return result


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "service": "TOPSIS Decision Analysis API",
        "status":  "running",
        "usage":   "POST /topsis with: file (CSV), weights, impacts, email"
    })


@app.route('/topsis', methods=['POST', 'OPTIONS'])
def run_topsis():
    if request.method == 'OPTIONS':
        return '', 200

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a .csv'}), 400

    weights_str = request.form.get('weights', '').strip()
    impacts_str = request.form.get('impacts', '').strip()
    email       = request.form.get('email',   '').strip()

    if not email or not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
        return jsonify({'error': 'Invalid or missing email address'}), 400

    try:
        weights = [float(w.strip()) for w in weights_str.split(',')]
    except ValueError:
        return jsonify({'error': 'Weights must be numeric values separated by commas'}), 400

    impacts = [i.strip() for i in impacts_str.split(',')]

    if len(weights) != len(impacts):
        return jsonify({'error': f'Weights ({len(weights)}) and impacts ({len(impacts)}) count must match'}), 400

    try:
        data = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400

    try:
        result = topsis(data, weights, impacts)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

    output = io.StringIO()
    result.to_csv(output, index=False)
    csv_bytes = output.getvalue().encode()

    # Send email in background — never blocks or crashes the response
    thread = threading.Thread(target=send_email_background, args=(email, csv_bytes))
    thread.daemon = True
    thread.start()

    return send_file(
        io.BytesIO(csv_bytes),
        mimetype='text/csv',
        as_attachment=True,
        download_name='topsis_result.csv'
    )


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
