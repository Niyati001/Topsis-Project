from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import re
import threading
import os
import base64
import urllib.request
import urllib.error
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

SENDER_EMAIL     = os.environ.get('SENDER_EMAIL', '')
SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY', '')


def send_email_sendgrid(to_email, csv_bytes):
    if not SENDGRID_API_KEY or not SENDER_EMAIL:
        print("ERROR: SendGrid credentials not set in environment variables")
        return "no_credentials"

    try:
        csv_b64 = base64.b64encode(csv_bytes).decode()

        payload = {
            "personalizations": [{"to": [{"email": to_email}]}],
            "from": {"email": SENDER_EMAIL, "name": "TOPSIS Decision Maker"},
            "subject": "Your TOPSIS Analysis Results",
            "content": [{
                "type": "text/html",
                "value": """
                <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto">
                  <h2 style="color:#1D9E75">TOPSIS Analysis Complete</h2>
                  <p>Your multi-criteria decision analysis results are attached.</p>
                  <ul>
                    <li><b>Topsis Score</b>: 0 to 1 — higher = better</li>
                    <li><b>Rank</b>: 1 = best alternative</li>
                  </ul>
                </div>
                """
            }],
            "attachments": [{
                "content": csv_b64,
                "type": "text/csv",
                "filename": "topsis_result.csv",
                "disposition": "attachment"
            }]
        }

        data = json.dumps(payload).encode('utf-8')
        req  = urllib.request.Request(
            'https://api.sendgrid.com/v3/mail/send',
            data=data,
            headers={
                'Authorization': f'Bearer {SENDGRID_API_KEY}',
                'Content-Type': 'application/json'
            },
            method='POST'
        )
        with urllib.request.urlopen(req) as resp:
            print(f"EMAIL OK — sent to {to_email}, status={resp.status}")
            return "ok"

    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"SENDGRID HTTP ERROR {e.code}: {body}")
        return f"http_error_{e.code}: {body}"
    except Exception as e:
        print(f"EMAIL EXCEPTION: {type(e).__name__}: {e}")
        return f"exception: {e}"


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


# ── Debug route — call this to test email directly ──────────────────────────
@app.route('/test-email', methods=['GET'])
def test_email():
    to = request.args.get('to', '')
    if not to:
        return jsonify({"error": "Pass ?to=your@email.com"}), 400

    result = send_email_sendgrid(to, b"Model,Score,Rank\nM1,0.82,1\nM2,0.61,2")
    return jsonify({
        "to": to,
        "sender": SENDER_EMAIL,
        "api_key_set": bool(SENDGRID_API_KEY),
        "result": result
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

    # Send email in background — never blocks CSV download
    thread = threading.Thread(target=send_email_sendgrid, args=(email, csv_bytes))
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
