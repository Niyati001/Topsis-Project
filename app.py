from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from email.message import EmailMessage
import pandas as pd
import numpy as np
import io
import re
import threading
import os
import smtplib

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

SMTP_HOST = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
SMTP_USERNAME = os.environ.get('SMTP_USERNAME', '')
SMTP_APP_PASSWORD = os.environ.get('SMTP_APP_PASSWORD', '').replace(' ', '')
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', SMTP_USERNAME)


def send_result_email_smtp(to_email, csv_bytes):
    message = EmailMessage()
    message['From'] = SENDER_EMAIL
    message['To'] = to_email
    message['Subject'] = 'Your TOPSIS Analysis Results'
    message.set_content('Your TOPSIS analysis results are attached.')
    message.add_alternative(
        """
        <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto">
          <h2 style="color:#1D9E75">TOPSIS Analysis Complete</h2>
          <p>Your TOPSIS analysis results are attached.</p>
        </div>
        """,
        subtype='html',
    )
    message.add_attachment(
        csv_bytes,
        maintype='text',
        subtype='csv',
        filename='topsis_result.csv',
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_APP_PASSWORD)
        server.send_message(message)

    return "ok"


def send_result_email(to_email, csv_bytes):
    if SMTP_USERNAME and SMTP_APP_PASSWORD:
        try:
            return send_result_email_smtp(to_email, csv_bytes)
        except Exception as e:
            print(f"SMTP EMAIL EXCEPTION: {type(e).__name__}: {e}")
            return f"smtp_exception: {e}"


def send_result_email_smtp(to_email, csv_bytes, result):

    top_results = result.sort_values(by="Rank").head(3)
    top_choice = top_results.iloc[0]
    preview_html = top_results.drop(columns=["Topsis Score"]).to_html(index=False)

    message = EmailMessage()
    message['From'] = SENDER_EMAIL
    message['To'] = to_email
    message['Subject'] = '📊 Your TOPSIS Results Are Ready'

    message.set_content("Your TOPSIS results are attached.")

    message.add_alternative(f"""
    <div style="font-family: Arial; max-width:600px; margin:auto; padding:20px;">

      <h2>📊 Your TOPSIS Results</h2>

      <p>Here are your top-ranked alternatives:</p>

      {preview_html}

      <p><strong>🏆 Top Choice:</strong> {top_choice[0]}</p>

      <p>The full results are attached as a CSV file.</p>

    </div>
    """, subtype='html')

    message.add_attachment(
        csv_bytes,
        maintype='text',
        subtype='csv',
        filename='topsis_result.csv',
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_APP_PASSWORD)
        server.send_message(message)

    return "ok"



def send_result_email(to_email, csv_bytes, result):
    if SMTP_USERNAME and SMTP_APP_PASSWORD:
        try:
            return send_result_email_smtp(to_email, csv_bytes, result)
        except Exception as e:
            print(f"SMTP EMAIL EXCEPTION: {type(e).__name__}: {e}")
            return f"smtp_exception: {e}"

    print("ERROR: SMTP credentials not set")
    return "no_credentials"


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

    result = send_result_email(to, b"Model,Score,Rank\nM1,0.82,1\nM2,0.61,2")
    return jsonify({
        "to": to,
        "sender": SENDER_EMAIL,
        "smtp_configured": bool(SMTP_USERNAME and SMTP_APP_PASSWORD),
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
        top_results = result.sort_values(by="Rank").head(3)
        top_choice = top_results.iloc[0]
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

    output = io.StringIO()
    result.to_csv(output, index=False)
    csv_bytes = output.getvalue().encode()

    # Send email in background — never blocks CSV download
    thread = threading.Thread(target=send_result_email, args=(email, csv_bytes, result))    
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5120)), debug=False)
