"""
TOPSIS Web Service - Part III
Flask backend that processes TOPSIS and sends results via email.

Install: pip install flask pandas numpy flask-mail flask-cors
Run: python app.py
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import io
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topsis.core import topsis, validate_inputs

app = Flask(__name__)
CORS(app)

# --- Email config (update with your SMTP settings) ---
# app.config['MAIL_SERVER'] = 'smtp.gmail.com'
# app.config['MAIL_PORT'] = 587
# app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USERNAME'] = 'your@email.com'
# app.config['MAIL_PASSWORD'] = 'your_app_password'
# from flask_mail import Mail, Message
# mail = Mail(app)


def validate_email(email):
    pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
    return bool(re.match(pattern, email))


@app.route('/topsis', methods=['POST'])
def run_topsis():
    # Validate file
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a CSV'}), 400

    weights_str = request.form.get('weights', '')
    impacts_str = request.form.get('impacts', '')
    email = request.form.get('email', '')

    # Validate email
    if not email or not validate_email(email):
        return jsonify({'error': 'Invalid or missing email address'}), 400

    # Validate weights
    try:
        weights = [float(w.strip()) for w in weights_str.split(',')]
        if any(w <= 0 for w in weights):
            return jsonify({'error': 'All weights must be positive'}), 400
    except ValueError:
        return jsonify({'error': 'Weights must be numeric values separated by commas'}), 400

    # Validate impacts
    impacts = [i.strip() for i in impacts_str.split(',')]
    if any(i not in ['+', '-'] for i in impacts):
        return jsonify({'error': 'Impacts must be + or - only'}), 400

    if len(weights) != len(impacts):
        return jsonify({
            'error': f'Number of weights ({len(weights)}) must equal impacts ({len(impacts)})'
        }), 400

    # Read CSV
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400

    # Run TOPSIS
    try:
        result = topsis(data, weights, impacts)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

    # Save result to in-memory CSV
    output = io.StringIO()
    result.to_csv(output, index=False)
    output.seek(0)

    # --- Optional: Send via email ---
    # msg = Message('TOPSIS Analysis Results', recipients=[email])
    # msg.body = 'Please find your TOPSIS analysis results attached.'
    # msg.attach('topsis_result.csv', 'text/csv', output.getvalue())
    # mail.send(msg)

    # Return file as download
    output_bytes = io.BytesIO(output.getvalue().encode())
    return send_file(
        output_bytes,
        mimetype='text/csv',
        as_attachment=True,
        download_name='topsis_result.csv'
    )


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'TOPSIS Web Service'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
