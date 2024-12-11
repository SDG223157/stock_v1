from flask import Flask, request, jsonify, render_template
import subprocess
import os

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get user input from POST request
        data = request.get_json()
        ticker = data.get("ticker", "AAPL")
        end_date = data.get("end_date", None)
        lookback_days = data.get("lookback_days", 365)
        crossover_days = data.get("crossover_days", 180)

        # Execute analysis.py with user input
        result = subprocess.run(
            ["python3", "analysis.py", ticker, end_date, str(lookback_days), str(crossover_days)],
            cwd=os.path.dirname(os.path.realpath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
