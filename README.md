# TOPSIS Multi-Criteria Decision Analysis System  https://img.shields.io/badge/https%3A%2F%2Fniyati001.github.io%2FTopsis-Project%2F


> End-to-end implementation of the TOPSIS algorithm — from a pip-installable Python package to a deployed REST API with a web interface.

---

## What This Project Covers

| Part | What I Built | Tech |
|------|-------------|------|
| **I** | TOPSIS algorithm from scratch + CLI tool | Python, NumPy, Pandas |
| **II** | Published Python package to PyPI | setuptools, pip, PyPI |
| **III** | REST API + Web Interface | Flask, HTML/CSS/JS |

---

## The Algorithm — TOPSIS

TOPSIS ranks decision alternatives by measuring how close each is to the **ideal best** and how far from the **ideal worst**.

**Steps:**
1. Normalize the decision matrix
2. Apply criterion weights
3. Compute positive and negative ideal solutions
4. Calculate Euclidean distances to each ideal
5. Compute performance scores → rank

```
Score(i) = D_worst(i) / (D_best(i) + D_worst(i))
```

---

## Part I — CLI Tool

### Install
```bash
pip install Topsis-Niyati-102303356
```

### Usage
```bash
topsis <InputFile.csv> <Weights> <Impacts> <OutputFile.csv>

# Example
topsis data.csv "1,1,1,2" "+,+,-,+" result.csv
```

### Input CSV format
```
Model,Storage,Price,Camera,Battery
M1,16,2,3,4
M2,16,2.5,4,4
M3,32,3,4,5
```
- Column 1: alternative names
- Columns 2+: numeric criteria values

### Output CSV format
Original data + two new columns: `Topsis Score` and `Rank`

### Validations
- Correct number of CLI arguments
- File exists and is readable
- Input file has ≥ 3 columns
- Columns 2–n are numeric only
- Weights and impacts count matches column count
- Impacts are `+` or `-` only
- Weights are positive numbers

---

## Part II — PyPI Package

**Package:** `Topsis-Niyati-102303356`

### Use as a library
```python
import pandas as pd
from topsis import topsis

data    = pd.read_csv("data.csv")
weights = [1, 1, 1, 2]
impacts = ['+', '+', '-', '+']

result = topsis(data, weights, impacts)
print(result[['Model', 'Topsis Score', 'Rank']])
```

---

## Part III — REST API + Web Interface

### Run backend locally
```bash
pip install flask flask-cors pandas numpy
python app.py
# → http://localhost:5000
```

### Open web interface
Just open `index.html` in any browser — no build step needed.

### API — POST `/topsis`

```bash
curl -X POST http://localhost:5000/topsis \
  -F "file=@data.csv" \
  -F "weights=1,1,1,2" \
  -F "impacts=+,+,-,+" \
  -F "email=you@example.com" \
  --output result.csv
```

---

## Project Structure

```
topsis-project/
├── topsis/
│   ├── __init__.py       # Package init
│   ├── core.py           # TOPSIS algorithm (NumPy/Pandas)
│   └── cli.py            # Command-line entry point
├── app.py                # Flask REST API
├── index.html            # Web interface
├── data.csv              # Sample input
├── setup.py              # PyPI config
├── requirements.txt
└── README.md
```

---

## Skills Demonstrated

- Algorithm implementation from academic paper → production code
- Python package development and PyPI publishing
- REST API design with Flask
- Vectorized computation with NumPy
- End-to-end delivery: CLI → package → API → UI

---

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
flask>=2.0.0
flask-cors>=3.0.0
```

## License
MIT
