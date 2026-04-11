import sys
import pandas as pd
from topsis.core import topsis


def main():
    if len(sys.argv) != 5:
        print("Usage: topsis <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        print("Example: topsis data.csv \"1,1,1,2\" \"+,+,-,+\" result.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    weights_str = sys.argv[2]
    impacts_str = sys.argv[3]
    output_file = sys.argv[4]

    # Parse weights
    try:
        weights = list(map(float, weights_str.split(',')))
    except ValueError:
        print("Error: Weights must be numeric values separated by commas (e.g., 1,2,1,3)")
        sys.exit(1)

    # Parse impacts
    impacts = [i.strip() for i in impacts_str.split(',')]

    # Read input file
    try:
        data = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Run TOPSIS
    try:
        result = topsis(data, weights, impacts)
        result.to_csv(output_file, index=False)
        print(f"Result saved to '{output_file}'")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
