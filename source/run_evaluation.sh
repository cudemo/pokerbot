#!/bin/bash

if [ ! -f "venv/bin/activate" ]; then
  echo "venv does not exist."
  exit 0
fi

# load venv:
source venv/bin/activate

# update / install requirements:
pip install -r requirements.txt

# Run the evaluation script.
# The config file specifies which baseline and participant implementations to load and evaluate
# If --store_result is set, the ranking is saved in the result.txt file.
python3 runEvaluation.py --store_result
