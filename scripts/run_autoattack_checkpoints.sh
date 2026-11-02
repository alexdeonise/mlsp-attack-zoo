#!/bin/bash
# autoattack standard la 1/255, n=200, pe checkpoint-urile cu alta preantrenare
# are nevoie de: python3 src/data.py --download imagenette
#                pip install git+https://github.com/fra31/auto-attack
cd "$(dirname "$0")/.."

for model in convnext_in22k vit_in1k; do
  echo "--- $model autoattack linf eps=1/255 n=200"
  python3 src/run_autoattack.py --dataset imagenette --model $model --eps 1/255 --n 200
done
