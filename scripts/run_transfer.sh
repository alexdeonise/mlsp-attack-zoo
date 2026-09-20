#!/bin/bash
# matricea de transfer 4x4: pgd generat pe modelul sursa, evaluat pe toate 4 modelele
# imagenette, eps 4/255, n=200. scrie results/transfer.csv
# descarca imagenette daca lipseste (330 MB, o singura data)
cd "$(dirname "$0")/.."

python3 src/data.py --download imagenette
python3 src/transfer.py --dataset imagenette --eps 4/255 --n 200
