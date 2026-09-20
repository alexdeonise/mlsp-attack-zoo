# mlsp-attack-zoo

Lucrarea 2 din grupul MLSP: cât de bine se transferă atacurile adversariale clasice (FGSM, BIM, PGD,
DeepFool, Carlini-Wagner, AutoAttack) pe arhitecturi moderne (ResNet-50, EfficientNet, ViT-B/16,
ConvNeXt), și care rezistă cel mai bine la ce buget de perturbație.

Echipa: Andreea (atacurile pe gradient și tabelul principal), David (Carlini-Wagner, AutoAttack,
transferabilitate, vizualizări, reproducibilitate).

## Planul tău, pas cu pas

- [Andreea](docs/plan-andreea.md)
- [David](docs/plan-david.md)

## Setup (15 minute)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/fra31/auto-attack     # pentru David
python src/data.py --download imagenette                 # ~330 MB, o singura data
```

Ai nevoie de GPU pentru grila completă. Dacă nu ai (laptop cu Radeon integrat, Mac fără CUDA), două variante:

- **Google Colab**, gratis, cu GPU T4: https://colab.research.google.com. Într-un notebook nou:
  `!git clone https://github.com/Maxxtra/mlsp-attack-zoo && cd mlsp-attack-zoo && pip install -r requirements.txt`
  apoi rulezi scripturile cu `!python src/...`. Rezultatele le descarci și le comiți de pe laptop.
- **Pe CPU**, pentru prima cifră: `--n 100`, modelul `efficientnet` (cel mai ușor), și la AutoAttack
  flag-ul `--fast` (doar APGD-CE, în loc de tot ansamblul). Câteva minute în loc de ore.

## Un lucru important despre date

Modelele preantrenate din `timm` sunt antrenate pe ImageNet (1000 de clase). Pe **Imagenette** (10 clase
care sunt chiar clase ImageNet) le poți folosi direct, fără să antrenezi nimic: alegem doar cele 10
logit-uri care contează. Ăsta e cel mai rapid drum către prima cifră, și e ce rulezi vineri.

Pe **CIFAR-10** clasele nu sunt clase ImageNet, deci modelul trebuie fine-tuned întâi
(`src/finetune_cifar.py`, ~10 minute pe GPU per model). Îl faci de luni.

## Structura

```
src/data.py          Imagenette + CIFAR-10, tensori in [0,1] (normalizarea o face modelul)
src/models.py        incarca un model timm si il impacheteaza cu normalizare + selectia claselor
src/run_attack.py    ruleaza un atac (FGSM/BIM/PGD/DeepFool/CW) la un eps, scrie results/results.csv   (Andreea)
src/run_autoattack.py  la fel, cu AutoAttack                                                            (David)
src/transfer.py      matricea de transfer 4x4                                                          (David)
src/finetune_cifar.py  fine-tune rapid pe CIFAR-10
results/             results.csv, o linie per (dataset, model, atac, norma, eps)
figures/             figurile, generate din results.csv
```

## Tutorial 1 (Andreea, până vineri): prima cifră

```bash
python src/run_attack.py --dataset imagenette --model resnet50 --attack pgd --eps 4/255
python src/run_attack.py --dataset imagenette --model vit_base_patch16_224 --attack pgd --eps 4/255
```

Scriptul: încarcă modelul, ia 1000 de imagini de test, măsoară acuratețea pe imagini curate, generează
imagini adversariale cu `torchattacks`, măsoară acuratețea pe ele. Scrie o linie în `results/results.csv`.
Diferența dintre cele două acurateți e prima ta cifră.

Următorul pas (până luni): implementezi FGSM și PGD de la zero în `src/my_attacks.py` (creezi tu
fișierul) și verifici că dau aceleași cifre ca torchattacks. Apoi rulezi grila: 4 atacuri × 4 modele ×
6 valori de eps. Un `for` peste `run_attack.py` e suficient; vezi `scripts/grid.sh` ca exemplu.

De citit: FGSM https://arxiv.org/abs/1412.6572, PGD https://arxiv.org/abs/1706.06083,
DeepFool https://arxiv.org/abs/1511.04599

## Tutorial 2 (David, până vineri): AutoAttack

```bash
python src/run_autoattack.py --dataset imagenette --model resnet50 --eps 4/255
```

AutoAttack e referința standard din domeniu (un ansamblu de 4 atacuri, fără hiperparametri de reglat).
Cifrele lui trebuie să fie cele mai „grele" din tabel, deci robust accuracy cel mai mic. Dacă un atac
de-al Andreei dă robust accuracy mai mic decât AutoAttack la același eps, ceva e greșit undeva.

Următorii pași: Carlini-Wagner (luni), matricea de transfer cu `src/transfer.py` (joi), curbele
tărie-vs-iterații și vizualizările (duminică).

De citit: AutoAttack https://arxiv.org/abs/2003.01690, Carlini-Wagner https://arxiv.org/abs/1608.04644

## Modelele

Numele din timm pe care le folosim, toate preantrenate pe ImageNet:

| scurt | timm |
|---|---|
| resnet50 | `resnet50.a1_in1k` |
| efficientnet | `efficientnet_b0.ra_in1k` |
| vit | `vit_base_patch16_224.augreg_in21k_ft_in1k` |
| convnext | `convnext_tiny.fb_in1k` |

## Rezultatele

`results/results.csv` are coloanele: `dataset, model, attack, norm, eps, n, clean_acc, robust_acc, seconds`.
Fiecare rulare adaugă o linie. Figurile le generați din CSV, nu de mână. Un experiment e gata când
scriptul îl reproduce de la zero și CSV-ul e comis.

## Ce e în lucrare și ce rămâne future work

În lucrare intră ce e în `results/` la 20 septembrie: grila Linf n=200 (4 modele × FGSM/BIM/PGD × 6 bugete,
plus DeepFool), AutoAttack n=500 la 1/255, 2/255, 4/255, și testul cu 1000 de clase (`results/test_1000_clase.txt`).

Rămân future work, cu scripturile gata în `scripts/`, dar nerulate:

| Script | Ce face | Scrie în |
|---|---|---|
| `scripts/run_transfer.sh` | matricea de transfer 4×4: PGD 4/255 generat pe modelul sursă, evaluat pe toate 4 modelele, n=200 | `results/transfer.csv` |
| `scripts/run_cw.sh` | Carlini-Wagner L2 pe cele 4 modele, n=200, 100 de pași | `results/results.csv` |
| `scripts/run_grila_l2.sh` | grila L2 (FGSM/BIM/PGD × 6 bugete de la 0.25 la 8.0) pe cele 4 modele, n=200 | `results/results.csv` |

Fiecare are și o variantă `_test.sh` cu n mic, ca să vezi că merge cap-coadă înainte de rularea mare.
`run_transfer.sh` merge pe batch-uri și pe GPU-ul de Mac (mps), deci încape și pe un laptop.
