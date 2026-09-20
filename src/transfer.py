"""Matricea de transfer: exemple adversariale generate pe modelul sursa (PGD, 10 pasi),
evaluate pe toate modelele. Lucreaza pe batch-uri, deci incape si pe un laptop.
    python src/transfer.py --dataset imagenette --eps 4/255 --n 200
Scrie results/transfer.csv cu coloanele: source, target, eps, n, robust_acc
Device: cuda daca exista, altfel mps (GPU-ul de Mac), altfel cpu. Se poate forta cu --device.
"""
import argparse, os, csv, torch, torchattacks
from data import get_loader
from models import load, NAMES
from run_attack import parse_eps

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "transfer.csv")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="imagenette")
    ap.add_argument("--eps", default="4/255")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=pick_device())
    a = ap.parse_args()
    eps = parse_eps(a.eps)
    print(f"device={a.device}  n={a.n}  eps={a.eps}  batch={a.batch}")

    models = {k: load(k, a.dataset, a.device) for k in NAMES}
    correct = {(s, t): 0 for s in NAMES for t in NAMES}
    total = 0
    for x, y in get_loader(a.dataset, n=a.n, batch=a.batch):
        x, y = x.to(a.device), y.to(a.device)
        total += len(y)
        for src, ms in models.items():
            x_adv = torchattacks.PGD(ms, eps=eps, alpha=eps / 4, steps=10, random_start=True)(x, y)
            with torch.no_grad():
                for tgt, mt in models.items():
                    correct[(src, tgt)] += (mt(x_adv).argmax(1) == y).sum().item()
        print(f"  {total}/{a.n} imagini")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "eps", "n", "robust_acc"])
        for src in NAMES:
            for tgt in NAMES:
                acc = correct[(src, tgt)] / total
                print(f"{src:>12} -> {tgt:<12} robust={acc:.3f}")
                w.writerow([src, tgt, a.eps, total, f"{acc:.4f}"])
    print("scris:", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
