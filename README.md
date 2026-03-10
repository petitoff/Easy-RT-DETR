# Easy-RT-DETR

Minimalny start implementacji RT-DETRv3 w PyTorch.

Aktualny zakres:
- backbone `ResNet` z `torchvision`,
- prosty hybrid encoder,
- query selection z wieloma grupami query,
- decoder z masked self-attention perturbation i multi-scale deformable cross-attention,
- treningowa gałąź one-to-many oraz CNN auxiliary dense head,
- matcher, lossy i testy smoke/shape.

To jest MVP pod dalszy rozwój. Architektura inferencyjna jest utrzymana w stylu DETR, a elementy specyficzne dla RT-DETRv3 pozostają treningowe.

## Instalacja

```bash
pip install -e .[dev]
pytest
```

## Szybki test treningu na CPU

```bash
.venv/bin/python scripts/train_synthetic_cpu.py
```

Skrypt:
- buduje mały wariant modelu,
- trenuje go na syntetycznych obrazach z prostokątami,
- wypisuje przebieg lossu,
- zapisuje checkpoint do `artifacts/synthetic_cpu_checkpoint.pt`.

## Trening na Penn-Fudan (CPU)

Najpierw pobierz dataset do `data/PennFudanPed`, a potem uruchom:

```bash
.venv/bin/python scripts/train_pennfudan_cpu.py
```

Domyślnie skrypt bierze mały subset, żeby CPU test trwał krótko.

## Wizualizacja predykcji

Po treningu możesz narysować GT i predykcje na obrazach:

```bash
.venv/bin/python scripts/visualize_pennfudan_predictions.py --checkpoint artifacts/pennfudan_cpu_3e.pt
```
