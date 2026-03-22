# Workflow i CLI

Po ostatnim refaktorze `Easy-RT-DETR` ma juz jeden glowny workflow eksperymentow zamiast wielu rozlacznych skryptow.

## Glowne elementy

Nowy stack sklada sie z:

- YAML configow w `configs/`,
- loadera configow i override'ow w `easy_rtdetr/configuration.py`,
- wspolnego solvera w `easy_rtdetr/engine/solver.py`,
- wspolnych builderow danych w `easy_rtdetr/data/`,
- builderow optimizera, schedulerow, warmupu i EMA w `easy_rtdetr/optim/`,
- glownego CLI w `easy_rtdetr/cli.py`,
- cienkich wrapperow w `scripts/`.

To jest teraz preferowana sciezka uruchamiania projektu.

## Glowne entrypointy

W repo pozostaly uniwersalne skrypty:

- `scripts/train.py`
- `scripts/eval.py`
- `scripts/visualize.py`
- `scripts/calibrate.py`
- `scripts/prepare_dataset.py`
- `scripts/run_remote.py`

Skrypty dataset-specific zostaly usuniete z glownego workflow.

## Trening

Przyklad treningu na VOC `car`:

```bash
.venv/bin/python scripts/train.py --config configs/voc_car/base.yaml
```

Przyklad treningu z override:

```bash
.venv/bin/python scripts/train.py \
  --config configs/voc_car/base.yaml \
  --set solver.epochs=5 \
  --set model.backbone_name=resnet34
```

Mozna tez wymusic zapis finalnego checkpointu pod konkretna sciezka:

```bash
.venv/bin/python scripts/train.py \
  --config configs/voc_car/base.yaml \
  --output artifacts/voc_car_best.pt
```

## Ewaluacja

Przyklad:

```bash
.venv/bin/python scripts/eval.py \
  --config configs/voc_car/base.yaml \
  --checkpoint runs/voc_car/.../checkpoints/best.pt
```

Wynik obejmuje miedzy innymi:

- `AP50`
- `AP75`
- `mAP@0.50:0.95`
- `avg_best_iou`
- proxy precision/recall
- duplikaty boxow

## Wizualizacja

Przyklad:

```bash
.venv/bin/python scripts/visualize.py \
  --config configs/voc_car/base.yaml \
  --checkpoint runs/voc_car/.../checkpoints/best.pt \
  --input car_on_street.avif \
  --output-dir artifacts/inference_vis
```

## Kalibracja score

Model zwraca surowe score po `sigmoid`, ale repo zawiera tez prosty etap kalibracji precision:

```bash
.venv/bin/python scripts/calibrate.py \
  --config configs/voc_car/base.yaml \
  --checkpoint runs/voc_car/.../checkpoints/best.pt \
  --output artifacts/voc_calibration.json
```

To jest przydatne do wizualizacji typu:

- `raw=...`
- `p=...`

gdzie `p` jest empiryczna precyzja oszacowana na zbiorze ewaluacyjnym.

## Co robi solver

Wspolny solver obsluguje:

- train loop,
- eval loop,
- checkpointing `best/last`,
- AMP,
- gradient clipping,
- scheduler learning rate,
- warmup,
- EMA wag modelu,
- logowanie i prosty profiling.

## Struktura wynikow

Kazdy run tworzy katalog z:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `metrics.json`
- `config_resolved.yaml`
- `train.log`

To upraszcza porownywanie eksperymentow i integracje ze zdalnym runnerem.
