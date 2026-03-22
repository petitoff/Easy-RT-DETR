# Zdalny Trening GPU

Projekt ma gotowy przeplyw uruchamiania skryptow na zdalnym Jupyter/Kaggle GPU.

## Glowny mechanizm

Wykorzystywane sa:

- `pyrun-jupyter` do uruchamiania projektu na zdalnym kernelu,
- `scripts/run_remote.py` jako lokalny runner,
- `scripts/remote_dispatch.py` jako entrypoint po stronie zdalnej,
- MinIO jako storage bridge dla datasetow i artefaktow.
- glowny workflow treningowy opiera sie juz na `scripts/train.py` i YAML configach.

## Dlaczego MinIO

Przy wiekszych datasetach synchronizacja przez websocket byla niepraktyczna:

- wolna,
- podatna na zrywanie transferu,
- zbyt ciezka dla calych katalogow datasetow.

Dlatego zostal dodany flow:

1. lokalny dataset jest stage'owany do MinIO,
2. generowany jest presigned URL,
3. zdalny kernel pobiera archive bezposrednio do `data/`,
4. po treningu checkpoint moze wrocic przez MinIO z powrotem lokalnie.

To rozwiazalo problem z:

- uploadem Penn-Fudan,
- checkpointami `.pt`,
- zdalnym treningiem na Kaggle GPU.

## Dostepne presety

Runner ma przygotowane presety:

- `train-pennfudan`
- `train-synthetic`
- `eval-pennfudan`
- `visualize-pennfudan`
- `train-voc-car`
- `eval-voc-car`
- `visualize-voc-car`

W praktyce presety sa teraz cienka warstwa nad wspolnymi wrapperami:

- `scripts/train.py`
- `scripts/eval.py`
- `scripts/visualize.py`

## Typowy przeplyw

Przyklad dla VOC `car`:

```bash
.venv/bin/python scripts/run_remote.py train-voc-car \
  --device cuda \
  --minio-endpoint http://188.245.77.217:9000 \
  --minio-access-key admin \
  --minio-secret-key '***' \
  -- --set solver.epochs=5 --set solver.batch_size=4
```

## Co dziala dobrze

- zdalne uruchamianie skryptow treningowych,
- reuse datasetow stage'owanych do MinIO,
- staging datasetow do MinIO,
- pobieranie checkpointow,
- uruchamianie na Kaggle GPU bez lokalnego CUDA.

## Co wymaga dyscypliny

- po duzej zmianie architektury stare checkpointy nie sa kompatybilne,
- nazewnictwo obiektow datasetow w MinIO powinno byc stabilne,
- duze eksperymenty warto odpalac z kontrolowanym `artifact-dir`,
- lepiej traktowac MinIO jako kanoniczne miejsce dla duzych artefaktow niz polegac tylko na websocket sync.
