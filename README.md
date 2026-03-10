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

## Zdalne uruchamianie na Jupyter/Kaggle GPU

Jeżeli masz działający zdalny Jupyter server i zainstalowane `pyrun-jupyter`, możesz odpalać trening lub wizualizację z tego repo przez:

```bash
.venv/bin/python scripts/run_remote.py train-pennfudan \
  --url http://localhost:8888 \
  --token YOUR_TOKEN \
  --device cuda \
  -- --epochs 15 --batch-size 2 --image-size 256 --output artifacts/pennfudan_kaggle_15e.pt
```

Skrypt:
- synchronizuje cały projekt do zdalnego kernela,
- odpala właściwy entrypoint przez wrapper `scripts/remote_dispatch.py`,
- pobiera artefakty, np. checkpointy `artifacts/*.pt`.

Dla większych datasetów możesz ominąć ciężki upload przez kernel i stage'ować dataset przez MinIO:

```bash
.venv/bin/python scripts/run_remote.py train-pennfudan \
  --device cuda \
  --minio-endpoint http://188.245.77.217:9000 \
  --minio-access-key admin \
  --minio-secret-key '...' \
  -- --epochs 15 --batch-size 2 --image-size 256 --output artifacts/pennfudan_kaggle_15e.pt
```

Jeżeli lokalnie istnieje `data/PennFudanPed.zip` albo `data/PennFudanPed`, preset `train-pennfudan` automatycznie:
- wrzuci dataset archive do MinIO,
- wygeneruje presigned URL,
- pobierze i rozpakuję dataset po stronie zdalnego kernela do `data/`,
- uruchomi trening już bez synchronizowania lokalnego `data/` przez websocket.

Dostępne presety:
- `train-pennfudan`
- `train-synthetic`
- `eval-pennfudan`
- `visualize-pennfudan`
- `custom --script path/to/script.py`

Treningowe skrypty mają teraz `--device auto|cpu|cuda`, więc ten sam entrypoint działa lokalnie i zdalnie.

Uwagi praktyczne:
- `artifacts/` są domyślnie wykluczane z synchronizacji, żeby nie wysyłać starych checkpointów.
- preset `train-synthetic` wyklucza też `data/`, bo dataset nie jest tam potrzebny.
- ustawienia MinIO możesz podać przez `--minio-*` albo przez `scripts/.env` (`MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, opcjonalnie `MINIO_BUCKET`).
