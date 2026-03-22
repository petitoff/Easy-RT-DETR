# Easy-RT-DETR

Ta dokumentacja opisuje aktualny stan projektu `Easy-RT-DETR` po przebudowie warstwy treningowej i eksperymentalnej.

Projekt jest praktyczna implementacja modelu w stylu RT-DETRv3 w PyTorch. Celem nie bylo wierne odtworzenie calego ekosystemu PaddleDetection, tylko zbudowanie czytelnego i rozwijalnego pipeline'u:

- od modelu i lossow,
- przez wspolny trening i ewaluacje,
- po zdalne biegi na GPU przez Jupyter/Kaggle,
- wraz z wizualizacja i formalnymi metrykami `AP50/AP75/mAP`.

## Co juz dziala

Aktualnie repo zawiera:

- backbone `ResNet` z `torchvision`,
- wieloskalowy `HybridEncoder`,
- `QuerySelection` z wieloma grupami query,
- decoder z deformable cross-attention,
- contrastive denoising dla treningu,
- rozdzielenie galezi `o2o` i `o2m`,
- training-only auxiliary dense head w stylu PP-YOLOE,
- formalna ewaluacje `AP50`, `AP75`, `mAP@0.50:0.95`,
- wspolny solver z:
  - AMP,
  - warmupem,
  - schedulerami,
  - EMA,
  - checkpointingiem,
  - profilingiem,
- YAML configi eksperymentow,
- nowe uniwersalne CLI i wrappery w `scripts/`,
- zdalny trening na GPU przez `pyrun-jupyter` i MinIO.

## Glowna motywacja

Projekt sluzy do szybkiego iterowania nad architektura typu DETR bez blokowania sie na:

- custom kernels,
- zaleznosciach od PaddlePaddle,
- koniecznosci posiadania lokalnego CUDA,
- recznym utrzymywaniu wielu rozproszonych skryptow treningowych.

W praktyce oznacza to:

- architektura i logika treningowa rozwijane sa lokalnie,
- pelniejsze biegi treningowe mozna odpalac na zdalnym GPU,
- eksperymenty sa opisywane przez configi zamiast przez osobne skrypty per dataset,
- efekty mozna od razu sprawdzac na obrazach i formalnych metrykach.

## Jak czytac te dokumentacje

- [Architektura](architecture.md) opisuje obecny sklad modelu i glowne odchylenia od referencji RT-DETRv3.
- [Workflow i CLI](workflow.md) opisuje nowy sposob pracy z repo po refaktorze.
- [Eksperymenty i Wyniki](experiments.md) zbiera najwazniejsze uruchomienia treningu i obserwacje.
- [Zdalny Trening GPU](remote-training.md) opisuje przeplyw Jupyter/Kaggle + MinIO.
- [Status i Roadmapa](status.md) podsumowuje, co jest juz mocne, a co nadal jest otwarte.
