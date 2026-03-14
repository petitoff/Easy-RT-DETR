# Easy-RT-DETR

Ta dokumentacja opisuje, co zostalo zaimplementowane w projekcie `Easy-RT-DETR`, jakie eksperymenty zostaly wykonane oraz na jakim etapie znajduje sie model.

Projekt jest praktyczna, rozwijana iteracyjnie implementacja modelu w stylu RT-DETRv3 w bibliotece PyTorch. Celem nie bylo skopiowanie calego ekosystemu PaddleDetection 1:1, tylko zbudowanie dzialajacego i testowalnego pipeline'u:

- od modelu i lossow,
- przez trening lokalny na CPU,
- po zdalny trening na GPU przez Jupyter/Kaggle,
- wraz z ewaluacja i wizualizacja predykcji.

## Co juz dziala

Aktualnie repo zawiera:

- backbone `ResNet` z `torchvision`,
- wieloskalowy `HybridEncoder`,
- `QuerySelection` z wieloma grupami query,
- decoder z deformable cross-attention,
- contrastive denoising dla treningu,
- rozdzielenie gałęzi `o2o` i `o2m`,
- training-only auxiliary dense head w stylu PP-YOLOE,
- trening lokalny na CPU,
- zdalny trening na GPU przez `pyrun-jupyter`,
- staging datasetow i artefaktow przez MinIO,
- narzedzia do ewaluacji i wizualizacji.

## Glowna motywacja

Projekt sluzy do szybkiego iterowania nad architektura typu DETR bez blokowania sie na:

- custom kernels,
- zaleznosciach od PaddlePaddle,
- koniecznosci posiadania lokalnego CUDA.

W praktyce oznacza to:

- architektura i logika treningowa rozwijane sa lokalnie,
- pelniejsze biegi treningowe mozna odpalac na zdalnym GPU,
- efekty mozna od razu sprawdzac na obrazach i prostych metrykach proxy.

## Jak czytac te dokumentacje

- [Architektura](architecture.md) opisuje obecny sklad modelu oraz roznice wzgledem referencji RT-DETRv3.
- [Eksperymenty i Wyniki](experiments.md) zbiera najwazniejsze uruchomienia treningu i obserwacje.
- [Zdalny Trening GPU](remote-training.md) opisuje przeplyw Jupyter/Kaggle + MinIO.
- [Status i Roadmapa](status.md) podsumowuje, co jest juz mocne, a co nadal jest otwarte.
