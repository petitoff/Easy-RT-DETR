# Eksperymenty i Wyniki

Ten dokument zbiera najwazniejsze etapy eksperymentow wykonanych do tej pory.

## 1. Weryfikacja bazowa na CPU

Poczatkowo model byl sprawdzany lokalnie na CPU:

- testy shape,
- `forward/backward`,
- prosty trening na sztucznym datasecie,
- zapis i odczyt checkpointow.

To potwierdzilo, ze sama logika modelu i treningu dziala bez CUDA.

## 2. Penn-Fudan Pedestrian

Penn-Fudan byl pierwszym realnym datasetem:

- maly,
- prosty,
- jedna klasa,
- bardzo dobry do debugowania boxow i lossow.

### Najwazniejsze obserwacje

Kluczowe poprawy jakosci przyszly z:

- `letterbox` zamiast deformujacego resize,
- lepszego postprocessu,
- dluzszego treningu,
- parity z RT-DETRv3 w query selection i denoising,
- auxiliary branch w stylu PP-YOLOE.

### Mocny punkt odniesienia

Najbardziej udany checkpoint Penn-Fudan po treningu na GPU dawal profil:

- `avg_best_iou ~= 0.824`
- `pred_precision_proxy@0.50 ~= 0.984`
- `gt_recall@0.50 ~= 0.639`
- `duplicate_pair_ratio@0.40 = 0.000`

W praktyce oznaczalo to:

- bardzo dobre polozenie boxow,
- brak duplikatow,
- sensowny recall jak na maly eksperymentalny setup.

## 3. VOC 2007 - tylko klasa `car`

VOC `car` byl pierwszym sensownym testem poza jedna klasa pieszych.

### Wczesne biegi

Pierwsze checkpointy pokazywaly, ze model:

- poprawnie lokalizuje samochody,
- jest raczej konserwatywny score'owo,
- generalizuje na customowe zdjecia lepiej niz czysto zabawkowy baseline.

### Przebudowa architektury

W kolejnych iteracjach dodano:

- parity elementow transformera wzgledem `ppdet`,
- nowy `HybridEncoder`,
- auxiliary path w stylu PP-YOLOE,
- training recipe z EMA, schedulerem i lepszym backbone.

### Mocniejszy wariant VOC

Po dluzszym treningu z `ResNet34`, AMP i nowym training stackiem uzyskano profil:

- `avg_best_iou ~= 0.662`
- `pred_precision_proxy@0.50 ~= 0.747`
- `gt_recall@0.50 ~= 0.518`
- `duplicate_pair_ratio@0.40 = 0.000`
- `AP50 ~= 0.473`
- `AP75 ~= 0.357`
- `mAP@0.50:0.95 ~= 0.337`

To jest obecny formalny punkt odniesienia dla samochodow.

## 4. BDD100K vehicle-3

Dodano tez pipeline dla nowoczesniejszego datasetu drogowego:

- klasy `car`
- `truck`
- `bus`

Zakres zaimplementowany:

- przygotowanie odfiltrowanego subsetu,
- dataset loader,
- integracja z MinIO,
- cache po stronie Kaggle przez `mini-mlflow-cli`,
- smoke training na GPU.

To jest baza pod dalsze bardziej realistyczne eksperymenty wieloklasowe.

## 5. Jak interpretowac score modelu

Score przy boxie:

- jest scorem po `sigmoid`,
- sluzy do rankingu i filtrowania,
- nie jest idealnie skalibrowanym prawdopodobienstwem.

Dlatego w repo dodano tez prosty etap kalibracji precision. Dzieki temu wizualizacje moga pokazywac:

- `raw=...` jako surowy score modelu,
- `p=...` jako oszacowana precyzja empiryczna na zbiorze ewaluacyjnym.

## 6. Wnioski z wynikow

Najwazniejsze wnioski praktyczne:

- model jest juz uzywalnym research prototype, a nie tylko szkicem,
- pipeline dziala lokalnie i zdalnie,
- Penn-Fudan pokazal mocna jakosc lokalizacji,
- VOC `car` pokazal sensowna generalizacje i daje juz formalne `mAP`,
- BDD100K daje nowy kierunek pod bardziej realistyczne sceny drogowe,
- najwiekszy dalszy zysk prawdopodobnie bedzie teraz z lepszego recipe i wiekszych datasetow, nie tylko z dalszego komplikowania architektury.
