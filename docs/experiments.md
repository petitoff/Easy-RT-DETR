# Eksperymenty i Wyniki

Ten dokument zbiera najwazniejsze etapy eksperymentow wykonanych do tej pory.

## 1. Pierwsze uruchomienie na CPU

Poczatkowo model byl sprawdzany lokalnie na CPU:

- testy shape,
- `forward/backward`,
- prosty trening na sztucznym datasecie,
- zapis i odczyt checkpointow.

To potwierdzilo, ze sama logika modelu i treningu dziala bez CUDA.

## 2. Penn-Fudan Pedestrian

Penn-Fudan zostal wybrany jako pierwszy realny dataset, bo:

- jest maly,
- ma prawdziwe obrazy,
- ma gotowe boxy,
- jedna klasa upraszcza debugowanie.

### Najwazniejsze obserwacje

Najpierw okazalo sie, ze problem nie lezal w anotacjach. Boxy z masek i oficjalnych adnotacji pokrywaly sie poprawnie.

Najwieksza poprawa przyszla z:

- `letterbox` zamiast brutalnego resize do kwadratu,
- lepszego postprocessu,
- dluzszego treningu,
- stopniowego zblizania treningu do referencji RT-DETRv3,
- auxiliary branch w stylu PP-YOLOE.

### Mocny punkt odniesienia

Najbardziej udany checkpoint Penn-Fudan po treningu na GPU dawal profil:

- `avg_best_iou ~= 0.824`
- `pred_precision_proxy@0.50 ~= 0.984`
- `gt_recall@0.50 ~= 0.639`
- `duplicate_pair_ratio@0.40 = 0.000`

W praktyce oznaczalo to:

- bardzo dobre polozenie boxow,
- brak problemu z duplikatami,
- sensowny recall jak na maly eksperymentalny setup.

## 3. VOC 2007 - tylko klasa `car`

Nastepny krok to trenowanie modelu tylko na samochodach z Pascal VOC 2007.

Powody:

- inna domena niz piesi,
- bardziej zroznicowane obrazy,
- mozliwosc szybkiego sprawdzenia transferu poza jeden benchmark.

### Baseline VOC `car`

Pierwszy sensowny checkpoint dla samochodow trenowany na GPU dal:

- `avg_best_iou ~= 0.642`
- `pred_precision_proxy@0.50 ~= 0.769`
- `gt_recall@0.50 ~= 0.490`
- `duplicate_pair_ratio@0.40 = 0.000`

To nie jest poziom Penn-Fudan, ale wynik byl obiecujacy:

- model poprawnie lokalizowal samochody,
- nawet na customowych zdjeciach potrafil zwracac sensowne boxy,
- score byl raczej konserwatywny niz agresywny.

### Nowy HybridEncoder

Po przebudowie `HybridEncoder` wykonano kolejny bieg na VOC `car`.

Wynik po `5` epokach:

- `avg_best_iou ~= 0.581`
- `pred_precision_proxy@0.50 ~= 0.672`
- `gt_recall@0.50 ~= 0.356`
- `duplicate_pair_ratio@0.40 = 0.000`

Interpretacja:

- nowy neck nie wygral jeszcze po krotkim treningu,
- prawdopodobnie potrzebuje dluzszego biegu i ewentualnego dostrojenia hiperparametrow,
- sama zmiana architektury nie daje automatycznego zysku po bardzo krotkim fine-tuningu.

## 4. Wnioski z wynikow

Najwazniejsze wnioski praktyczne:

- model jest juz realnie uzywalny jako prototyp detekcji,
- pipeline dziala lokalnie i zdalnie,
- poprawa parity z referencja realnie pomogla na Penn-Fudan,
- VOC `car` pokazal, ze model generalizuje poza jedna klase,
- nowy neck wymaga jeszcze pelniejszego eksperymentu treningowego.

## 5. Jak interpretowac score typu `0.29`

Score przy boxie:

- jest scorem po `sigmoid`,
- sluzy do rankingu i filtrowania,
- nie jest idealnie skalibrowanym prawdopodobienstwem.

Dlatego poprawny box moze miec stosunkowo niski score, szczegolnie gdy:

- trening byl krotki,
- dataset jest maly,
- domena testowa rozni sie od treningowej,
- model zostal ustawiony bardziej konserwatywnie.
