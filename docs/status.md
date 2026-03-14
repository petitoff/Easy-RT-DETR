# Status i Roadmapa

## Aktualny status projektu

Najuczciwszy opis obecnego etapu:

`Easy-RT-DETR` jest juz dzialajacym research prototype / strong MVP, a nie tylko szkicem architektury.

Projekt ma:

- dzialajacy model w PyTorch,
- dzialajacy trening lokalny,
- dzialajacy trening zdalny na GPU,
- rzeczywiste eksperymenty na dwoch datasetach,
- narzedzia do wizualizacji i ewaluacji.

## Co jest juz mocne

- pipeline end-to-end od treningu do wizualizacji,
- parity kilku waznych elementow RT-DETRv3:
  - denoising,
  - `o2o/o2m`,
  - query selection,
  - auxiliary head w stylu PP-YOLOE,
- dobre wyniki na Penn-Fudan,
- sensowna detekcja samochodow na VOC i zdjeciach customowych,
- integracja z Kaggle GPU przez Jupyter i MinIO.

## Co jest srednio dojrzale

- ocena modelu nadal opiera sie glownie o proxy metryki zamiast pelnego `mAP`,
- eksperymenty sa jeszcze stosunkowo krotkie,
- nowy `HybridEncoder` nie zostal jeszcze uczciwie sprawdzony w dluzszym treningu,
- repo nie ma jeszcze pelnej polityki wersjonowania checkpointow po zmianach architektury.

## Co nadal jest otwarte technicznie

- dluzsze treningi VOC i porownania po wiekszej liczbie epok,
- trening wieloklasowy zamiast tylko `car`,
- klasyczne benchmarkowanie `mAP`,
- stabilniejszy eksperymentalny protokol porownawczy,
- dalsze zblizanie necka i auxiliary path do referencji PaddleDet,
- opcjonalnie export i inferencja bardziej produktowa.

## Najwazniejszy stan modelu dzisiaj

Jesli patrzec pragmatycznie:

- najlepsza jakosc lokalizacji i czystosci boxow byla osiagnieta na Penn-Fudan,
- VOC `car` pokazal sensowna generalizacje,
- nowy `HybridEncoder` jest obiecujacy architektonicznie, ale wymaga dluzszego treningu, zanim bedzie mozna powiedziec, ze jest lepszy od poprzedniego prostszego necka.

## Rekomendowane kolejne kroki

Najbardziej sensowna kolejnosc na nastepne iteracje:

1. Ustabilizowac dokumentowany benchmark VOC `car`.
2. Przetrenowac nowy `HybridEncoder` dluzej na GPU.
3. Dodac `mAP` i bardziej formalna ewaluacje.
4. Rozszerzyc trening na wiecej klas VOC lub inny realny dataset.
5. Dopiero potem dalej komplikowac architekture.

## Uwagi praktyczne

- stare checkpointy od prostszego necka nie sa kompatybilne z aktualnym kodem po zmianie `HybridEncoder`,
- to jest swiadoma cena szybkiego rozwoju architektury,
- w praktyce trzeba traktowac checkpointy jako powiazane z konkretna wersja kodu.
