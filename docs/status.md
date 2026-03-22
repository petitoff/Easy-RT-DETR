# Status i Roadmapa

## Aktualny status projektu

Najuczciwszy opis obecnego etapu:

`Easy-RT-DETR` jest juz dzialajacym research prototype / strong MVP, a nie tylko szkicem architektury.

Projekt ma:

- dzialajacy model w PyTorch,
- dzialajacy trening lokalny,
- dzialajacy trening zdalny na GPU,
- rzeczywiste eksperymenty na wielu datasetach,
- narzedzia do wizualizacji i ewaluacji.

## Co jest juz mocne

- pipeline end-to-end od treningu do wizualizacji,
- wspolny config system i nowe CLI,
- wspolny solver z EMA, warmupem i schedulerami,
- parity kilku waznych elementow RT-DETRv3:
  - denoising,
  - `o2o/o2m`,
  - query selection,
  - auxiliary head w stylu PP-YOLOE,
- dobre wyniki na Penn-Fudan,
- sensowna detekcja samochodow na VOC i zdjeciach customowych,
- formalne metryki `AP50/AP75/mAP`,
- integracja z Kaggle GPU przez Jupyter i MinIO.

## Co jest srednio dojrzale

- ocena modelu nadal opiera sie glownie o proxy metryki zamiast pelnego `mAP`,
- eksperymenty sa jeszcze stosunkowo krotkie na wiekszych datasetach,
- nowy `HybridEncoder` nie zostal jeszcze uczciwie sprawdzony w dluzszym treningu,
- repo nie ma jeszcze pelnej polityki wersjonowania checkpointow po zmianach architektury.

## Co nadal jest otwarte technicznie

- dluzsze treningi VOC i BDD po wiekszej liczbie epok,
- trening wieloklasowy zamiast tylko `car`,
- szersze benchmarkowanie `mAP`,
- stabilniejszy eksperymentalny protokol porownawczy,
- dalsze zblizanie necka i auxiliary path do referencji PaddleDet,
- opcjonalnie export i inferencja bardziej produktowa.

## Najwazniejszy stan modelu dzisiaj

Jesli patrzec pragmatycznie:

- najlepsza jakosc lokalizacji i czystosci boxow byla osiagnieta na Penn-Fudan,
- VOC `car` pokazal sensowna generalizacje,
- BDD100K vehicle-3 jest juz gotowy jako bardziej realistyczny kierunek danych drogowych,
- nowy `HybridEncoder` jest obiecujacy architektonicznie, ale wymaga dluzszego treningu, zanim bedzie mozna powiedziec, ze jest lepszy od poprzedniego prostszego necka.

## Rekomendowane kolejne kroki

Najbardziej sensowna kolejnosc na nastepne iteracje:

1. Ustabilizowac benchmark VOC `car` i BDD vehicle-3.
2. Przetrenowac nowy `HybridEncoder` dluzej na GPU.
3. Rozszerzyc trening na wiecej klas i bardziej realistyczne dane drogowe.
4. Dopiac bardziej produktowe eksporty i inferencje.
5. Dopiero potem dalej komplikowac architekture.

## Uwagi praktyczne

- stare checkpointy od prostszego necka nie sa kompatybilne z aktualnym kodem po zmianie `HybridEncoder`,
- to jest swiadoma cena szybkiego rozwoju architektury,
- w praktyce trzeba traktowac checkpointy jako powiazane z konkretna wersja kodu.
