# Architektura

## Aktualny przeplyw modelu

Obecna sciezka modelu ma postac:

`ResNet backbone -> HybridEncoder -> QuerySelection -> RTDETRDecoder -> PostProcessor`

Trening rozszerza to o dodatkowe elementy:

- `contrastive denoising`,
- galezie `o2o` i `o2m`,
- `auxiliary dense head` w stylu PP-YOLOE.

## Backbone

Backbone jest oparty o `torchvision` `ResNet`:

- `resnet18`,
- `resnet34`,
- `resnet50`,
- `resnet101`.

Backbone zwraca trzy poziomy cech o stride:

- `8`,
- `16`,
- `32`.

## HybridEncoder

Aktualny `HybridEncoder` jest juz wyraznie blizej referencji niz poczatkowy prosty neck.

Zaimplementowane elementy:

- projekcja kanalow do wspolnego `hidden_dim`,
- transformer encoder na wybranych poziomach cech,
- 2D sinus-cosinus positional embedding,
- top-down fusion w stylu FPN,
- bottom-up fusion w stylu PAN,
- blokowe laczenie cech typu `CSPRep`.

To jest istotna zmiana wzgledem pierwszej wersji, ktora miala jedynie:

- proste dodawanie map cech,
- zwykle bloki konwolucyjne,
- brak transformerowego wzbogacenia najwyzszego poziomu.

## Query Selection

`QuerySelection` odpowiada za przygotowanie wejscia do dekodera.

Obecne elementy:

- anchor-like proposals generowane z encoder memory,
- `valid_mask` dla pozycji spoza poprawnego zakresu,
- wiele grup query,
- `learnt_init_query`,
- rozdzielenie `o2o` i `o2m`,
- training-only denoising queries.

To jest juz sensownie zblizone do referencyjnego `_get_decoder_input` z `ppdet`, chociaz nadal nie jest to pelny port bit-po-bicie.

## Decoder

Decoder ma:

- deformable cross-attention,
- masked self-attention dla grup query,
- iteracyjny refinement boxow,
- osobne heady klas i bbox per layer.

Wczesniej decoder byl bardziej uproszczony. Zostal poprawiony tak, aby:

- lepiej obslugiwac reference points,
- sensowniej inicjalizowac attention,
- dawac bardziej stabilny refinement boxow.

## Denoising

W modelu zaimplementowano contrastive denoising dla treningu:

- noisy labels,
- noisy boxes,
- metadata `dn_meta`,
- splitting outputs na czesc denoising i matching,
- osobny loss dla denoising bez Hungarian matchingu.

To jest jeden z kluczowych elementow, ktory zblizyl pipeline do RT-DETRv3 i poprawil stabilnosc treningu.

## Auxiliary O2M Head

Obecna treningowa galaz pomocnicza jest wzorowana na PP-YOLOE:

- osobna galaz `cls`,
- osobna galaz `reg`,
- assignment typu `ATSS -> TaskAligned`,
- `Varifocal Loss`,
- `GIoU`,
- `DFL`.

Wazne:

- auxiliary head jest `training-only`,
- inferencja publiczna nadal korzysta z glownej sciezki DETR,
- to nie jest pelny port calego `PPYOLOEHead.post_process()`.

## Postprocess

Postprocess modelu obejmuje:

- `top-k` selection,
- `score threshold`,
- `NMS`.

Po strojeniach praktyczne domyslne wartosci okazaly sie sensowne:

- `score_threshold = 0.18`,
- `nms_threshold = 0.25`.

To bylo potrzebne, bo we wczesniejszych wersjach model potrafil zwracac zduplikowane boxy.

## Co nadal odbiega od referencji

Najwazniejsze braki wzgledem referencji RT-DETRv3:

- brak pelnego portu wszystkich szczegolow `PPYOLOEHead`,
- brak pelnej parity calego ekosystemu PaddleDet,
- brak klasycznej oceny `mAP` w repo,
- brak kompatybilnosci starych checkpointow po wiekszych zmianach architektury,
- brak szerokiego benchmarku na wiecej niz jednym lub dwoch datasetach.
