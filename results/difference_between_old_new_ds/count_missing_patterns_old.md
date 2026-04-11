ndreea@DESKTOP-F4GTCRF:~/bachelor-thesis$ python3 scripts/check_diff_ds_versions.py  --old old-metadata-ds/fracture_assembly_metadata_2_20_artifact.val.txt --new new-metadata-ds/fracture_assembly_metadata_2_20_artifact.val.txt

```python
Old dataset: 3651 fracture patterns
New dataset: 3651 fracture patterns
Difference:  0 fracture patterns

Meshes in old: 3651
Meshes in new: 3651
Meshes missing from old: 0
```

ndreea@DESKTOP-F4GTCRF:~/bachelor-thesis$ python3 scripts/check_diff_ds_versions.py  --old old-metadata-ds/fracture_assembly_metadata_2_20_everyday.val.txt --new new-metadata-ds/fracture_assembly_metadata_2_20_everyday.val.txt

```python
Old dataset: 4805 fracture patterns
New dataset: 7679 fracture patterns
Difference:  2874 fracture patterns

Meshes in old: 91
Meshes in new: 91
Meshes missing from old: 0

--- Fracture patterns in new but NOT in old: 2874 ---
  Vase: 820/1741 patterns missing (47.1%)
  Bottle: 620/1254 patterns missing (49.4%)
  ToyFigure: 338/847 patterns missing (39.9%)
  Mug: 232/698 patterns missing (33.2%)
  Mirror: 229/675 patterns missing (33.9%)
  Bowl: 218/500 patterns missing (43.6%)
  Cup: 102/425 patterns missing (24.0%)
  Plate: 73/213 patterns missing (34.3%)
  WineBottle: 61/264 patterns missing (23.1%)
  DrinkingUtensil: 35/159 patterns missing (22.0%)
  PillBottle: 31/172 patterns missing (18.0%)
  Teapot: 17/92 patterns missing (18.5%)
  Cookie: 16/82 patterns missing (19.5%)
  DrinkBottle: 14/86 patterns missing (16.3%)
  BeerBottle: 14/93 patterns missing (15.1%)
  WineGlass: 12/93 patterns missing (12.9%)
  Teacup: 11/73 patterns missing (15.1%)
  Statue: 11/99 patterns missing (11.1%)
  Ring: 11/79 patterns missing (13.9%)
  Spoon: 9/34 patterns missing (26.5%)
```

ndreea@DESKTOP-F4GTCRF:~/bachelor-thesis$ python3 scripts/check_diff_ds_versions.py  --old old-metadata-ds/fracture_assembly_metadata_2_20_everyday.train.txt --new new-metadata-ds/fracture_assembly_metadata_2_20_everyday.train.txt

```python
Old dataset: 21494 fracture patterns
New dataset: 34075 fracture patterns
Difference:  12581 fracture patterns

Meshes in old: 405
Meshes in new: 405
Meshes missing from old: 0

--- Fracture patterns in new but NOT in old: 12583 ---
  Vase: 3312/6719 patterns missing (49.3%)
  Bottle: 2391/4943 patterns missing (48.4%)
  ToyFigure: 1494/3762 patterns missing (39.7%)
  Mug: 1105/2910 patterns missing (38.0%)
  Mirror: 1103/3024 patterns missing (36.5%)
  Bowl: 844/2223 patterns missing (38.0%)
  Cup: 602/1932 patterns missing (31.2%)
  Plate: 329/1002 patterns missing (32.8%)
  WineBottle: 290/1315 patterns missing (22.1%)
  DrinkingUtensil: 203/896 patterns missing (22.7%)
  PillBottle: 128/652 patterns missing (19.6%)
  Teapot: 123/754 patterns missing (16.3%)
  WineGlass: 118/717 patterns missing (16.5%)
  Spoon: 114/403 patterns missing (28.3%)
  BeerBottle: 112/704 patterns missing (15.9%)
  Ring: 95/587 patterns missing (16.2%)
  DrinkBottle: 79/528 patterns missing (15.0%)
  Teacup: 63/491 patterns missing (12.8%)
  Cookie: 41/238 patterns missing (17.2%)
  Statue: 37/275 patterns missing (13.5%)


--- Fracture patterns in old but NOT in new: 2 ---
  DrinkingUtensil: 1
  Spoon: 1
```