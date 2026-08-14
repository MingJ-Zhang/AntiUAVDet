# Prediction schema

Every model writes one JSON per test set:

```
predictions/<dataset>/<model>.json
predictions/<dataset>/<model>.meta.json
```

## `*.json`

```json
{
  "meta": {
    "model": "dino",
    "dataset": "fdd",
    "framework": "mmdet",
    "img_size": [768, 1344],
    "score_thr_stored": 0.001,
    "max_dets_per_image": 300
  },
  "images": [
    {"id": 1, "file_name": "000006.jpg", "width": 1920, "height": 1080}
  ],
  "predictions": [
    {
      "image_id": 1,
      "file_name": "000006.jpg",
      "category_id": 0,
      "bbox": [x, y, w, h],
      "score": 0.873,
      "area": 1234.5,
      "iscrowd": 0
    }
  ]
}
```

## Hard rules

1. **Coordinates = original-image pixels** (xywh), not the 768×1344 network input.
2. **`image_id` comes from the GT JSON** (never a file-traversal index).
3. **Store the full set**: `score >= 0.001`, top-300 per image. Filter later for visualization.
4. **`category_id` is always 0** (single-class). Using 1 zeroes out COCOeval.
5. **`images` must match GT**.

```bash
python tools/validate_pred.py predictions/<ds>/<model>.json data/gt/<ds>_test.json
python tools/run_metrics.py data/gt/<ds>_test.json predictions/<ds>/<model>.json \
    --out metrics/<ds>/<model>.json -m <name> -d <ds>
```

`validate_pred.py` must print `RESULT: PASS`.
