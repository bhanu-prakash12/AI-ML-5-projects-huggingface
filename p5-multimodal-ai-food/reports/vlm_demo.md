# VLM demo (SmolVLM-500M, real run)

- {'n': 3, 'valid_json': 3, 'needed_retry': 0, 'load_s': 6.6, 'total_ms_median': 6548, 'total_ms_max': 7585, 'vlm_ms_median': 6487}

| image | CLIP top-1 | final dish | JSON ok | attempts | kcal/100g (USDA) | total ms | vlm ms |
|---|---|---|---|---|---|---|---|
| 0000.jpg | cheese_plate (1.0) | greek_salad | True | 1 | 48.0 | 7585 | 7395 |
| 0100.jpg | spaghetti_bolognese (0.9952) | spaghetti_bolognese | True | 1 | 90.0 | 6548 | 6487 |
| 0200.jpg | lasagna (0.1996) | lasagna | True | 1 | 139.0 | 5708 | 5672 |
