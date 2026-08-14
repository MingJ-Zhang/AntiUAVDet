# Datasets

Images are **not** redistributed. Apply to the original authors, then point `ANTIUAVDET_DATA` at a folder that contains the three roots below.

Test-set **GT snapshots** ship in [`data/gt/`](../data/gt/) so COCOeval is bit-stable.

| Key | Folder under `$ANTIUAVDET_DATA` | Test images | Test image dir | GT snapshot |
|---|---|---:|---|---|
| `fdd` | `FDD` | 1150 | `test/foggy/` | `data/gt/fdd_test.json` |
| `antiuav` | `antiuav` | 2200 | `test/test/img/` | `data/gt/antiuav_test.json` |
| `dut_dve` | `DUT-Dve_YOLO_Benchmark_View` | 1200 | `images/test/` | `data/gt/dut_dve_test.json` |

All three are **single class** (`category_id = 0`). Input size is `768 (H) × 1344 (W)`, `keep_ratio=False`.

AntiUAV test GT has occasional noise (missed / mislabelled target boards). Confirm frames before qualitative figures.

See also [`configs/datasets/`](../configs/datasets/).
