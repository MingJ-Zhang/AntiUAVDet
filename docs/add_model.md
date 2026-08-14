# Add a model

1. Put the source tree under `models/<name>/` (do not rewrite the trainer).
2. Append an entry to [`configs/model_zoo.yaml`](../configs/model_zoo.yaml):

```yaml
- id: mydet
  display: MyDet
  framework: mmdet   # mmdet | ultralytics | rt-detr | dfine | deim
  repo: models/mydet
  configs:
    fdd: configs/mydet_fdd.py
    antiuav: configs/mydet_antiuav.py
    dut_dve: configs/mydet_dut_dve.py
  infer_opts: {}
  params_M: 0.0
  gflops: 0.0
```

3. If the framework is new, add `tools/infer_<fw>.py` that writes a raw list
   `[{image_id, category_id: 0, bbox: [x,y,w,h], score}, ...]` in **original-image** pixels.
4. Point `ckpts` at `weights/<id>/<dataset>.pth` (files are git-ignored).
5. Run `python tests/test_layout.py` and a `--dry-run` benchmark.

Keep Ultralytics forks on separate `PYTHONPATH`s.
