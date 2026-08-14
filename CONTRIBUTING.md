# Contributing

1. Put new detector code under `models/<name>/` without rewriting the original trainer.
2. Register it in `configs/model_zoo.yaml`.
3. Predictions must obey `tools/SCHEMA.md`.
4. Do not commit weights or dataset images.
5. Keep Ultralytics forks on separate `PYTHONPATH`s.

```bash
python tests/test_layout.py
```

Branches: `feat/<name>`, `fix/<scope>`, `docs/<scope>`.
