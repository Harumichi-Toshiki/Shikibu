# Shikibu
This is the trainer for Nuenagon.

Shikibu learns the evaluation function used by Nuenagon, a USI alpha-beta search engine. Feature extraction and data handling are written in Rust; training is done in Python with PyTorch.

設計書は `docs/DESIGN.md`。

## M1：特徴抽出（`shikibu_feat`）

`crates/shk_feat/` の Rust クレートを pyo3 + maturin で Python モジュール `shikibu_feat` としてビルドする。

### ビルド

```
python -m venv .venv && . .venv/bin/activate        # maturin develop には仮想環境が必要
pip install cshogi pytest numpy maturin
maturin develop --release -m crates/shk_feat/Cargo.toml
```

### テスト

```
python -m pytest -q tests
```

### API

| 関数 | 内容 |
|---|---|
| `num_features(version) -> int` | 特徴量の次元数（v0 = 15000、v1 = 115352） |
| `extract_sfen(sfen, perspective, version) -> list[int]` | SFEN から特徴（昇順・重複込み） |
| `decode_psv(rec: bytes) -> (sfen, score, move, gamePly, game_result)` | PSV（40バイト）を読む |
| `extract_psv(rec: bytes, perspective, version) -> list[int]` | PSV から特徴（昇順・重複込み） |

`perspective`：0 = 先手視点、1 = 後手視点。`version`：0 = RayPair_v0、1 = RayPair_v1。

### 変更禁止

`docs/`、`reference/`、`tests/`、`PROTECTED.sha256`、`check_protected.py` は変更禁止。確認は `python check_protected.py`。

## M3：学習ループ（`shikibu`）

`python/shikibu/` の Python パッケージ。`shikibu_feat`（M1・M2）でデータを読み、PyTorch で学習する。

### 学習

```
PYTHONPATH=python python -m shikibu.train --config configs/colab_v0.toml            # 最初から
PYTHONPATH=python python -m shikibu.train --config configs/colab_v0.toml --resume   # checkpoint.pt から続ける
```

- 設定は TOML（`configs/colab_v0.toml`、`configs/colab_v1.toml` が例）。
- 出力は `output.out_dir` に `checkpoint.pt`（最新の保存）と `log.jsonl`（1行1 JSON）。
- Colab での実行手順は `notebooks/colab_train.ipynb`。

### 変更禁止

`tests/m3_data.py` と `tests/test_m3_acceptance.py` は変更禁止（キットで配置）。
