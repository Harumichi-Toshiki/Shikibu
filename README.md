# Shikibu（式部）

Shikibu は、将棋の USI 探索エンジン **Nuenagon** が使う評価関数を学習するプログラムです。
評価関数は **RayPair** という特徴量と、NNUE に似た小さなニューラルネットワークで、教師の評価値と勝敗から学習します。

## 関係する開発物

| 名前 | 役割 | このリポジトリとの関係 |
|---|---|---|
| **Shikibu**（式部） | Nuenagon 用の評価関数の学習器（このリポジトリ） | — |
| **Nuenagon** | 独自の評価関数で動く αβ 探索の USI エンジン（C++） | Shikibu が書き出した重み（`.shk`）を読む。盤面・指し手生成は Dainagon のものを流用する |
| **Dainagon** | 深層学習（DL）と MCTS のエンジン | Nuenagon の盤面・指し手生成の元になる |
| **Dajo**（太政） | Dainagon と Nuenagon の合議を行う | 別リポジトリ。本リポジトリの対象外 |

教師データは、Ponkotsu WCSC35 から蒸留した PSV（1局面 40 バイト、約 19.6 億局面）です。水匠10 の PSV（約 1 億局面）もあります。

## できること

- PSV の統計、勝敗が記録されているかの判定、評価値の尺度（eval_scale）の推定
- 事前シャッフルとシャード分割（1GB 単位）
- Rust による特徴抽出（RayPair_v0 / v1）と、並列のバッチ生成（再開・キャッシュ対応）
- PyTorch による学習（30 分ごとの保存と再開、進捗の表示）
- 重みファイル `.shk` の書き出しと、書き出し後の読み直しによる確認
- エンジンに渡す前の量子化の確認、エンジン用の正解データの作成

## 構成

```
crates/shk_feat/     Rust：局面の解読、特徴抽出、ローダー、統計（Python からは shikibu_feat）
python/shikibu/      Python：ネットワーク、学習ループ、書き出し、量子化の確認
tools/               CLI：データ（shk_tools.py）、Colab の監視（shk_monitor.py）、正解データ（make_engine_golden.py）
configs/             学習の設定（TOML）。colab_v0.toml・colab_v1.toml が Colab 用
notebooks/           Colab のノートブック（学習・速度の確認）
tests/               受け入れテスト（73 件）
docs/                設計書と文書（下の一覧）
```

## すぐ始める

### Colab で学習する

1. Google Drive に学習データと出力先を用意する（`docs/GUIDE_COLAB.md` の 0）。
2. `notebooks/colab_train.ipynb` を開き、ランタイムを GPU（L4）にして、節 1 から順に実行する。
3. 学習の途中で、`mon.watch()` の表示を止めないでください（`docs/OPERATIONS.md` の 1）。

詳しい手順は **[docs/GUIDE_COLAB.md](docs/GUIDE_COLAB.md)** にあります。

### 手元の PC で動かす

Linux で動作を確かめています。仮想環境はリポジトリの外に作ることを勧めます。

```
python -m venv ~/venvs/shikibu && . ~/venvs/shikibu/bin/activate
pip install maturin pytest numpy cshogi torch
maturin develop --release -m crates/shk_feat/Cargo.toml   # Rust の拡張を入れる（仮想環境が必要）
python -m pytest -q tests                                 # 73 件が通ることを確認する
```

学習は、設定ファイルのパス（`train_glob`・`out_dir`・`cache_dir`）を手元のパスに書き換えたコピーで行います。

```
PYTHONPATH=python python -m shikibu.train --config my_local.toml            # 最初から
PYTHONPATH=python python -m shikibu.train --config my_local.toml --resume   # checkpoint.pt から続ける
PYTHONPATH=python python -m shikibu.export --checkpoint <out_dir>/checkpoint.pt --out model.shk
```

データの統計と eval_scale の推定は、次のコマンドです。

```
python tools/shk_tools.py stats data/*.psv
python tools/shk_tools.py estimate-scale data/*.psv
```

## 文書

| 文書 | 内容 | 読む人 |
|---|---|---|
| [docs/GUIDE_COLAB.md](docs/GUIDE_COLAB.md) | Colab で学習を回す手順（最初から） | 学習を回す人 |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | 運用の注意、切れたときの再開、よくあるエラー | 学習を回す人 |
| [docs/CONFIG.md](docs/CONFIG.md) | 設定ファイル（TOML）の全キー、既定値、検査 | 学習を回す人・開発者 |
| [docs/FORMATS.md](docs/FORMATS.md) | PSV、`.shk`、`checkpoint.pt`、`log.jsonl`、`status.json` の形式 | 開発者・エンジンの開発者 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | コードの構成と、データの流れ | 開発者 |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | 開発の進め方、変更禁止ファイル、テスト、マイルストーン、未解決の課題 | 開発者 |
| [docs/DESIGN.md](docs/DESIGN.md) | 設計書（凍結区画の第F部を含む。変更禁止） | 全員 |
| [docs/COLAB.md](docs/COLAB.md) | Colab で回すための方針（変更禁止） | 全員 |
| `REPORT_M1.md` 〜 `REPORT_M5.md` | マイルストーンごとの報告 | 開発者 |

## 変更禁止のファイル

`docs/DESIGN.md`、`docs/COLAB.md`、`tests/`、`reference/`、`perf/`、`PROTECTED.sha256`、`check_protected.py` は変更できません。
確認は次のコマンドで行い、`OK` と表示されることを確かめてください。

```
python check_protected.py
```

詳しくは [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) の 1 を参照してください。
