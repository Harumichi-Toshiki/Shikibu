# Shikibu（式部）

Shikibu は、将棋の USI 探索エンジン **Nuenagon** が使う評価関数を学習するプログラムです。
評価関数は **RayPair** という特徴量と、NNUE に似た小さなニューラルネットワークで、教師データの評価値と勝敗から学習します。
どの教師データでも、どこに置いても使えるように作っています。学習データ・出力先・Drive のフォルダ構成は、利用者が決めます。

## 関係する開発物

| 名前 | 役割 | このリポジトリとの関係 |
|---|---|---|
| **Shikibu**（式部） | Nuenagon 用の評価関数の学習器（このリポジトリ） | — |
| **Nuenagon** | 独自の評価関数で動く αβ 探索の USI エンジン（C++） | Shikibu が書き出した重み（`.shk`）を読む。盤面・指し手生成は Dainagon のものを流用する |
| **Dainagon** | 深層学習（DL）と MCTS のエンジン | Nuenagon の盤面・指し手生成の元になる |
| **Dajo**（太政） | Dainagon と Nuenagon の合議を行う | 別リポジトリ。本リポジトリの対象外 |

教師データは、評価値と勝敗を持つ PSV（1局面 40 バイト）です。どの教師エンジンのデータでも、同じ形式なら使えます。
評価値の尺度が違う教師のデータを混ぜるときは注意が要ります（`docs/OPERATIONS.md` の 5）。

## できること

- PSV の統計、勝敗が記録されているかの判定、評価値の尺度（eval_scale）の推定
- 事前シャッフルとシャード分割
- Rust による特徴抽出（RayPair_v0 / v1、比較実験用の HalfKP と、その因数分解つき）と、並列のバッチ生成（再開・キャッシュ対応）
- PyTorch による学習（定期保存と再開、進捗の表示、同じ出力フォルダでの二重起動の防止）
- 重みファイルの書き出しと、書き出し後の読み直しによる確認（RayPair は `.shk`、HalfKP はやねうら王の `nn.bin`）
- エンジンに渡す前の量子化の確認、エンジン用の正解データの作成

## 構成

```
crates/shk_feat/     Rust：局面の解読、特徴抽出、ローダー、統計（Python からは shikibu_feat）
python/shikibu/      Python：ネットワーク、学習ループ、ロック、書き出し、量子化の確認
tools/               CLI：データ（shk_tools.py）、Colab の監視（shk_monitor.py）、正解データ（make_engine_golden.py）
configs/             学習の設定のひな形（TOML）。<…> の目印を自分の環境の値に書き換えて使う
notebooks/           Colab のノートブック（学習・速度の確認）。設定セルの変数を書き換えて使う
tests/               受け入れテスト
docs/                設計書と文書（下の一覧）
```

## すぐ始める

### Colab で学習する

1. 学習データ（PSV）と、出力先のフォルダを Google Drive に用意する。
2. `notebooks/colab_train.ipynb` を開き、ランタイムを GPU にする。
3. 節 2 の**設定セル**の `<…>` の目印を、自分の環境の値に書き換える。
4. 節 1 から順に実行する。学習の途中で、`mon.watch()` の表示を止めないでください（`docs/OPERATIONS.md` の 1）。

詳しい手順は **[docs/GUIDE_COLAB.md](docs/GUIDE_COLAB.md)** にあります。

### 手元の PC で動かす

Linux で動作を確かめています。仮想環境はリポジトリの外に作ることを勧めます。

```
python -m venv ~/venvs/shikibu && . ~/venvs/shikibu/bin/activate
pip install maturin pytest numpy cshogi torch
maturin develop --release -m crates/shk_feat/Cargo.toml   # Rust の拡張を入れる（仮想環境が必要）
python -m pytest -q tests                                 # 全件が通ることを確認する
```

学習は、ひな形の設定をコピーして、`<…>` の目印を手元の値に書き換えたファイルで行います。

```
cp configs/colab_v0.toml my_local.toml   # 書き換えてから使う。<…> が残っていると、学習器は止まる
PYTHONPATH=python python -m shikibu.train --config my_local.toml            # 最初から
PYTHONPATH=python python -m shikibu.train --config my_local.toml --resume   # checkpoint.pt から続ける
PYTHONPATH=python python -m shikibu.export --checkpoint <出力フォルダ>/checkpoint.pt --out model.shk
```

データの統計と eval_scale の推定は、次のコマンドです。

```
python tools/shk_tools.py stats <PSV のパス>...
python tools/shk_tools.py estimate-scale <PSV のパス>...
```

### HalfKP を学習して、やねうら王・Nuenagon で使う（比較実験用）

同じ学習器で、やねうら王の halfkp_256x2-32-32 と同じ HalfKP も学習できます。RayPair との比較に使います。

```
cp configs/colab_halfkp.toml my_halfkp.toml   # <…> を書き換える。network は既定で HalfKPf_256Network（因数分解つき）
PYTHONPATH=python python -m shikibu.train --config my_halfkp.toml
PYTHONPATH=python python -m shikibu.export --checkpoint <出力フォルダ>/checkpoint.pt --out nn.bin
# 標準出力に「FV_SCALE=<整数>」が出る
```

- 書き出した `nn.bin` を、やねうら王・Nuenagon の評価関数ファイルとして読ませます。
- 同じ行に出た `FV_SCALE=<整数>` を、エンジンの USI オプション `FV_SCALE` に設定してください（省略すると既定値の 16 になり、評価値の尺度が合いません）。
- この値で、エンジンの cp は学習の値 z × eval_scale にほぼ一致します。
- 因数分解なしの `HalfKP_256Network` も、`network` を書き換えれば選べます。`--fv-scale <整数>` で FV_SCALE を指定することもできます。

## 文書

| 文書 | 内容 | 読む人 |
|---|---|---|
| [docs/GUIDE_COLAB.md](docs/GUIDE_COLAB.md) | Colab で学習を回す手順（最初から） | 学習を回す人 |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | 運用の注意、切れたときの再開、二重起動の防止、よくあるエラー | 学習を回す人 |
| [docs/CONFIG.md](docs/CONFIG.md) | 設定ファイル（TOML）の全キー、既定値、検査 | 学習を回す人・開発者 |
| [docs/FORMATS.md](docs/FORMATS.md) | PSV、`.shk`、`nn.bin`（HalfKP）、`checkpoint.pt`、`log.jsonl`、`status.json`、`train.lock` の形式 | 開発者・エンジンの開発者 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | コードの構成と、データの流れ | 開発者 |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | 開発の進め方、変更禁止ファイル、テスト、マイルストーン、未解決の課題 | 開発者 |
| [docs/DESIGN.md](docs/DESIGN.md) | 設計書（凍結区画の第F部を含む。変更禁止） | 全員 |
| [docs/COLAB.md](docs/COLAB.md) | Colab で回すための方針（変更禁止） | 全員 |
| `REPORT_M1.md` 〜 `REPORT_M51.md` | マイルストーンごとの報告 | 開発者 |

## 変更禁止のファイル

`docs/DESIGN.md`、`docs/COLAB.md`、`tests/`（M5.1 の変更を含む）、`reference/`、`perf/`、`PROTECTED.sha256`、`check_protected.py` は変更できません。
確認は次のコマンドで行い、`OK` と表示されることを確かめてください。

```
python check_protected.py
```

詳しくは [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) の 1 を参照してください。
