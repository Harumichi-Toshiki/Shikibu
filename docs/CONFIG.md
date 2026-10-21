# 設定ファイル（TOML）の全キー

学習器 `python -m shikibu.train --config <TOML>` が読む設定ファイルの一覧です。
既定値と検査は `python/shikibu/train.py` の `DEFAULTS`・`validate()` を読んで書きました。
設計書の節（§）は `docs/DESIGN.md` を指します。

- 未知のセクション・未知のキーがあると、`ValueError` で止まります（終了コード 1）。キーのつづりの間違いは、ここで分かります。
- 省略したキーには既定値が入ります。ただし既定値が「未指定」（空文字・0）のキーは、**必ず書きます**（下の表で「必須」と書いたもの）。
- 例：`configs/colab_v0.toml`（Colab 用の実物）。

---

## [data]：学習データ

| キー | 型 | 既定値 | 意味 | 関連 |
|---|---|---|---|---|
| `train_paths` | 文字列のリスト | `[]` | 学習用 PSV のパスの並び。`train_glob` が空のときだけ使う | §S3.2 |
| `train_glob` | 文字列 | `""` | 学習用 PSV のパスの glob。**空でなければ `train_paths` より優先**。並びは昇順（`sorted`） | §S3.2 |
| `val_path` | 文字列 | `""` | 検証用 PSV の1ファイル。空なら検証しない | §S6 検証 |
| `cache_dir` | 文字列 | `""` | 指定すると、学習用ファイルを Drive からここ（ローカル SSD）へ先読みコピーする。空ならコピーしない。**専用のフォルダにすること**（起動時と終了時に `shk_cache_` で始まるファイルを消す） | §S3.2 |
| `max_cached` | 整数 | `3` | `cache_dir` に同時に置くファイル数の目安。実際の上限は `max_cached + 1`（読んでいる1個を含む）。`cache_dir` を使うときは1以上 | §S3.2（「3〜4個まで」） |
| `shuffle_buffer` | 整数 | `0` | 読み込み時のシャッフル窓の局面数。0 でシャッフルなし。1局面 40 バイトなので、メモリは約 `shuffle_buffer × 40` バイト（2,000万なら約 800MB） | §S3.2 |
| `num_threads` | 整数 | `1` | 特徴抽出を行うスレッド数（作業係）。1以上 | §S9 |

- `train_glob` も `train_paths` も空だと、`ValueError`（「data.train_paths か data.train_glob を指定する」）。
- `train_glob` に当たるファイルが1つも無いと、`ValueError`（「学習データが1つも無い」）。
- `val_path` を書いたのに存在しないと、読み込みの時点で `ValueError` になる（事前の検査はしていない）。

---

## [model]：ネットワーク

| キー | 型 | 既定値 | 意味 | 関連 |
|---|---|---|---|---|
| `network` | 文字列 | `"RayPair_v0Network"` | `"RayPair_v0Network"` か `"RayPair_v1Network"`。特徴セットと層の大きさが決まる | §F3.1 |
| `ft_width` | 整数 | `128` | 第1層の幅 W。凍結仕様は 128。**検査はしていない**。ふつうは変えない | §F3.1 |

- `network` に登録されていない名前は `ValueError`（登録済みの名前が表示される）。
- `ft_width` を 128 以外にした重みは、学習はできても §F4 の凍結仕様（W = 128）から外れる。ヘッダには W が書かれるので、エンジンが対応していなければ読み込みを拒否する（§E2.3・§F4.3）。

---

## [train]：学習

| キー | 型 | 既定値 | 意味 | 関連 |
|---|---|---|---|---|
| `batch_size` | 整数 | `16384` | 1回の更新に使う局面数。1以上 | §S6（既定 16,384）。Colab は 65,536 |
| `lr` | 実数 | `1e-3` | 学習率の最大値（ウォームアップ後の上限） | §S6 |
| `weight_decay` | 実数 | `0.0` | AdamW の重み減衰。`l1_weight`・`l2_weight`・`out_weight` だけに掛かる（`ft` と bias には掛からない） | §S6 |
| `warmup_steps` | 整数 | `0` | ウォームアップの更新回数。0 なら即座に最大学習率 | §S6 |
| `total_steps` | 整数 | 未指定（0） | **必須・1以上**。学習率のコサイン減衰の長さ（更新回数）。`max_steps` とは別 | §S6 |
| `max_steps` | 整数 | 未指定（0） | **必須・1以上**。**通算**の更新回数の上限。この値に達したら保存して終わる | §S6 |
| `min_lr_ratio` | 実数 | `0.0` | 減衰の最後の学習率（`lr` に対する比）。0〜1 の値を想定（検査はしていない） | §S6 |
| `lam` | 実数 | `0.75` | 目標 `t = λ·sigmoid(score/eval_scale) + (1−λ)·result` の λ。**0〜1** | §S6 |
| `eval_scale` | 文字列または正の数 | `"auto"` | `"auto"` なら勝敗から推定する。数値なら、その値を使う（推定はしない） | §S4・§F5 |
| `scale_sample` | 整数 | `10000000` | 勝敗の確認と eval_scale の推定に使う局面数の上限。全学習ファイルから、ファイルの大きさに比例した数の連続ブロック（1ブロック 1万局面）を取る。局面数がこの値以下なら全部使う | §S4 |
| `allow_missing_results` | 真偽値 | `false` | 勝敗が欠損しているデータを使うことを許すか。下の「勝敗欠損」を参照 | §S4-2 |
| `seed` | 整数 | `0` | 重みの初期化・ファイルの並び・シャッフル・検査用の局面の抜き出し、すべての乱数の種 | — |
| `device` | 文字列 | `"auto"` | `"auto"`（CUDA があれば使う）、`"cpu"`、`"cuda"` など | — |
| `torch_threads` | 整数 | `1` | PyTorch の CPU スレッド数 | — |
| `deterministic` | 真偽値 | `false` | `true` で決定的な計算を要求する（警告付き）。CPU・`torch_threads = 1`・`shuffle_buffer = 0` のとき、通しの学習と途中再開の結果がビット単位で一致することを確かめている | M3 テスト |
| `save_every_steps` | 整数 | `0` | この更新回数ごとに保存する。0 なら時間だけで保存する | §S6 保存と再開 |
| `save_every_minutes` | 実数 | `30.0` | この分数ごとに保存する。**0 にしてはいけない**（下の注意） | §S6 保存と再開 |
| `val_every_steps` | 整数 | `0` | この更新回数ごとに検証する。0 で検証しない。`val_path` が無いと何もしない | §S6 検証 |
| `val_positions` | 整数 | `100000` | 検証に使う `val_path` の先頭の局面数。`val_path` があるとき 1以上 | §S6 検証 |
| `log_every_steps` | 整数 | `100` | `log.jsonl` の train 行を書く間隔（更新回数）。**0 にしてはいけない**（下の注意） | — |
| `progress_every_sec` | 実数 | `30.0` | 画面に進捗を出し、`status.json` を書く間隔（秒）。0 なら `log_every_steps` ごと | — |

### 勝敗欠損（`result_missing`）の扱い

勝敗欠損とは、データの `game_result` の 95% 超が 0（引き分け）であることを指します（§S4-2）。
判定は、学習開始時に抜き出した局面（`scale_sample` 以下）で行います。

| `allow_missing_results` | `lam` | `eval_scale` | 結果 |
|---|---|---|---|
| `false` | 何でもよい | 何でもよい | **エラーで止まる**（「勝敗が欠損している」） |
| `true` | 1.0 **以外** | 何でもよい | エラー（「lam = 1.0 が必要」） |
| `true` | 1.0 | `"auto"` | エラー（「eval_scale を数値で指定する」） |
| `true` | 1.0 | 数値 | 学習する。`eval_scale_source` は `"config"` |

この4行は、欠損データを全部 0 にした小さなデータで、実際に実行して確かめました。
勝敗欠損のときは、`lam` を 1.0 にしないと、教師は「手番側の勝率 0.5 に近づけ続ける」形になり、評価値の尺度を決められません。
`lam` の既定値は 0.75 なので、欠損データでは必ず `lam = 1.0` を明示します。

この判定は**新しく学習を始めるときだけ**行われます。`--resume` のときは行われません（下の「再開」を参照）。

---

## [output]：出力

| キー | 型 | 既定値 | 意味 |
|---|---|---|---|
| `out_dir` | 文字列 | 未指定（空） | **必須**。出力フォルダ。無ければ作る。中に `checkpoint.pt`・`log.jsonl`・`status.json` を書く |

- `checkpoint.pt` があるのに `--resume` を付けないと、上書きせずに `FileExistsError`（「続けるなら --resume を付ける」）になり、終了コード 1 で止まります。
- `--resume` を付けて `checkpoint.pt` が無いときは、警告を出して最初から学習します。

---

## Colab 用の設定（`configs/colab_v0.toml`・`configs/colab_v1.toml`）

2つの違いは `model.network` と `output.out_dir` だけです。残りは同じ値です。

| キー | 値 | 理由（ファイル冒頭のコメントより） |
|---|---|---|
| `data.train_glob` | `…/PonkotsuWCSC35_unique_*.psv`（Drive） | 学習用の PSV（約19.6億局面） |
| `data.val_path` | `""` | 検証用の PSV が今の配置には無い |
| `data.cache_dir` | `/content/shk_cache` | Colab のローカル SSD |
| `data.max_cached` | 3 | |
| `data.shuffle_buffer` | 20000000 | 約 800MB のメモリ |
| `data.num_threads` | 11 | L4 ランタイムの論理コアに合わせる |
| `train.batch_size` | 65536 | |
| `train.lr` / `weight_decay` / `warmup_steps` | 1e-3 / 1e-4 / 1000 | 初期値。実測のあとで調整する |
| `train.total_steps` / `max_steps` | 179000 / 179000 | 65,536 × 179,000 ≈ 117億局面（約6周） |
| `train.min_lr_ratio` | 0.01 | |
| `train.lam` | 0.75 | |
| `train.eval_scale` | `"auto"` | |
| `train.scale_sample` | 10000000 | |
| `train.progress_every_sec` | 30.0 | |
| `output.out_dir` | `/content/drive/MyDrive/shikibu/runs/colab_v0`（v1 は `colab_v1`） | Drive に保存する |

ノートブックは、`max_steps` の行だけを書き換えて `/content/run_config.toml` を作ります（リポジトリの設定は変えません）。

---

## 再開（`--resume`）のときの規則

`checkpoint.pt` から続けるとき、コードは次のように動きます。

- **checkpoint の値が優先されるもの**：`eval_scale`（と由来）、ファイルの並び（`file_perm`）、読み込み位置、乱数の状態、モデル・オプティマイザ。
- **設定ファイルの値をそのまま使うもの**：`batch_size`、`lr`、`total_steps`、`max_steps`、`shuffle_buffer`、`cache_dir` など。
- **確認されるもの**：`network` と `lam` が checkpoint と違えば `ValueError`。`file_perm` とファイル数が違えば `ValueError`。
- **確認されないもの**：`eval_scale` を設定で変えても無視される（checkpoint の値を使う）。勝敗欠損の判定もやり直さない。
- `max_steps` は**通算**なので、再開のたびに大きくします（同じ値のまま再開すると、何もせずに終わります）。

注意：
- `total_steps` を途中で変えると、学習率のコサイン減衰の形が変わります。
- `train_glob` を変えてファイルの数が同じでも中身が違うと、ファイル数の検査では見つかりません（コードの読みから。確かめてはいない）。

---

## 注意（設定の落とし穴）

| 設定 | 起こること | 根拠 |
|---|---|---|
| `log_every_steps = 0` | `ZeroDivisionError` のトレースバックが出て終了（終了コード 1）。**検査はされていない** | 確認済み（`log_every_steps = 0` で実行） |
| `save_every_minutes = 0` | 時間による保存の条件が毎ステップ真になり、**毎ステップ保存**される（0 で「保存しない」ではない） | コードの読み（実行はしていない） |
| `save_every_steps = 0` | 更新回数ごとの保存は行わない。時間による保存（`save_every_minutes`）と終了時の保存だけ | コードの読み |
| `shuffle_buffer > 0` で再開 | 窓の中にあった局面は失われる（重複はしない） | `shk_feat` の仕様（REPORT_M2） |
| `device = "cuda"`（GPU の経路） | 固定メモリ経由の転送と、決定的な計算の設定は、Colab の GPU でしか確かめていない。この環境（CPU のみ）では動かせない | REPORT_M3・M4 |

