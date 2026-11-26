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
| `check_mode` | 文字列 | `"sample"` | 開始前の勝敗の確認に使う局面。`"sample"`（従来。`scale_sample` 以下の抜き出し）・`"probe"`（各ファイルの先頭・中央・末尾から `probe_records` 件ずつ。ファイルの大きさを先に検査する）・`"full"`（**全件**を読む。時間は全データに比例。数時間かかりうる）。M9 | M9（§S4） |
| `probe_records` | 整数 | `3000` | `check_mode = "probe"` のとき、1 ファイルから読む局面数（3 ブロックに分ける）。3 以上 | M9 |
| `check_cache` | 真偽値 | `false` | `true` なら、勝敗の確認と eval_scale の推定の結果を、データのフォルダの `shikibu_check_cache.json` に保存し、同じデータ（ファイルの名前・大きさ・更新時刻）と同じ確認の設定なら、次回から読むだけにする。変わったら作り直す。書けなくても学習は続ける | M9 |
| `shuffle_warm` | 整数 | `0` | 0 なら従来どおり（シャッフルバッファを満たしてから学習を始める）。`shuffle_buffer` 以下の値なら、その件数から学習を始め、バッチを出すたびに `batch_size` ずつバッファを増やして `shuffle_buffer` まで広げる（待ち時間の短縮）。学習の初めの数百ステップの並びが変わる | M9 |
- `train_glob` も `train_paths` も空だと、`ValueError`（「data.train_paths か data.train_glob を指定する」）。
- `train_glob` に当たるファイルが1つも無いと、`ValueError`（「学習データが1つも無い」）。
- `train.log_every_steps` は 1 以上、`train.save_every_steps`・`train.val_every_steps` は 0 以上（0 は無効）、`train.progress_every_sec` は 0 以上。違反すると、キーの名前を含むエラーで終わる（M5.1）。
- パスに `<…>` の目印が残っていると、「設定のひな形のまま」のエラーで、何も書かずに終わる（M5.1）。
- `val_path` を書いたのに存在しないと、読み込みの時点で `ValueError` になる（事前の検査はしていない）。

---

## [model]：ネットワーク

| キー | 型 | 既定値 | 意味 | 関連 |
|---|---|---|---|---|
| `network` | 文字列 | `"RayPair_v0Network"` | 従来の名前（`"RayPair_v0Network"`・`"RayPair_v1Network"`・`"HalfKP_256Network"`・`"HalfKPf_256Network"`）か、**構造の名前**（例：`"halfkahm2_512x2-16-32_f"`）。構造の名前は `<特徴>_<W>x2-<L1>-<L2>[_f][_psqt][_k3k3]` の形で、特徴・幅・層・オプションを決める。読み方は [README](../README.md) の「構造の名前の読み方」 | §F3.1・§S11・§S12 |
| `ft_width` | 整数 | `0` | 第1層の幅 W。**`0` なら構造の名前の W（従来の名前なら 128 か 256）に合わせる**。構造の名前で W を書いたときは、`ft_width` は 0 か同じ値でなければならない（違えば `ValueError`）。RayPair はふつう変えない | §F3.1・§S11・§S12 |

- `network` が従来の名前でも構造の名前でもなければ `ValueError`（構造の名前の形、登録済みの名前、特徴・W・L1・L2・オプションの範囲を日本語で示す）。
- 構造の名前の W は 32 の倍数（32〜8192）、L1・L2 は 8 の倍数（8〜1024）。オプションは `f`（halfk… だけ）・`psqt`・`k3k3`。同じオプションを 2 回書くと誤り。
- checkpoint の `network` には、構造の名前を**正規の形**（小文字、オプションは `f` → `psqt` → `k3k3` の順）で書く。従来の名前は、そのまま書く。
- 検証データ（`data.val_path`）は、`data.train_glob` や `data.train_paths` に含まれていても学習から除かれる（§S12.5）。
- `ft_width` を 128 以外にした RayPair の重みは、学習はできても §F4 の凍結仕様（W = 128）から外れる。`.shk` のヘッダには W が書かれるので、エンジンが対応していなければ読み込みを拒否する（§E2.3・§F4.3）。

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
| `ft_forward` | 文字列 | `"dense"` | 第1層（特徴量の変換）の順伝播。`"dense"`（従来。因数分解つきでは W_eff を毎回密に作る）・`"sparse"`（有効な特徴の行だけを足す。因数分解は行の番号の付け替えで行い、W_eff を作らない）。結果は数学的に同じ（丸めの差だけ）。M9 | M9 |
| `ft_grad` | 文字列 | `"dense"` | 第1層の勾配。`"sparse"` は使われた行だけの疎な勾配。**`ft_optimizer = "lazy_adam"` が必要**。因数分解つきの網では `ft_forward = "sparse"` も必要。M9 | M9 |
| `ft_optimizer` | 文字列 | `"adamw"` | `"adamw"`（従来）・`"lazy_adam"`（第1層の重みだけ、そのステップで使われた行だけを更新する Adam。**AdamW と数学的に同じではない**。下の「第1層の遅延型 Adam」を参照）。2 層目以降は AdamW のまま。M9 | M9 |
| `fast` | 真偽値 | `false` | `true` で、CUDA のとき 2 層目以降を bf16 の自動混合精度で計算する。第1層の重みと累積は fp32 のまま。書き出しは fp32 の重みから行う。GPU が無いときは無視する。M9 | M9 |
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

## ひな形（`configs/colab_v0.toml`・`configs/colab_v1.toml`・`configs/colab_halfkp.toml`・`configs/colab_lab.toml`）

リポジトリの設定は、利用者が使うための**ひな形**です。データの場所と出力先は、`<…>` の目印で書いてあります。

| キー | ひな形の値 | 書き換え |
|---|---|---|
| `data.train_glob` | `"/content/drive/MyDrive/<学習データのフォルダ>/*.psv"` | 学習用の PSV のパス（glob） |
| `output.out_dir` | `"/content/drive/MyDrive/<出力のフォルダ>/runs/colab_v0"` | 出力フォルダ |
| `data.val_path` | `""` | 検証用の PSV を用意したら指定する（空なら検証しない） |
| `data.cache_dir` | `"/content/shk_cache"` | そのままでよい（Colab のローカル） |
| `model.network` | `"RayPair_v0Network"`（v1 は `RayPair_v1Network`。`colab_halfkp.toml` は `"HalfKPf_256Network"`、`ft_width = 256`。`colab_lab.toml` は構造の名前 `"halfkp_256x2-32-32_f"`、`ft_width = 0`） | 特徴セットに合わせる |

- 学習の値（`batch_size`・`lr`・`total_steps`・`max_steps` など）は、データに合わせて調整してよい。
- **`<…>` の目印が残っていると、学習器は「設定のひな形のまま」と表示して、何も書かずに終了コード 1 で止まります**（`train_paths`・`train_glob`・`val_path`・`cache_dir`・`out_dir` を見る）。
- ノートブックは、設定セルの変数でこの目印を置き換えて、出力フォルダに `config.toml` を書き出してから使います。

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

### 第1層の遅延型 Adam（`ft_optimizer = "lazy_adam"`）

- 数式は Adam と同じ。ただし、そのステップで使われなかった行（特徴）は、重み・1次・2次のモーメントのどれも動かさない。
- 密な AdamW は、使われなかった行も 1 次モーメントの慣性で動き、2 次モーメントは減衰する。**よって、同じ設定でも損失の推移は完全には一致しない**。遅延型は「同程度に下がる」ことを確かめる（M9 のテスト）。
- 重みの減衰はかけない（今の設定でも第1層の重みは減衰の対象外なので、減衰の扱いは同じ）。
- 1 ステップ目だけは、密な AdamW と完全に一致する（使われなかった行の 1 次モーメントが 0 のため）。
- checkpoint の optimizer の形が違うので、`ft_optimizer` を途中で変えて `--resume` すると止まる。


| 設定 | 起こること | 根拠 |
|---|---|---|
| `log_every_steps = 0` | M5.1 で検査するようにした。0 は「train.log_every_steps は 1 以上」の設定エラーになる（旧い版は `ZeroDivisionError`） | 確かめた（テスト） |
| `save_every_minutes = 0` | 時間による保存の条件が毎ステップ真になり、**毎ステップ保存**される（0 で「保存しない」ではない） | コードの読み（実行はしていない） |
| `save_every_steps = 0` | 更新回数ごとの保存は行わない。時間による保存（`save_every_minutes`）と終了時の保存だけ | コードの読み |
| `shuffle_buffer > 0` で再開 | 窓の中にあった局面は失われる（重複はしない） | `shk_feat` の仕様（REPORT_M2） |
| `device = "cuda"`（GPU の経路） | 固定メモリ経由の転送と、決定的な計算の設定は、Colab の GPU でしか確かめていない。この環境（CPU のみ）では動かせない | REPORT_M3・M4 |


---

## DL の学習（`python -m shikibu.train_dl`、M11）

DL（方策と価値の 2 つの頭）の設定です。RayPair・HalfKP の設定（上の表）とは別のキーを使い、`train_dl` の検査で、知らないキーは止まります。
ひな形は `configs/colab_dl.toml` です。`model.network` が `dl_` で始まる設定は、`shk_monitor` が自動で `train_dl` を起動します。

### [data]

| キー | 型 | 既定値 | 意味 |
|---|---|---|---|
| `train_glob` / `train_paths` | 文字列 / 文字列のリスト | `""` / `[]` | 学習用の PSV。どちらか 1 つ以上 |
| `val_path` | 文字列 | `""` | 検証用の PSV（任意）。学習に使わないファイルを指定する。`val_positions` 局面を読む |
| `val_hcpe_glob` | 文字列 | `""` | Dainagon のテストセット（hcpe、任意）。`from_hcpe` で読み、検証の損失は `score / 600` で測る |
| `cache_dir` | 文字列 | `""` | `train.py` と同じ（ローカルへの先読み）。空なら使わない |
| `max_cached` | 整数 | `3` | `cache_dir` に同時に置くファイル数の上限（1 以上） |
| `shuffle_buffer` | 整数 | `0` | 学習中のシャッフル窓（局面数）。0 なら使わない |
| `num_threads` | 整数 | `1` | 作業係の数（詰めた局面の作成）。Colab は論理コア数 − 1 程度 |

### [model]

| キー | 型 | 既定値 | 意味 |
|---|---|---|---|
| `network` | 文字列 | 必須 | 構造の名前 `dl_<C>_<trunk>_<policy>_<value>[_オプション]`（`docs/DL_BLOCKS.md`）。プリセット `A`：`dl_192_SHHHHT_H_HH`、`B`：`dl_128_SSHHHHHHHHTT_H_HH`、`X2`：`dl_192_SSHHHHHHTT_H_HH` |

### [train]

| キー | 型 | 既定値 | 意味 |
|---|---|---|---|
| `optimizer` | 文字列 | `"adamw"` | 今は `"adamw"` だけ。ほかの値は設定の誤りとして何も書かずに止まる（オプティマイザを作る口は `train_dl.make_optimizer` の 1 か所） |
| `batch_size` | 整数 | `4096` | 1 回の更新の局面数（1 以上）。L4 で足りなければ下げる（`REPORT_M11.md` のメモリの見積もり） |
| `lr` | 実数 | `4e-4` | 学習率の最大値 |
| `weight_decay` | 実数 | `1e-3` | AdamW の重み減衰。BN・LN の重み、バイアス、相対位置バイアス、位置の埋め込み、スカラーのゲートには掛けない |
| `warmup_steps` | 整数 | `0` | ウォームアップの更新回数 |
| `total_steps` | 整数 | 必須 | 学習率のコサイン減衰の長さ（1 以上） |
| `max_steps` | 整数 | 必須 | 通算の更新回数の上限（1 以上）。達したら保存して終わる |
| `min_lr_ratio` | 実数 | `0.0` | 減衰の最後の学習率（`lr` に対する比） |
| `eval_scale` | 文字列または正の数 | `"auto"` | `"auto"` なら勝敗から推定する（`scale_sample` 以下の局面で）。数値なら、その値を使う |
| `scale_sample` | 整数 | `10000000` | eval_scale の推定に使う局面数の上限 |
| `value_mix_lambda` | 実数 | `0.5` | 価値の損失の λ（0〜1）。CE（Binned Value）と BCE（勝敗）の比 |
| `value_loss_scale` | 実数 | `1.5` | 価値の損失全体の倍率 |
| `label_smoothing` | 実数 | `0.03` | 方策の交差エントロピーのラベル平滑化（0 以上 1 未満） |
| `grad_clip` | 実数 | `1.0` | 勾配のノルムの上限（`clip_grad_norm_`） |
| `amp` | 文字列 | `"bf16"` | `"bf16"`（CPU・CUDA）、`"fp16"`（CUDA のみ。`GradScaler` を使い、状態を checkpoint に保存）、`"off"` |
| `compile` | 真偽値 | `true` | `torch.compile(model)` で学習する。checkpoint にはコンパイル前の重みを保存する |
| `channels_last` | 真偽値 | `true` | 入力と重みを channels_last の形にする |
| `device` | 文字列 | `"auto"` | `"auto"`（CUDA があれば使う）、`"cpu"`、`"cuda"` など |
| `seed` | 整数 | `0` | 乱数の種（重み・ファイルの並び・シャッフル） |
| `save_every_minutes` | 実数 | `30.0` | この分数ごとに保存する。0 にしてはいけない |
| `max_minutes` | 実数 | `0` | 経過時間（この実行の分）がこれを超えたら、保存して正常に終わる（`end` 行の `stopped_by` が `"max_minutes"`）。0 なら無制限。Colab の 1 回 8 時間は `480` |
| `val_every_steps` | 整数 | `0` | この更新回数ごとに検証する。0 なら終わりの 1 回だけ（検証があるとき） |
| `val_positions` | 整数 | `100000` | 検証に使う先頭の局面数（`val_path`・`val_hcpe_glob` のとき 1 以上） |
| `log_every_steps` | 整数 | `100` | `log.jsonl` の train 行を書く間隔（1 以上） |
| `progress_every_sec` | 実数 | `30.0` | 画面の進捗と `status.json` を書く間隔（秒） |

### [output]

| キー | 型 | 既定値 | 意味 |
|---|---|---|---|
| `out_dir` | 文字列 | 必須 | 出力フォルダ（`checkpoint.pt`・`log.jsonl`・`status.json`・`train.lock`） |

### 損失（設定との対応）

```
loss = CE(p, move; label_smoothing) + value_loss_scale × (value_mix_lambda × CE(value_logits, hl_gauss(勝率)) + (1 − value_mix_lambda) × BCE(result_logit, result))
勝率 = sigmoid(score / eval_scale)
```

検証の `value_mse` は `v`（Binned Value の期待値）と `sigmoid(score / eval_scale)` の差の 2 乗の平均です（hcpe は `score / 600`）。

---

## M11.1：組ごとの混ぜ方（`[[data.sources]]`）と hcpe・`require_move`

### [[data.sources]]（組。`train_dl` だけ）

指し手のあるデータと、評価値・勝敗だけのデータを、1 バッチに混ぜて学習するための設定です。組ごとに、方策・価値・勝敗を学ぶかを決めます。
組が無ければ、`data.train_glob` を 1 つの組（PSV、3 つとも学ぶ、割合 1.0）として扱うので、M11 と同じ結果になります。

| キー | 型 | 既定値 | 意味 |
|---|---|---|---|
| `name` | 文字列 | `"source0"` など | 表示とログの名前。組の中で重複しないこと |
| `glob` | 文字列 | 必須 | 学習データ（glob）。`val_path` のファイルは学習から外す |
| `format` | 文字列 | `"psv"` | `"psv"`（40 バイト）または `"hcpe"`（cshogi の 38 バイト） |
| `weight` | 実数 | `1.0` | 1 バッチの中の割合（正の数。全部の組の合計で割る） |
| `policy` | 真偽値 | `true` | この組の局面で方策を学ぶ。`true` なら指し手の無い局面は捨てる（`require_move = true`）。**`true` のときは、指し手のある局面が 1 つも無いと、学習を始める前に止まる** |
| `value` | 真偽値 | `true` | この組の局面で価値（Binned Value）を学ぶ |
| `result` | 真偽値 | `true` | この組の局面で勝敗（Result Head）を学ぶ |
| `eval_scale` | 文字列または正の数 | `"auto"` | この組の評価値の尺度。`value = false` なら使わない。`"auto"` は組ごとに推定する（PSV は既存の推定、hcpe は先頭の局面から推定） |

- 1 バッチの局面数：組 i は `round(weight_i / Σweight × batch_size)` を切り捨て、端数は最後の組が受け持つ（合計は `batch_size`）。どの組も 1 局面以上になること（小さすぎる `batch_size` は設定の誤り）。
- 作業係の数（`num_threads`）とシャッフル窓（`shuffle_buffer`）は、重みで組に分ける（各組のスレッドは 1 以上）。
- 周回は組ごとに進む（その組のファイルの順番だけ並べ替える）。checkpoint には組ごとの読み込み位置・周回・順番を保存し、`--resume` で全部の組を続きから読む。
- 損失：項ごとに、その項を学ぶ局面だけで平均する（対象が 0 の項は 0）。`value = false` の組の局面は価値の損失に入らない。

```toml
[[data.sources]]
name = "moves_hcpe"
glob = "<指し手のあるデータ>/*.hcpe"
format = "hcpe"
weight = 0.5
policy = true
value = false
result = true
eval_scale = "auto"

[[data.sources]]
name = "big_psv"
glob = "<評価値と勝敗だけのデータ>/*.psv"
format = "psv"
weight = 0.5
policy = false
value = true
result = true
eval_scale = "auto"
```

### `require_move`・`fmt`（`BatchLoader` の引数、Python から直接使うとき）

| 引数 | 既定値 | 意味 |
|---|---|---|
| `output` | `"features"` | `"dl"` で詰めた局面（`fmt` を使える）。`"features"` は PSV だけ |
| `fmt` | `"psv"` | `"hcpe"` で cshogi の hcpe（38 バイト）を読む（`output="dl"` のときだけ） |
| `require_move` | `True` | `True` なら指し手の変換できない局面を捨てる。`False` なら `move = -1` のまま残す（捨てない） |

読み解けないレコード・勝敗が範囲外のレコードは、どちらの場合も捨てます。捨てた数は `skipped()` の合計と、`skip_reasons()`（"unreadable"・"result_range"・"no_move"）で理由別に分かります。

### 「使えないものが続く」ときの止め方

読んだレコードが 100 万件続けて 1 局面も使えなかったら、待ち続けずに `RuntimeError` で止まります（理由の内訳つき）。
`buffer_fill` の間は、30 秒ごとに「読んだ・使えた・理由別」を標準エラーと `log.jsonl`（`kind = "fill"`）に出します。

### キャッシュ（`cache_dir`）

- 組ごとに `cache_dir/src<番号>` のフォルダを使います（`BatchLoader` は作るとき、フォルダの `shk_cache_` のファイルを消すため、組どうしで共有しない）。
- コピーが失敗したときは、理由と回数を標準エラーと `log.jsonl`（`kind = "cache_event"`）に出して、最大 3 回までやり直します。
- DL のひな形の既定は `cache_dir = ""`（Drive から直接読む）。大きいデータを `cache_dir` で読むと、1 ファイルのコピーが終わるまで読み始められないので、待ち時間が長くなります。
