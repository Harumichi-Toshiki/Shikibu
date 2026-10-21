# コードの構成

Shikibu のコードの役割と、データの流れを説明します。
仕様（何を計算するか）は `docs/DESIGN.md` にあります。ここでは「どう作ってあるか」を書きます。

---

## 1. 全体の流れ

```
PSV（Drive の学習データ、または事前シャッフル済みのシャード）
  │  （cache_dir があるとき）コピー係がローカル SSD へ先読み
  ▼
読み係（1スレッド）：ファイルの順に読む → シャッフル窓（shuffle_buffer）→ batch_size 局面ごとに束ねる
  │  番号つきの束
  ▼
作業係（num_threads スレッド）：PSV の解読 → 両視点の特徴抽出（RayPair_v0 / v1）
  │  番号つきの結果（nn.EmbeddingBag の形式）
  ▼
BatchLoader（Python のイテレータ）：番号の順に並べ直して dict（numpy）で返す
  │
  ▼
train.py（PyTorch）：to_device → 順伝播 → 損失 → AdamW → 重みのクリップ（§F3.3）
  │
  ├─ 30分ごと・終了時 → checkpoint.pt（モデル・オプティマイザ・読み込み位置・乱数）
  ├─ log.jsonl・status.json・画面の進捗
  ▼
export.py → .shk（§F4）を書き、読み直して確かめる
  │
  ├─ quantsim.py：§E4 の量子化を Python で再現し、float との差を測る
  └─ tools/make_engine_golden.py：エンジン用の正解データ（JSON）
```

事前の処理（手元の PC で1回だけ）：

```
PSV（複数ファイル）→ shk_tools shuffle → shard_NNNNN.psv（各 1GB = 2,500 万局面）
                    （統計）shk_tools stats / estimate-scale
```

---

## 2. Rust のクレート `crates/shk_feat`

Python からは `shikibu_feat` という名前で呼びます（`Cargo.toml` の `[lib] name`）。
クレートの名前（`shk_feat`）と、Python のモジュール名（`shikibu_feat`）が違う点に注意してください。

| ファイル | 役割 | 主な公開関数・型 |
|---|---|---|
| `src/lib.rs` | pyo3 の入口。引数の検査と、Python に出す関数の登録 | `num_features`・`extract_sfen`・`decode_psv`・`extract_psv`・`bench_extract`・`shuffle_shards`・`BatchLoader` |
| `src/position.rs` | 局面の表現、SFEN の読み書き、PackedSfen のデコーダ、PSV の解読。解読は8ビットの表引き（LUT） | `parse_sfen`・`to_sfen`・`decode_psfen`・`decode_psv` |
| `src/raypair.rs` | 特徴量 RayPair（§F2）。ライン表（`LINES`）、駒の一覧（`Pieces`）、基数ソート | `num_features`・`features`・`features_from` |
| `src/psv.rs` | PSV レコードの読み取りの共通部品 | `score`・`ply`・`result`・`read_full`・`checked_size`・`for_each_record` |
| `src/loader.rs` | 学習用のバッチを作る `BatchLoader`（読み係・作業係・コピー係・再開） | `BatchLoader`（`__next__`・`state`・`cache_stats`） |
| `src/shuffle.rs` | 事前シャッフルとシャード分割（2パス方式） | `shuffle_shards` |
| `src/stats.rs` | 統計（勝敗欠損の判定）と eval_scale の推定（黄金分割探索） | `psv_stats`・`estimate_eval_scale` |
| `src/rng.rs` | 乱数（SplitMix64）。状態を保存・復元できる | `Rng` |
| `src/bench.rs` | 計測用：Python を介さない解読と特徴抽出の速さ | `bench_extract` |

- `unsafe` は使っていません。
- 外部の依存は pyo3・numpy のみです（`Cargo.toml`）。

---

## 3. Python パッケージ `python/shikibu`

| ファイル | 役割 |
|---|---|
| `lock.py` | 二重起動の防止（`train.lock`）。ロックの取得・心拍（10 秒ごと）・解放。古いロックの判定と、`--force-unlock` による引き継ぎ（M5.1） |
| `networks.py` | ネットワークの定義（`RayPairNetwork`・`RayPair_v0Network`・`RayPair_v1Network`）。`NETWORKS` に名前で登録し、`build(name)` で作る。重みの名前は §F4.2 と同じ |
| `train.py` | 学習ループ。設定の読み込み・検査（`load_config`・`validate`）、勝敗欠損の確認と eval_scale の決定、損失・オプティマイザ・学習率・重みのクリップ、保存と再開、ログ（`Status`・`JsonlLog`） |
| `export.py` | `checkpoint.pt` → `.shk`（§F4）。書き出しのあとで読み直して確かめる。`read_shk` もここ |
| `quantsim.py` | §E4 の量子化を Python で再現する確認用ツール（凍結区画ではない） |
| `__init__.py` | パッケージの説明だけ |

CLI は `python -m shikibu.train`・`python -m shikibu.export`・`python -m shikibu.quantsim` です（`PYTHONPATH=python` が必要）。

---

## 4. その他のフォルダ

| パス | 中身 | 変更できるか |
|---|---|---|
| `tools/shk_tools.py` | データの CLI：`stats`・`estimate-scale`・`shuffle` | 変更可 |
| `tools/shk_monitor.py` | Colab 用の監視ツール（学習を裏で起動し、進みを表示する） | 変更可 |
| `tools/make_engine_golden.py` | エンジン用の正解データを作る CLI | 変更可 |
| `bench/bench_loader.py` | ローダーの速さを、スレッド数と version を変えて測る | 変更可 |
| `perf/perf_check.py` | 速度の合格判定（変更禁止） | 変更禁止 |
| `reference/raypair_ref.py` | 特徴量の参照実装（Python、遅い。付録A）（変更禁止） | 変更禁止 |
| `tests/` | 受け入れテスト（変更禁止） | 変更禁止 |
| `configs/` | 設定ファイル（`colab_v0.toml`・`colab_v1.toml`） | 変更可 |
| `notebooks/colab_train.ipynb` | Colab の学習用ノートブック | 変更可 |
| `notebooks/colab_perf.ipynb` | Colab の速度確認用ノートブック（M2.5） | 変更可 |
| `docs/` | 設計書（`DESIGN.md`・`COLAB.md` は変更禁止）と、この文書群 | 文書は変更可 |

---

## 5. ローダーの仕組み（`loader.rs`）

### 5.1 スレッドの役割

| 係 | 数 | 仕事 |
|---|---|---|
| コピー係 | 0 か 1（`cache_dir` があるときだけ） | Drive などから `cache_dir` へファイルをコピーする（先読み） |
| 読み係 | 1 | ファイルの順に、40 バイトずつ読む。シャッフル窓から1件ずつ選ぶ。`batch_size` 件を束ねて作業係へ送る |
| 作業係 | `num_threads` | 束を解読し、両視点の特徴を作る（`features_of`） |
| 呼び出し側 | Python の主スレッド | 結果を番号の順に並べ直して、1バッチずつ返す |

- 読み係は、局面の並び（ファイルの順・シャッフル）を1か所で決めます。作業係は並びに関係なく特徴だけを作ります。
- そのため、並びは `num_threads` にも `cache_dir` にも依りません（テストで確かめている）。
- 作業係の結果は、番号つきで届きます。呼び出し側は `pending`（BTreeMap）に入れて、番号の順に取り出します。

### 5.2 メモリの上限（枠）

- 同時に作業係へ渡してある未回収のバッチは、最大 `2 × num_threads + 2` 個です。
- 読み係は、呼び出し側が1バッチ返すたびに1枠を受け取ってから、次の束を作ります（`credit` のチャネル）。
- そのため、作業が遅くても読みだけが先に進みすぎることはありません。

### 5.3 コピー係とキャッシュ（`cache_dir`）

- 同時に置くファイルは最大 `max_cached + 1` 個です（`CacheHub` の `cap`）。読んでいる最中のファイルも数えます。
- コピーは、`shk_cache_<番号6桁>_<元の名前>.part` の名前で作り、揃ってから改名します。途中のファイルは数えられません。
- 読み係がそのファイルを開くまで待ちます（`wait_for_copy`）。順番は、読む順と同じです。
- 読み終えたファイルは消して、枠を空けます。
- 起動時と終了時に、`shk_cache_` で始まるファイルを消します。**`cache_dir` は専用のフォルダにしてください。**
- コピーの記録（ファイル名・バイト数・秒・MB/秒）は `cache_stats()` で取れ、`log.jsonl` の `cache` 行になります。

### 5.4 シャッフル（学習中）

- 窓には最大 `shuffle_buffer` 件を入れておき、窓から乱数で1件ずつ選んで出します（`swap_remove`）。出すたびに、読み係が1件補充します。
- 全ファイルを読み終えたら、窓を空になるまで出し切ります。1周で全局面がちょうど1回ずつ出ます（REPORT_M2 の仕様。テストで確認）。

### 5.5 再開（`state` と `resume_state`）

- `state()` は、直前に返したバッチの**直後**の読み位置を返します。中身は `file`（ファイル番号）・`offset`（そのファイルのバイト位置）・`rng`（シャッフルの乱数の状態）・`n_files`（ファイル数）です。
- `resume_state` を渡すと、その位置から読み直します。`n_files` が違えばエラーです。
- `shuffle_buffer = 0` のとき、読み位置の先が次の局面なので、続きは完全に一致します。
- `shuffle_buffer > 0` のときは、窓の中にあった局面は失われます（重複はしません）。

### 5.6 周回（エポック）

- `train.py` が、周回ごとにファイルの並びを作ります（`epoch_perm`：`random.Random(seed + 周回番号)` で順列を作る）。
- その並びで `BatchLoader` を作り、乱数の種は `seed + 周回番号` にします。
- 1周が終わったら、周回番号を 1 増やして次の並びで作り直します。
- 並びは checkpoint の `file_perm` に保存します。読み込み位置（`file`）がファイル番号を指すので、並びを途中で変えると位置がずれるためです。

---

## 6. 事前シャッフル（`shuffle.rs`、§S3.1）

2パスで行います。

1. **パス1**：全局面を乱数で `K` 個のバケツ（`_buckets_tmp/bucket_NNNNN.bin`）へ振り分ける。
   - `K = ceil(全体のバイト数 / (mem_limit_mb の 1/4))`。上限は 1,024（同時に開くファイル数を抑えるため）。
2. **パス2**：バケツの順を乱数で並べ、バケツを1つずつメモリに読んで、Fisher–Yates で混ぜる。`records_per_shard` 件ごとに `shard_NNNNN.psv` へ連続して書く。
   - 使い終わったバケツは消します。一時ファイルは最大で全データの量です。
- 出力フォルダに `shard_*.psv` が既にあればエラーです（上書きしない）。
- 最後のシャードだけが端数で、それ以外はちょうど `records_per_shard` 件です（テストで確かめている）。
- **検証用のシャードは自動では除かれません。** 持ち主が、最後のシャードを `shard_val.psv` に改名して使っています（設計書 §S3.1 の 4 とは違う。下の「設計書との違い」を参照）。

---

## 7. 勝敗の確認と eval_scale の推定（`stats.rs`、§S4）

- `psv_stats`：全ファイルを1回読み、`game_result` の数え上げ・score の範囲・手数を集計する。`game_result == 0` が 95% を超えると `result_missing = True`。
- `estimate_eval_scale`：Algorithm R で `sample` 件を一様に抽出し、`|score| ≥ 30000` を除いたうえで、`BCE(sigmoid(score / S), r)` を `log S` について黄金分割探索（範囲 [50, 5000]、終了幅 1e-4）で最小化する。
- BCE は softplus の形で書いてあるので、評価値が大きくても発散しません。

学習の中では、この2つを**全学習ファイルからの連続ブロック（`scale_sample`）に対して**行います（`train.py` の `sample_scale_positions`）。抜き出した局面は一時フォルダ（`tempfile.gettempdir()`。Colab ではローカル）のファイルに書き、終わったら消します。

---

## 8. 学習ループ（`train.py`）

### 8.1 初期化と再開

1. 設定を読み、検査する（`load_config` → `validate`）。
2. 出力フォルダに `checkpoint.pt` があれば、`--resume` が無いと止まる。あれば、`network` と `lam` が合うかを確かめて読み込む。
3. 新規のときだけ、勝敗の確認と eval_scale の決定を行う（`_decide_eval_scale`）。
4. 乱数の種を決め、モデル・オプティマイザを作る。
5. 検証用のローダー（`val_path` があるとき）と、学習用のローダーを作る（`_open_loader`）。

### 8.2 1ステップ

1. `BatchLoader` から1バッチを受け取る（周回の終わりなら次の周回の並びで作り直す）。
2. `to_device`：numpy を `torch.from_numpy` で包み、CUDA のときは固定メモリ経由で非同期に送る。
3. 順伝播：`EmbeddingBag`（sum）で両視点の蓄積値を作り、手番側を先に連結する。その後、0〜1 にクリップする処理（ClippedReLU）を3回通し、最後の線形層で z を出す（§F3.2）。
4. 損失：`BCEWithLogits(z, λ·sigmoid(score/eval_scale) + (1−λ)·result)`。
5. 逆伝播・AdamW の更新。学習率は `lr × lr_factor(step)`（ウォームアップ＋コサイン）。
6. `clip_weights_`（§F3.3）。

### 8.3 保存

- 30 分ごと（`save_every_minutes`）と、`save_every_steps` ごと、終了時に保存します。
- `checkpoint.pt.tmp` に書いてから改名するので、途中で切れても古い `checkpoint.pt` が残ります。

### 8.4 進捗の表示（`Status` と `shk_monitor`）

- 画面：`progress_every_sec` 秒ごとに、`[ 12.3%] step … / … 残り … 万局面/秒` の1行を出します。同時に `status.json` を書きます。
- 残り時間の見込み（`eta`）は、**この実行の train 段階の平均速度**から計算します（残りのステップ × 1ステップの平均時間）。再開した直後は、再開してからの速さで見積もります。
- `log.jsonl` の `train` 行には、`progress`・`eta_sec`・`positions_per_sec`（直前の区間の速さ）が入ります。
- `tools/shk_monitor.py` は、`status.json` と `log.jsonl` を読んで表示します。残り時間は、直近 20 行の平均速度から計算し直します。
- 学習は、この表示とは別の裏のプロセス（`subprocess`）で動きます。表示のセルを止めても、学習は止まりません。

---

## 9. 書き出しと確認

- `export.py`：`checkpoint.pt` から重みを取り出し（`export_tensors`）、§F4 の形式で書く。書いたあとで `read_shk` で読み直し、ヘッダ・CRC32・全テンソル・eval_scale・λ を照らす。
- `quantsim.py`：§E4 の整数の計算を numpy で行い、float 版との z・評価値（cp）・損失の差を測る。エンジンと同じ丸め（0 から遠い方への四捨五入）を使う。
- `make_engine_golden.py`：`cshogi` で乱数に指し進めた局面を作り、特徴と量子化の途中の値、cp を JSON に書く。

---

## 10. 設計書（§S1）との違い

設計書 §S1 の構成と、実際のファイル配置には違いがあります。コードの側は、動くことを優先して次のようにしました。

| 設計書（§S1） | 実際 | 理由・備考 |
|---|---|---|
| `crates/shk_loader/`（読み込み・ローダー） | `crates/shk_feat/src/loader.rs` | M2 の実装で、特徴抽出と同じクレートに入れた（理由の記録は無い） |
| `crates/shk_tools/`（CLI） | `tools/shk_tools.py`（Python）と、Rust の関数 | CLI は Python で書いた（M2） |
| `python/shikibu/features.py`（Rust のラッパ） | なし（`shikibu_feat` を直接呼ぶ） | M1 で Rust の関数を直接公開した |
| `python/shikibu/networks/`（ネットワークごとのファイル） | `python/shikibu/networks.py`（1ファイル） | M3 で1ファイルにした |
| 検証用シャードの自動除外（§S3.1 の 4） | なし。持ち主が手で改名する | 設定の `val_path` で指定する方式にした（M3）。今の配置には検証用が無い |
| 評価値フィルタ（§S6 の `|score|` の上限） | なし | 未実装（M3 報告の引き継ぎ） |
| 2〜3 シャードのシャッフルバッファ（§S3.2） | `shuffle_buffer`（局面数で指定） | 局面数で指定できる形にした |
| 保存に含まれるスケジューラ（§S6） | なし（学習率は更新回数から計算） | `lr_factor` が更新回数だけで決まるので、状態を持たない |

