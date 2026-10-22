# ファイル形式

Shikibu が読み書きするファイルの形式をまとめます。
それぞれの項目は、コード（`crates/shk_feat/src/`・`python/shikibu/`・`tools/`）を読み、小さな実データで出力を確かめました。

| ファイル | 形式 | 作るもの | 読むもの |
|---|---|---|---|
| PSV | 1局面 40 バイトの固定長 | 持ち主が用意した教師データ | ローダー・統計・シャッフル |
| `.shk` | 独自の2進形式（§F4） | `export`（`checkpoint.pt` から） | エンジン（Nuenagon）、`read_shk`、量子化の確認 |
| `checkpoint.pt` | PyTorch の `torch.save` の dict | 学習（30分ごと・終了時） | 再開（`--resume`）、書き出し |
| `log.jsonl` | 1行1 JSON | 学習 | 監視ツール、ノートブック |
| `status.json` | JSON（1つ） | 学習（段階が変わるたびと、進捗を出すたび） | 監視ツール |
| エンジン用の正解データ | JSON（1つ） | `tools/make_engine_golden.py` | Nuenagon の N3 |
| シャード `shard_NNNNN.psv` | PSV と同じ | `shk_tools shuffle` | 学習 |
| `train.lock` | JSON（1つ）。学習中だけある | 学習の開始時、10 秒ごとに書き直す | 学習の開始時（二重起動の確認） |
| `config.toml` | TOML（ひな形を書き換えたもの） | ノートブックの節 5 | 学習器（`--config`） |

---

## 1. PSV（PackedSfenValue、1局面 40 バイト）

§S2 のとおりです。すべてリトルエンディアンです。

| オフセット | 型 | 内容 | 読み方 |
|---|---|---|---|
| 0〜31 | 32 バイト | PackedSfen（局面の圧縮形式、やねうら王の形式） | `position.rs` の `decode_psfen` で SFEN に直す |
| 32〜33 | i16 | score（**手番側から見た**評価値。cp） | `psv::score` |
| 34〜35 | u16 | move（学習には使わない） | — |
| 36〜37 | u16 | gamePly（手数） | `psv::ply` |
| 38 | i8 | game_result（**手番側から見て**：1 = 勝、0 = 引き分け、−1 = 負） | `psv::result` |
| 39 | u8 | padding（0） | — |

- 1ファイルの大きさは必ず 40 の倍数です。そうでなければ `ValueError`（「サイズが40バイトの倍数でない」）になります。
- PackedSfen の中身（マス番号・手番・玉の位置・駒のハフマン符号・持ち駒）は、cshogi の符号表どおりに読みます（REPORT_M1）。デコーダの正しさは、cshogi の `to_psfen` と照らして確かめています（テスト1）。
- マス番号は `y = s % 9`、`x = 8 − s / 9`（`1一` が 0。REPORT_M1）。

確かめたこと：小さな PSV（12,000 局面）で、`game_result` が −1・0・1 の値だけになり、`shk_tools stats` が正しく数えること。

---

## 2. `.shk`（重みファイル、§F4）

### 2.1 ヘッダ（180 バイト）

すべてリトルエンディアンです。`export.py` の `_pack_header` と `read_shk` で確かめています。

| オフセット | 型 | 内容 | 例（RayPair_v0 の確認用ファイル） |
|---|---|---|---|
| 0 | char[8] | マジック `SHIKIBU\0` | `SHIKIBU\0` |
| 8 | u32 | 形式バージョン（現在 1） | 1 |
| 12 | char[32] | 特徴セット ID（ASCII。残りは `\0`） | `RayPair_v0` |
| 44 | char[32] | ネットワーク ID（同上） | `RayPair_v0Network` |
| 76 | u32 | 特徴の次元数（v0 = 15000、v1 = 115352） | 15000 |
| 80 | u32 | W（第1層の幅） | 128 |
| 84 | u32 | L1 | 16 |
| 88 | u32 | L2 | 16 |
| 92 | f32 | eval_scale（エンジンは `cp = z × eval_scale`） | 4999.8203125 |
| 96 | f32 | λ（記録用） | 0.75 |
| 100 | u64 | 学習した局面数（記録用）。**`step × batch_size` の近似値** | 102400（400 × 256） |
| 108〜171 | u8[64] | 予約（すべて 0。`read_shk` は 0 でなければ拒否） | 0 |
| 172 | u64 | 本体のバイト数 | 7698116 |

### 2.2 本体

float32 の配列を、次の順に隙間なく並べます（§F4.2）。行優先です。

| 順 | 名前 | 形 | v0（W=128、L1=L2=16）の要素数 |
|---|---|---|---|
| 1 | `ft_weight` | [次元数][W] | 15000 × 128 |
| 2 | `ft_bias` | [W] | 128 |
| 3 | `l1_weight` | [L1][2W] | 16 × 256 |
| 4 | `l1_bias` | [L1] | 16 |
| 5 | `l2_weight` | [L2][L1] | 16 × 16 |
| 6 | `l2_bias` | [L2] | 16 |
| 7 | `out_weight` | [1][L2] | 1 × 16 |
| 8 | `out_bias` | [1] | 1 |

- `l1_weight[j][k]` の k は、0〜W−1 が手番側、W〜2W−1 が非手番側です（PyTorch の `nn.Linear.weight` と同じ並び）。
- 本体の長さは `形の要素数の合計 × 4` と一致しなければなりません。

### 2.3 フッタ

| 型 | 内容 |
|---|---|
| u32 | 本体の CRC32（IEEE 802.3。zlib の `crc32` と同じ） |

### 2.4 全体の大きさ

`180 + 本体のバイト数 + 4` です。v0 のとおりに確かめた例では、本体は 7,698,116 バイト、ファイルは 7,698,300 バイトでした（一致）。
v1（W=128、L1=L2=32、特徴 115,352）の本体は、同じ計算で約 59MB（59,097,988 バイト。M3 報告の「約 60MB」と合う）になります。

### 2.5 読むときの検査

`read_shk` は次を検査し、どれかが違うと `ValueError` を出します。

| 検査 | 失敗したときのメッセージ（例） |
|---|---|
| ファイルが短すぎる | 「ファイルが短すぎる」 |
| マジック | 「マジックが違う（SHIKIBU ではない）」 |
| 形式バージョン | 「形式バージョンが未対応」 |
| 予約領域 | 「予約領域が 0 でない」 |
| ファイルの長さ | 「ファイルの長さが本体の長さと合わない（切れている可能性）」 |
| CRC32 | 「CRC32 が一致しない」 |
| 本体の長さと形 | 「本体の長さが形と合わない」 |

確かめたこと：書き出したファイルの 1 バイトを変えると、「CRC32 が一致しない」で拒否されること。

### 2.6 書き出し（`export_shk`）

- `checkpoint.pt` の `network` と `config.model.ft_width` からモデルを組み、重みを読み込んで書き出します。
- 書き出したあと、同じファイルを `read_shk` で読み直し、ヘッダ・全テンソル・`eval_scale` と `lam`（float32 に丸めたもの）が一致するかを確かめます。一致しなければ `ValueError`（「書き出しの確認に失敗」）です。

---

## 2b. `nn.bin`（やねうら王の HalfKP 形式、比較実験用、§S11）

HalfKP（`HalfKP_256Network`・`HalfKPf_256Network`）の重みは、やねうら王の halfkp_256x2-32-32 と同じ `nn.bin` で書きます。`.shk` とは別の形式です。

- 並び（すべてリトルエンディアン）：ヘッダ（u32 版 `0x7AF32F16`、u32 ファイルのハッシュ、u32 構造の文字列の長さ、構造の文字列）→ 特徴変換（u32 ハッシュ、int16 バイアス[256]、int16 重み[125388][256]）→ ネットワーク（u32 ハッシュ、隠れ層1：int32 バイアス[32]・int8 重み[32][512]、隠れ層2：int32 バイアス[32]・int8 重み[32][32]、出力層：int32 バイアス[1]・int8 重み[1][32]）。
- 因数分解つき（`HalfKPf_256Network`）は、書き出すときに各玉の位置の特徴へ畳み込みます（`W[k×1548+p] += W[125388+p]`）。
- 量子化：特徴変換は ×127（int16）、隠れ層の重みは ×64（int8）、バイアスは ×8128（int32）、出力層は s = eval_scale × FV_SCALE / 8128 を掛けてから量子化します。丸めは 0 から遠い方への四捨五入で、範囲を超えたら範囲にクリップします。
- **FV_SCALE**：指定が無ければ `max(1, floor(8128 / eval_scale))`。書き出しの標準出力に `FV_SCALE=<整数>` と表示されます。エンジンの `FV_SCALE` に同じ値を設定すると、cp ≒ 学習の z × eval_scale になります。
- 書き出したあと、同じファイルを読み直し、ヘッダ・ハッシュ・全配列が一致するかを確かめます。
- マスの番号・駒の番号は、やねうら王と同じ HalfKP の定義です（定義は `tests/halfkp_ref.py` にある）。

---

## 3. `checkpoint.pt`（学習の状態）

`torch.save` の dict です。`torch.load(path, map_location="cpu", weights_only=False)` で読めます。
書き込みは `checkpoint.pt.tmp` に書いてから改名するので、途中で切れても古い `checkpoint.pt` が残ります。

確かめた実物（学習 20 ステップ、ファイルの並びが `[0, 2, 1]`）のキーは次のとおりです。

| キー | 型 | 内容 |
|---|---|---|
| `model` | state_dict | 重み（`ft.weight`・`ft_bias`・`l1`・`l2`・`out`） |
| `optimizer` | dict | AdamW の状態 |
| `step` | int | 通算の更新回数 |
| `epoch` | int | 周回番号（0 始まり。全ファイルを1周した回数） |
| `loader_state` | dict | 読み込み位置。`file`（ファイル番号）・`offset`（バイト位置）・`rng`（シャッフルの乱数の状態）・`n_files` |
| `file_perm` | list[int] | 今の周回でのファイルの並び（0〜n−1 の順列）。M3 の checkpoint には無い（M4 で追加） |
| `eval_scale` | float | 使っている eval_scale |
| `eval_scale_source` | `"auto"` または `"config"` | eval_scale の由来 |
| `lam` | float | λ |
| `network` | str | ネットワーク名 |
| `config` | dict | 学習に使った設定（全体） |
| `rng` | dict | 乱数の状態。キーは `torch`・`python`・`numpy`（CUDA があれば `cuda` も） |

- M3 の形式（`file_perm` が無い）からも `--resume` で続けられます（そのときは、その周回の残りを設定の順のまま続けます）。
- `positions` はこの中には無く、`.shk` を書くときに `step × batch_size` で計算します。

---

## 4. `log.jsonl`（学習の記録）

1行1 JSON です。`type` で行の種類を分けます。
学習を再開すると、`start` 行が追記されます（ファイルは上書きされません）。

| `type` | 書くとき | 項目 |
|---|---|---|
| `start` | 学習（または再開）の開始時に1回 | `step`・`epoch`・`resumed`（再開なら true）・`eval_scale`・`eval_scale_source`・`lam`・`network`・`device`・`n_files`・`scale_positions_used`（勝敗確認と推定に使った局面数）・`config`（設定全体） |
| `phase` | 段階に入るとき（`prefetch`・`buffer_fill`・`train` だけ） | `name` |
| `train` | `log_every_steps` ごと | `step`・`epoch`・`loss`・`lr`・`progress`（`step / max_steps`）・`eta_sec`（残り秒数の見込み）・`positions_per_sec`（直前の区間の速さ）・`elapsed_sec` |
| `val` | `val_every_steps` ごと（`val_path` があるとき） | `step`・`val_loss`・`val_mse` |
| `save` | 保存したとき | `step`・`epoch`・`reason`（`"periodic"` または `"final"`） |
| `cache` | 保存のたびと、コピーが進んだとき（`cache_dir` があるとき） | `step`・`epoch`・`copied`（前回から増えた分のコピーの記録のリスト）・`copied_total` |
| `end` | 学習の終わりに1回 | `step`・`epoch`・`elapsed_sec` |

- `copied` の1件は `{"file", "bytes", "seconds", "mb_per_s"}` です。`mb_per_s` の MB は 10⁶ バイトです。
- `status.json` の段階（`data_check`・`eval_scale`）は、`log.jsonl` には書かれません（`start` より前のため）。
- 監視ツールは、最後の `train` 行の速さ（直近 20 行の平均）から残り時間を計算します。

確かめた例（20 ステップの確認用の学習）：`start`・`phase`×3・`save`×2・`cache`×2・`train`×4・`end`×1 の行が書かれました。

---

## 5. `status.json`（今の段階）

1つの JSON です。段階が変わるたびと、進捗を画面に出すたびに、一時ファイルから改名して上書きします。
監視ツールはこれを読みます。

```json
{"phase": "train", "history": ["data_check", "eval_scale", "prefetch", "buffer_fill", "train"],
 "step": 1234, "max_steps": 179000, "progress": 0.0069, "eta_sec": 3600.0, "updated": 1792563497.9}
```

| キー | 内容 |
|---|---|
| `phase` | 今の段階（下の表） |
| `history` | この実行で入った段階の順。**実行ごとに作り直すので、再開したら前の実行の段階は含まない** |
| `step`・`max_steps` | 通算の更新回数と上限 |
| `progress` | `step / max_steps` |
| `eta_sec` | 残り秒数の見込み（この実行の train 段階の平均速度から）。終わりは 0.0 |
| `updated` | 書いた時刻（UNIX 時刻） |

段階の名前：

| 名前 | いつ |
|---|---|
| `data_check` | 勝敗欠損の確認（局面の抜き出しと `psv_stats`）。新規の学習だけ |
| `eval_scale` | eval_scale を推定するとき（設定が `"auto"` のとき）だけ |
| `prefetch` | `cache_dir` があるとき |
| `buffer_fill` | `shuffle_buffer > 0` のとき |
| `train` | 学習中 |
| `done` | 正常に終わった |
| `error` | 設定・データのエラーで終わった（`main` が `ValueError`・`RuntimeError`・`FileExistsError` を捕まえたとき） |

確かめた例：`eval_scale` を設定で与えたとき、`history` は `data_check` → `train` → `done` でした。

---

## 5b. `train.lock`（二重起動の防止、M5.1）

出力フォルダの `train.lock` は JSON です。学習器が、出力フォルダに何かを書く前に作ります。

```json
{"pid": 12345, "host": "ホスト名", "started": 1792563497.9, "heartbeat": 1792563527.9}
```

| キー | 内容 |
|---|---|
| `pid` | 学習器のプロセス番号 |
| `host` | `socket.gethostname()` の値 |
| `started` | 開始時刻（UNIX 時刻） |
| `heartbeat` | 最後に書いた時刻（10 秒ごと。ファイルの更新時刻も同じ頃になる） |

- 正常終了・エラー終了・SIGTERM で消えます。SIGKILL や電源断では残ります（次の起動で古いロックとして引き継がれる）。
- 判定の規則は `docs/OPERATIONS.md` の 3 を参照してください。

---

## 6. エンジン用の正解データ（`shikibu-engine-golden-v1`）

`tools/make_engine_golden.py` が作る JSON です。Nuenagon の N3（推論の一致）で使います。

```
{
  "format": "shikibu-engine-golden-v1",
  "shk_sha256": "<対応する .shk の SHA-256（16進）>",
  "header": { ...read_shk のヘッダ... },
  "positions": [
    {"sfen": "...", "stm_features": [...], "nstm_features": [...],
     "acc_stm": [...], "acc_nstm": [...], "raw": 287, "cp": 177}, ...
  ]
}
```

| 項目 | 内容 |
|---|---|
| `stm_features`・`nstm_features` | 特徴の番号（昇順・重複込み）。手番側・非手番側 |
| `acc_stm`・`acc_nstm` | 量子化した第1層の蓄積値（長さ W の int 配列） |
| `raw` | 最終出力の int（`z = raw / 8128`） |
| `cp` | `round(raw × eval_scale / 8128)`（倍精度、0 から遠い方への四捨五入、**クリップしない**） |

- 局面は、平手初期局面から乱数で指し進めて作ります（`--seed` で決まる。200 手で打ち切り）。
- 先頭は平手初期局面です（確かめた：50 局面のうち先頭が平手初期局面、後手番が 25、持ち駒のある局面が 23）。
- `shk_sha256` は、そのとき使った `.shk` ファイル全体の SHA-256 です。エンジンの側で同じ `.shk` を使っているかを確かめられます。
- `cp` はクリップしていないので、詰みの値（§E4 の ±(詰み値 − 1)）と重なることがあります。エンジン側でクリップしてください。

---

## 7. ツールの出力（JSON）

`shk_tools stats`（`psv_stats` の戻り値を表示）：

| キー | 内容 |
|---|---|
| `records` | 局面数 |
| `result_counts` | `{"-1": 負の数, "0": 引き分けの数, "1": 勝ちの数}`（手番側から見た値） |
| `score_min`・`score_max`・`score_mean` | score の最小・最大・平均 |
| `result_missing` | 勝敗欠損のとき true（`game_result == 0` が 95% 超） |
| `ply_min`・`ply_max`・`ply_mean` | 手数の最小・最大・平均 |

`shk_tools estimate-scale`：

| キー | 内容 |
|---|---|
| `eval_scale` | 推定値 |
| `bce` | 推定に使った局面の平均 BCE |
| `used` | 推定に使った局面数（|score| ≥ 30000 を除いた中から抽出） |

