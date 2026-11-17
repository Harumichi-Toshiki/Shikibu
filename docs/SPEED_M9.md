# M9 の使い方：開始前の待ち時間と第1層の速さ

すべて設定（TOML）のオプションです。**既定の値は今の動作のまま**なので、何もしなければ今までどおり動きます。
持ち主が Colab の GPU で速さと損失を確かめてから、既定を変えてください。

> **注意（L4 の測定値・M9.1）：M9 の第1層のオプションは、L4 では既定より遅かった。使わないこと。**
> HalfKP 256（因数分解なし）：既定 E0 157.5 万局面/秒、順伝播 sparse（E1）157.3、順伝播・勾配 sparse＋遅延型（E2）52.4、遅延型のみ（E3）119.6、E2＋bf16（E4）52.1。
> HalfKA_hm 512（因数分解つき）：既定 84.4 万局面/秒、順伝播 sparse（E1）44.5（その後の E2 はメモリ不足で失敗）。
> bf16（`fast`）は効かなかった（2 層目以降の計算は時間の一部）。`ft_forward`・`ft_grad`・`ft_optimizer = "lazy_adam"` は、既定のまま（`dense`・`adamw`）で使うこと。
> 第1層の専用カーネル（`ft_kernel = "triton"`、M9.1）は、既定より速いかどうかを E5 の行で確かめる（REPORT_M91.md）。

## 1. オプション一覧

| 設定 | 既定 | 何をするか |
|---|---|---|
| `data.check_mode` | `"sample"` | 勝敗の確認に使う局面。`"sample"`（従来・上限 `scale_sample`）、`"probe"`（各ファイルの先頭・中央・末尾から `probe_records` 件ずつ）、`"full"`（全件） |
| `data.probe_records` | `3000` | probe のとき、1 ファイルから読む局面数 |
| `data.probe_files` | `0` | probe のとき、読むファイル数の上限（0 は全ファイル）。読む回数は約 3 × この数 |
| `data.check_cache` | `false` | 確認の結果を `shikibu_check_cache.json`（データのフォルダ）に保存し、同じデータなら次回から読むだけにする |
| `data.shuffle_warm` | `0` | 0 は従来どおり（バッファを満たしてから学習）。`shuffle_buffer` 以下の値なら、その件数から学習を始める |
| `train.ft_forward` | `"dense"` | 第1層の順伝播。`"sparse"` は有効な行だけを足す（因数分解つきでも W_eff を作らない） |
| `train.ft_grad` | `"dense"` | 第1層の勾配。`"sparse"` は疎な勾配（`ft_optimizer = "lazy_adam"` が必要） |
| `train.ft_optimizer` | `"adamw"` | `"lazy_adam"` は第1層の重みだけ、使った行だけを更新する Adam（下の注意を読むこと） |
| `train.ft_kernel` | `"torch"` | 第1層の計算。`"triton"`（M9.1）は専用カーネル。`ft_forward`・`ft_grad` は `"dense"` のみ。GPU が無いときは解釈モード（環境変数 `TRITON_INTERPRET=1`）でだけ動く（遅い） |
| `train.fast` | `false` | CUDA のとき、2 層目以降を bf16 の自動混合精度で計算する（CPU では無視） |

設定の検査は学習の前に行われ、組み合わせが不正なら止まります（例：`ft_grad = "sparse"` は `ft_optimizer = "lazy_adam"` が必要。因数分解つきの網では `ft_forward = "sparse"` も必要）。

## 2. 開始前の段階の時間（`log.jsonl` の `startup` 行）

学習の開始前に、次の時間が 1 行で記録されます（`start` 行の次）。

```json
{"type": "startup", "seconds": {"data_check": 12.3, "first_batch": 1.2}, "cache": "miss", "check_mode": "sample", "scale_positions_used": 10000000, "shuffle_warm": 0, "ft_forward": "dense", "ft_grad": "dense", "ft_optimizer": "adamw", "fast": false}
```

- `data_check`：勝敗の確認と eval_scale の推定の時間
- `first_batch`：シャッフルバッファの充填を含む、最初のバッチまでの時間
- `cache`：`"off"`（キャッシュを使わない）・`"miss"`（作り直した）・`"hit"`（読んだだけ）

段階の名前（`status.json`）は、`data_check` → `eval_scale` → `prefetch` → `buffer_fill` → `train` です。

## 3. 持ち主の Colab での手順

### 3.1 速さの一覧（`tools/shk_speedcheck.py`）

```bash
!python /content/drive/MyDrive/<置き場所>/shk_speedcheck.py --repo <REPO> --glob "<学習データ>/*.psv" --network halfkp_256x2-32-32_f --batch 65536 --threads 11 --shuffle-buffer 20000000 --cache-dir /content/shk_cache
```

- `--cache-dir` は**専用のフォルダ**にする（学習の `cache_dir` と同じく、`shk_cache_` で始まるファイルを消す）。
- 結果の行の意味：
  - `A`〜`D`：従来の測り方（読み込み・転送・計算・全部込み）
  - `E0`〜`E4`：第1層のオプションの組み合わせごとの計算（同じバッチを繰り返す。E0 が既定）
  - `F1`〜`F4`：開始前の待ち時間（F1 sample、F2 probe、F3 シャッフル既定、F4 shuffle_warm）
- 開始前の待ち（F）を飛ばして E までだけ測るには `--skip-startup` を付ける。

### 3.2 設定の書き方（例）

一つずつ足して、損失の推移（`log.jsonl` の `train` 行の `loss`）と速さを比べてください。

開始を速くする：

```toml
[data]
check_mode = "probe"
probe_files = 500
check_cache = true
shuffle_buffer = 20000000
shuffle_warm = 2000000
```

第1層を疎にする：

```toml
[train]
ft_forward = "sparse"
ft_grad = "sparse"
ft_optimizer = "lazy_adam"
```

bf16（ふるい分け用）：

```toml
[train]
fast = true
```

既定に戻すには、そのキーを消す（または上の値に戻す）。途中で `ft_optimizer` を変えて `--resume` すると止まります（checkpoint の形が違うため）。

## 4. 注意（数学と動作の違い）

- **`lazy_adam` は AdamW と数学的に同じではありません。** 使われなかった行は、重み・1 次・2 次のモーメントのどれも動きません。密な AdamW は、使われなかった行も 1 次モーメントの慣性で動きます。1 ステップ目だけは一致します（テストで確認済み）。重みの減衰は第1層にはもともとかからないので、そこは同じです。
- `check_mode = "probe"` と `"sample"` は、読む局面が違うので、**eval_scale の値が変わることがあります**（`start` 行の `eval_scale` で確認できます）。
- `shuffle_warm > 0` は、学習の初めの数百ステップの並びが変わります。1 エポックで全局面が 1 回ずつ使われることは変わりません（テストで確認済み）。
- `check_cache` は、ファイルの名前・大きさ・更新時刻で判定します。データを Drive にコピーし直すと更新時刻が変わるので、作り直されます。
- `fast` は CUDA でだけ効きます。bf16 の丸めのぶん、損失の推移は fp32 と少し違います。書き出しは常に fp32 の重みから行います。

## 5. 確かめ（テスト）

`tests/test_m9_acceptance.py`（24 件）。主なもの：

- 1 エポックで全局面が 1 回ずつ使われる（シャッフルの各方式、キャッシュあり）
- probe の範囲（先頭・中央・末尾）と、範囲外の勝敗の飛ばし、壊れたファイルの拒否
- 確認の結果のキャッシュ（hit で eval_scale が同じ、壊れたファイルでも止まらない）
- 疎な順伝播と密な W_eff の出力が一致、疎な勾配を densify したものが密な勾配と一致
- 遅延型 Adam の 1 ステップ目は AdamW と一致、動く行の集合が AdamW と一致、損失が同程度に下がる
- `fast` は CPU で結果を変えない
- 不正な組み合わせは学習の前に止まる
