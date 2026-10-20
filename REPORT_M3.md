# M3 報告：学習ループ（`shikibu`）

## 結果

**完了。** 合格条件を満たした（`python -m pytest -q tests` が 62 passed、`python check_protected.py` が OK）。
ただし Colab（GPU）での実行はしていない。GPU の経路（`device = "cuda"`、固定メモリ転送）は、この環境では動かせないため未検証。

## テスト結果

```
$ python -m pytest -q tests
..............................................................           [100%]
62 passed in 58.26s
```

内訳：M1 34 件、M2 15 件、M3 13 件（`tests/test_m3_acceptance.py`）。
ビット単位の再開（通しで 40 ステップ ＝ 23 ステップで止めて `--resume` で 40 ステップ）は CPU・決定的設定で一致した。

## 保護ファイルの確認

```
$ python check_protected.py
OK
```

手順1でキット（`shikibu_kit_m3.zip`）を検証して配置した。`tests/m3_data.py`・`tests/test_m3_acceptance.py`・`PROTECTED.sha256` を KIT.json の SHA-256 で確かめてからコピーし、リポジトリ直下の zip を削除した（コミット `1de13f5`）。Rust 側（M1・M2・M2.5）は変更していない。

## 作ったファイル

| ファイル | 役割 |
|---|---|
| `python/shikibu/__init__.py` | パッケージの説明だけ |
| `python/shikibu/networks.py` | RayPair_v0Network / RayPair_v1Network。§F3.2 の計算、名前の登録（`NETWORKS`・`build`）、§F4.2 の名前での書き出し・設定 |
| `python/shikibu/train.py` | 学習ループ。設定読み込み、eval_scale と勝敗欠損の判定、損失・オプティマイザ・学習率・重みのクリップ、保存と再開、ログ。CLI は `python -m shikibu.train --config <TOML> [--resume]` |
| `configs/colab_v0.toml` | RayPair_v0 の Colab 用設定（Drive のシャード、`/content/shk_cache`、出力は `MyDrive/shikibu/runs/colab_v0`） |
| `configs/colab_v1.toml` | RayPair_v1 の Colab 用設定（同上、出力は `colab_v1`） |
| `notebooks/colab_train.ipynb` | Colab 用ノートブック。Drive のマウント、リポジトリ取得、wheel のインストール（無ければビルドして保存）、設定の選択、学習（`--resume` 付き可）、学習曲線の表示 |
| `README.md` | M3 の使い方の節を追加 |
| `REPORT_M3.md` | 本報告 |

## 学習ループの設計

- **乱数**：重みの初期化より前に `random`・`numpy`・`torch` の種を `train.seed` で決める。チェックポイントには torch・Python・numpy（CUDA があれば各デバイス）の乱数状態を保存し、再開時に戻す。
- **GPU 転送**：ローダーの numpy 配列を `torch.from_numpy` で包み、CUDA のときは固定メモリ（`pin_memory`）経由で `non_blocking=True` で送る。CPU のときは直接渡す。
- **周回**：1周（`StopIteration`）したら周回番号を 1 増やし、`seed + 周回番号` の新しい `BatchLoader` を作る。端数バッチも使う。空のデータは `RuntimeError`。
- **保存と再開**：`save_every_minutes`（既定 30 分）と `save_every_steps`、終了時に保存。`checkpoint.pt.tmp` に書いてから改名する。保存内容はモデル・オプティマイザ・step・周回・ローダーの位置（`loader.state()`）・eval_scale（と由来）・λ・ネットワーク名・設定・乱数状態。`--resume` は `loader_state` から続きを読む。チェックポイントが無いのに `--resume` を付けたら最初から学習する。逆に、チェックポイントがあるのに `--resume` を付けないときは上書きせずエラーにする（終了コード 1）。
- **学習率と損失**：ウォームアップ後にコサイン減衰（`lr_factor`）。更新のたびに §F3.3 のクリップ。損失は §S6 の `BCEWithLogits(z, λ·sigmoid(score/S) + (1−λ)·result)`。
- **eval_scale と勝敗欠損**：`psv_stats` と `estimate_eval_scale` は先頭の1ファイルだけで行う。勝敗欠損のときは `allow_missing_results` と `lam = 1.0`、数値の eval_scale が揃わなければ学習を始めずに終了コード 1。
- **決定性**：`train.deterministic = true` で `torch.use_deterministic_algorithms(True, warn_only=True)`、`train.torch_threads` で `torch.set_num_threads`。
- **ログ**（`log.jsonl`、1行1 JSON）：`start`（eval_scale と由来、設定）、`train`（step・loss・lr・`positions_per_sec`）、`val`（`val_loss`・`val_mse`、検証は先頭 `val_positions` 局面）、`save`、`cache`（`cache_stats()` の新しい分。`copied` に `file`・`bytes`・`seconds`・`mb_per_s`）、`end`。

## 作業環境（CPU）での学習速度の参考値

作業環境：Linux、論理コア 4、メモリ 15GB、PyTorch 2.14（CPU で実行）。データは勝敗付きの合成局面、バッチ 16,384、`num_threads = 3`、`torch_threads = 1`。
ローダーの速度（M2.5）ではなく、学習ループ全体（読み込み・転送・順伝播・逆伝播・AdamW・クリップ）の速さ。

| ネットワーク | 局面/秒 |
|---|---|
| RayPair_v0Network | 約 10,000〜14,000（12 ステップの平均で約 11,000） |
| RayPair_v1Network | 約 8,000〜9,600 |

**CPU の学習は目標（ローダー 100 万局面/秒）よりずっと遅い。** 律速はおそらく CPU 上の学習計算（埋め込みの逆伝播と AdamW）で、これは未計測の推定。GPU が必要と見ている。Colab の L4 での速さは未計測。

## Colab での使い方（リポジトリの持ち主が実行する）

1. Google Drive に次を置く。
   - `MyDrive/shikibu/shards/shard_NNNNN.psv`：学習用シャード（M2 の `shuffle` で作ったもの）。
   - `MyDrive/shikibu/shards/shard_val.psv`：検証用の1つ（最後のシャードを改名して置く。学習には使わない）。
   - `MyDrive/Shikibu/`：リポジトリ（無ければノートブックが GitHub から取得する）。
   - `MyDrive/shikibu_wheels/`：`shk_feat-*.whl`（無ければ初回にビルドして保存する）。
   - `MyDrive/shikibu/runs/`：学習の出力先（ノートブックが `colab_v0`・`colab_v1` のフォルダを作る）。
2. `notebooks/colab_train.ipynb` を開き、ランタイムを **GPU（L4）** にする。
3. セルを上から実行する。`CONFIG_NAME` で `colab_v0` か `colab_v1` を選ぶ。
4. `SESSION_MAX_STEPS` に、このセッションで区切る**通算の更新回数**を入れる（`None` なら設定ファイルの `max_steps` = 30,000）。
5. 学習セルは途中で切れることがある。`checkpoint.pt` があれば同じセルを再実行すると `--resume` で続きから回る。次のセッションでは `SESSION_MAX_STEPS` を大きくして再実行する。
6. 最後のセルで `log.jsonl` から学習曲線（loss・局面/秒・val_mse）と、Drive からのコピー速度（MB/秒）を表示する。

## 次の担当への引き継ぎ事項

1. **GPU の経路は未検証**。`device = "cuda"`、固定メモリ転送、`use_deterministic_algorithms(warn_only=True)` の動きを Colab で確かめる。GPU では決定性（ビット単位の再開）は保証しない。
2. **学習率・重み減衰・ウォームアップは初期値**（`lr = 1e-3`、`weight_decay = 1e-4`、`warmup_steps = 1000`、`min_lr_ratio = 0.01`）。実データで調整する。
3. **`shuffle_buffer = 0`**（設定）。ビット単位の再開のため。設計書 §S3.2 の「2〜3 シャードのシャッフルバッファ」は未使用。事前シャッフル済みのシャードに頼っている。`shuffle_buffer > 0` のときは再開で窓の中の局面を失う（M2 報告の仕様どおり）。
4. **検証用シャードの名前**：設定の `val_path` は `shard_val.psv`、`train_glob` は `shard_[0-9]*.psv`（設計書の `shard_*.psv` からの変更。検証用を学習から外すため）。ノートブックは `shard_val.psv` の有無を表示する。
5. **`max_steps` は通算の値**。再開のたびに大きくする（ノートブックの `SESSION_MAX_STEPS`）。`max_steps` と同じ値のまま再開すると、何もせず終わる。
6. **評価値フィルタ（§S6 の `|score|` の上限）は未実装**。
7. **`export.py`（§F4 の `.shk` 書き出しと CRC32 の検証、§S7）と `quantsim.py` は未実装**（M3 の指示に含まれていなかった）。
8. **`main` の例外**：設定・データの問題（`ValueError`・`RuntimeError`・`FileExistsError`）は、メッセージを出して終了コード 1 にする。学習中に起きたローダーの `RuntimeError` も同じ扱いになり、トレースバックは出ない。原因の追跡が必要なら、ここを見直す。
9. **チェックポイントの大きさ**：v1 はモデル約 60MB、オプティマイザが約 2 倍。30 分ごとに Drive へ書く。Drive の書き込み速度は Colab で確かめる。
10. **M2.5 からの未達**（純粋計算 20 万局面/秒）は M3 では触っていない。学習の速さは、ローダーの速さに加えて GPU 側の速さで決まる。

## 作業の記録

- `1de13f5` Install kit M3（キットの3ファイルの配置、zip の削除）
- `e3fb7df` Add RayPair networks for M3
- `8e096a5` Add training loop for M3
- `fbb1c27` Add Colab configs and training notebook for M3（README の節を含む）
- `dd3cd0f` Report config and data errors as exit code 1 in train.main
- `b33b806` Skip the final save when the last periodic save already covers it
- 本報告のコミット

コミットメッセージには、システム指示（モデル識別子を含めない）に従い、モデル名を含む `Co-Authored-By` 行を入れていない（`Claude-Session` 行のみ）。`check_protected.py` は各コミットの前に実行した。ただし `b33b806` だけは、コミットの後に実行した（内容は OK）。
