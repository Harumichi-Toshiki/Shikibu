# REPORT_M4：書き出し・量子化・進捗表示

## 結果

**完了**。受け入れテストは 73 件すべて通過（M1 34 ＋ M2 15 ＋ M3 13 ＋ M4 11）。
M3 のチェックポイントから M4 のコードで `--resume` して学習が続くことも確認した（下記）。
中断条件には当たらなかった。

Colab 上の本番学習は、この環境からは実行していない（GPU と Drive が無い）。手順と設定は用意したが、起動時間・速度・ETA の精度は**未確認**。

## テスト

```
$ python -m pytest -q tests
73 passed in 63.09s (0:01:03)
```

内訳：M1 34（`test_m1_acceptance.py`）、M2 15（`test_m2_acceptance.py`）、M3 13（`test_m3_acceptance.py`）、M4 11（`test_m4_acceptance.py`）。

```
$ python check_protected.py
OK
```

## M3 チェックポイントからの再開

1. 変更前の M3 のコードで、30 ステップ学習した（`max_steps = 30`、`total_steps = 60`、バッチ 256、ダミーデータ 2 ファイル中 1 ファイル）。`checkpoint.pt` はリポジトリの外（scratchpad）に保存した。
   保存時点：step 30、epoch 2、キーは `config, epoch, eval_scale, eval_scale_source, lam, loader_state, model, network, optimizer, rng, step`。
2. M4 のコードで `--resume`（`max_steps = 60`）。
   - `log.jsonl` の末尾の start 行：`resumed = true`、step 30、epoch 2 から続行。
   - end 行：step 60、epoch 4。
   - `status.json`：`phase = done`、`history = [train, done]`、`progress = 1.0`。
   - 再保存された `checkpoint.pt` には `file_perm`（追加されたキー）が入り、M3 のキーはすべて残っている。
3. 続けて、この checkpoint から `.shk` を書き出し、量子化の確認と正解データ作りも動くことを確かめた（下の「使い方」の例と同じ CLI）。

## 作ったファイル・変えたファイル

| ファイル | 種別 | 役割 |
|---|---|---|
| `python/shikibu/export.py` | 新規 | `export_shk`（checkpoint → §F4 の `.shk`、書き出し後に読み直して確認）、`read_shk`（不正なら ValueError）、CLI |
| `python/shikibu/quantsim.py` | 新規 | `quantize`・`qforward`・`cp_of`・`compare`（§E4 の量子化を Python で再現、float 版との差と損失を測る）、CLI |
| `tools/make_engine_golden.py` | 新規 | Nuenagon の N3 用の正解データ（JSON、形式 `shikibu-engine-golden-v1`）を作る CLI |
| `python/shikibu/train.py` | 変更 | ① 勝敗確認と eval_scale 推定を、全ファイルから抜いたランダムブロックで行う（`scale_sample` が上限）② 画面の進捗行（`progress_every_sec`）③ `log.jsonl` の `progress`・`eta_sec`、`phase` 行 ④ `status.json` ⑤ 周回ごとのファイル順の並べ替え（`seed + 周回番号`）と checkpoint の `file_perm` |
| `configs/colab_v0.toml` | 変更 | バッチ 65,536、shuffle_buffer 20M、scale_sample 10M、num_threads 11、`train_glob` を PonkotsuWCSC35 に、`total_steps`・`max_steps` を 179,000 に |
| `configs/colab_v1.toml` | 変更 | 同上（v1 のネットワーク） |
| `notebooks/colab_train.ipynb` | 書き換え | `mon.start` → `mon.watch` → `.shk` 書き出し → 量子化の確認。各セルに日本語の説明。ブランチは `claude/magical-wright-5hukjw` |
| `docs/DESIGN.md`、`PROTECTED.sha256`、`tests/m4_ref.py`、`tests/test_m4_acceptance.py`、`tools/shk_monitor.py` | キットで導入 | 手順 1（c710f98）。変更禁止の区画は触っていない |
| `shikibu_kit_m4.zip` | 削除 | 手順 1 の 4（導入後に削除） |
| `REPORT_M4.md` | 新規 | この報告 |

Rust 側（`crates/`）は変更していない。チェックポイントはキーの追加（`file_perm`）だけで、既存のキーの削除や意味の変更はない。

## コミット

```
c710f98 Install kit M4
1d307cb Add .shk export and reader (§F4)
f092641 Add quantization check tool (§E4, §S7)
e230f7c Add engine golden data tool (Nuenagon N3)
4f47024 Estimate eval_scale from random blocks of all files (§S4)
4cfcec5 Show training progress and write status.json (train.progress_every_sec)
088e654 Reorder training files at each epoch start (seed + epoch)
ac2fcac Update Colab configs for the full PSV set
1738157 Rewrite Colab notebook around the monitor tool
41d6f90 Write the scale sample to the local temp directory
```

各コミットの前に `python check_protected.py` を実行し、すべて `OK`。

## 使い方

### 手元の PC（Colab と同じコード）

```
pip install maturin cshogi pytest numpy torch
maturin develop --release -m crates/shk_feat/Cargo.toml
export PYTHONPATH=python

# 書き出し（読み直して確認する）
python -m shikibu.export --checkpoint <run>/checkpoint.pt --out <run>/colab_v0.shk

# 量子化の確認（先頭 n 局面。float 版との差と損失の差を表示）
python -m shikibu.quantsim --shk <run>/colab_v0.shk --psv <file.psv> --n 100000

# エンジン用の正解データ（ランダムに指し進めた局面。seed で決まる）
python tools/make_engine_golden.py --shk <run>/colab_v0.shk --out golden.json --n 1000 --seed 1
```

学習は `python -m shikibu.train --config configs/colab_v0.toml [--resume]`。進み具合は `<out_dir>/status.json` と `log.jsonl` に出る。

### Colab

`notebooks/colab_train.ipynb` を上から順に実行する。

1. Drive のマウント → リポジトリ（ブランチ `claude/magical-wright-5hukjw`）→ wheel
2. 設定の選択（`CONFIG_NAME`、`SESSION_MAX_STEPS`）
3. `mon.start(CONFIG, repo=REPO)`：裏で学習を開始（checkpoint があれば自動で `--resume`）
4. `mon.watch(CONFIG)`：進み具合の表示（表示を止めても学習は続く）。止めるときは `mon.stop()`
5. `.shk` の書き出し（`MyDrive/shikibu/runs/<設定名>/<設定名>.shk`）
6. 量子化の確認（`shikibu.quantsim`）

### 画面と status.json

- 画面：`[ 12.3%] step 1,234 / 179,000  残り 1時間02分  103.1万局面/秒`。`progress_every_sec` 秒ごと（既定 30）。0 のときは `log_every_steps` ごと。
- `status.json`：`{"phase", "history", "step", "max_steps", "progress", "eta_sec", "updated"}`。段階は `data_check` → （`eval_scale`）→ （`prefetch`）→（`buffer_fill`）→ `train` → `done` または `error`。`history` は、この実行で入った段階の順。
- `log.jsonl`：先頭は従来どおり `start`。`prefetch`・`buffer_fill`・`train` は `{"type": "phase", "name": …}` の行。train 行に `progress` と `eta_sec`。

## 判断したこと（確認してほしい点）

1. **勝敗確認・eval_scale 推定の局面**（§S4、`train.scale_sample`）：全学習ファイルから、ファイルの大きさに比例した数のランダムな連続ブロック（1 ブロック 1 万局面）を取る。局面数が上限以下なら全件を使う。抜いた局面は `/tmp`（Colab ではローカル）の一時ファイルに書き、終わったら消す。`start` 行の `scale_positions_used` に実際の数を書く。
   - M3 のテストの冒頭には「先頭の 1 ファイルだけ」と書いてあるが、テストファイルは変更禁止なので直していない。M4 の仕様（テスト冒頭のコメント）に従い、この方式にした。
2. **周回ごとのファイル順**：各周回の始めに `random.Random(seed + 周回番号)` でファイル順をシャッフルして `BatchLoader` に渡す。順番は checkpoint の `file_perm`（ファイル番号の並べ替え）に保存する。
   - **M3 の checkpoint（`file_perm` が無い）から再開したときは、その周回の残りを M3 と同じ設定の順のまま続ける。** 次の周回からシャッフルされる。途中の位置（loader_state）がファイル番号を指すため、順番を途中で変えると位置がずれるから。
3. **ETA**：この実行の train 段階の平均速度（最初のバッファ充填は含めない）から計算する。`--resume` 直後は、再開してからの速度で見積もる。
4. **段階の記録**：`status.json` は実行ごとに作り直す（`history` は前の実行を引き継がない）。`prefetch` は `cache_dir` を使うときだけ。`buffer_fill` の間に最初のバッチを先に取り、`train` に入る。
5. **量子化の確認の評価値の差**：`cp_float = z_float × eval_scale`（丸めなし）と、エンジンと同じ丸め `cp_quant = round(raw × eval_scale / 8128)` の差を測る。
6. **丸め**：`quantsim.rnd` は「0 から遠い方への四捨五入」を、`floor(|x|) + (|x| − floor(|x|) ≥ 0.5)` で書いた（テストの参照実装は `floor(|x| + 0.5)` で、テストの値では一致する）。
7. **正解データの局面**：cshogi の合法手からランダムに指す。200 手で打ち切り、詰みや手が無いときは初期局面から指し直す。特徴は `shikibu_feat.extract_sfen`（Rust）で作る。`cp` はエンジン規則どおりクリップしない（詰みの値との重なりは、エンジン側で §E4 の ±(詰み値−1) に収めること）。
8. **`configs` の `val_path`**：`shard_val.psv` の配置はもう無いので、空にした（検証は行わない）。`val_every_steps` も 0 にした。検証用の PSV を用意したら指定し直す。
9. **int16 のオーバーフロー**：`qforward` は蓄積値が int16 の範囲を超えたら ValueError にする（テストの参照実装と同じ）。ただし、ft の重みは ±127 以内、特徴は 1 視点あたり最大 208 個なので、理論上の最大は約 26,500 で、起きない。

## 検証の数字（参考）

M3 の checkpoint（30 ステップ、ダミーデータ、2000 局面）で `quantsim` を回した結果：

```
z の差（float と量子化）  平均 0.014251 / 最大 0.036326
評価値の差（cp）          平均 8.55 / 最大 21.59
損失  float 0.570226 / 量子化 0.573611  （差 +0.003385）
```

学習が 30 ステップだけの小さなモデルなので、本番のモデルの量子化の誤差とは違う。本番の確認用に使う。

## 次の担当への引き継ぎ

1. **Colab で本番を動かして計測する**：起動（`data_check` の時間。以前は先頭ファイルの全読みで約 10 分）、`scale_positions_used`、ETA の精度（学習曲線の速度）、Drive へのコピー速度（`prefetch`）。
2. **検証用の PSV を用意する**：今の `configs` は `val_path` が空。量子化の確認（`quantsim`）も学習に使ったファイルで行っているので、学習に使わない局面で確かめるのが望ましい。
3. **Nuenagon（N3）**：`make_engine_golden.py` の JSON は `shk_sha256` で対応する `.shk` を確かめられる。評価値は `cp` にクリップが無いので、エンジン側で ±(詰み値−1) に収めること。
4. **GPU 経路は未確認**：テストは CPU（決定的）で行った。L4 での動作（`device = "auto"`）は Colab で確認すること。
5. **`.shk` の書き出しのタイミング**：`checkpoint.pt` は一時ファイルから改名して保存するので、学習中でも読めるが、そのときの `positions` は保存時点の値。
6. **DESIGN.md は v1.1（キットの版）のまま**。凍結区画（第F部）は変更していない。
