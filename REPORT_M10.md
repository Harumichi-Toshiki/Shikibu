# M10 報告：やねうら王の SFNN（レイヤースタック）への対応

## 状態（先に）

受け入れ条件は未達です。13 件のうち 11 件が合格、2 件が失敗しています。

- 失敗：`test_build_and_forward[sfnn_halfkahm2_64-7-8_k3k3]` と `test_build_and_forward[sfnn_halfka2_128-15-8_f_k3k3]`
- 原因：テストの冒頭の確認 `assert len(buckets) >= 3` が、テスト側のデータで落ちています。
  `random_boards(24, seed=10101)` と SPARSE の局面は、k3k3 のバケツが 0 と 1 の 2 種類しか出ません
  （`tests/kingzone_ref.py` の `k3k3_bucket` だけで決まる値で、実装とは無関係です）。
  cshogi 1.0.9 と 1.1.1 で同じ局面集合になることも確かめました。
- 切り分け：テストのコピー（リポジトリの外、シードだけ 7401 に変更）では、同じ 3 件（`build_and_forward` の全件）が合格しました。
  式との比較（`atol 1e-4`）の部分も含めて通るので、残りは「シード（または確認の条件）」の問題です。
- 判断が必要：保護されたテストのシードを変えるかどうかは、持ち主が決めてください。
  保護ファイル（`tests/test_m10_acceptance.py`、`tests/conftest.py`）は変更していません。

## テスト結果

- `python -m pytest -q tests`（全件）：**2 failed, 195 passed**（1 warning は既存の M9 テストの UserWarning）
  - 失敗は上の 2 件だけ。M1〜M9.2 は全件合格。
- M10 の 13 件：11 合格、2 失敗（上記）。
  - 合格：`test_reference_matches_yaneuraou`、`test_bucket_numbering`、`test_parse_arch_sfnn`、
    `test_build_and_forward`（halfkp の 1 件）、`test_clip_sfnn`、`test_training_runs`、
    `test_export_sfnn_exact`（2 件）、`test_exported_eval_matches_float`、`test_cli_edition`、`test_lab_describe_sfnn`
- `python check_protected.py`：**OK**

## 変えたファイル

- `python/shikibu/networks.py`：`sfnn_<特徴>_<W>-<H1>-<H2>[_f][_k3k3]` の読み取り（`_parse_sfnn`）、`SFNNNetwork`（FT は従来と同じ経路、その後ろは SFNN の式）、`build` の振り分け。従来の構造は `"sfnn": False` を足しただけ。
- `python/shikibu/nnbin.py`：LEB128 の符号化・復号（numpy 版。参照実装と一致することを確かめた）、SFNN の nn.bin の書き込みと読み直し。
- `python/shikibu/export.py`：`_export_sfnn`（量子化・スタックの並び・読み直しの確認）、`nnbin_support` / `yo_arch` / `yo_edition` の SFNN 対応。従来の書き出しは変えていない。
- `python/shikibu/lab.py`：プリセット `sfnn_halfkahm2_512-15-32_f_k3k3` の追加、`describe` の SFNN 対応（パラメータ数、yo_arch、yo_edition）。
- `notebooks/shikibu_lab.ipynb`：比べる構造の一覧に SFNN を追加、書き出しの説明を更新（JSON の 4 行の変更だけ）。
- `PROTECTED.sha256`・`tests/sfnn_ref.py`・`tests/test_m10_acceptance.py`・`tests/golden/sfnn_golden.json`：キットの導入（手順1）。
- `shikibu_kit_m10.zip`：削除（キットの導入後）。
- 変更なし：`train.py`（既存の `clip_weights_` が `model.clip_` を呼ぶので、SFNN にも効く）、Rust（コードは変えていない。wheel を作って入れた）。

## 数値

- `test_exported_eval_matches_float`（テストと同じ 200 局面、構造 `sfnn_halfkahm2_64-7-8_f_k3k3`、学習 60 歩、eval_scale 600、FV_SCALE 13）：
  - 平均のずれ **10.80 cp**、最大のずれ **56.61 cp**（閾値は平均 50、最大 300）
  - 傾き（nn.bin の評価値を float の評価値に対して回帰）：1.003
- k3k3 の順伝播：float の式との比較（3 構造、バケツ 4〜5 種類を含む局面）で、最大の差 1e-5 以内。

## 速さ（CPU、4 スレッド、バッチ 4096、学習 1 歩＝順伝播・逆伝播・更新、乱数の特徴 30 個/局面）

- `sfnn_halfkahm2_512-15-32_f_k3k3`：約 **12,200 局面/秒**
- `halfkahm2_512x2-16-32_f`（従来の構造）：約 12,000 局面/秒
- GPU の値は持ち主が測ってください。実際のデータでは特徴の数が違うので、この値は目安です。

## 設計の選択

- k3k3 の層：局面をバケツで並べ替え、組ごとに 1 回ずつ掛ける。全部の組で計算してから選ぶ方法の約 1/9 の計算量。並べ替えのため 1 回の順伝播で同期が 1 回入る。
- FT は ArchNetwork と同じ経路を使う（M8 の因数分解、M9 の疎な計算、M9.1 の triton の分岐）。既定の動作は変えていない。
- weight decay は層の重み（l0・l1・l2）だけに掛ける（従来の全結合の重みと同じ扱い）。
- nn.bin の構造の文字列は yo_arch の形（例：`SFNN_halfkahm2_512_15_32_k3k3`）。読み込みでは使われない（参照実装と同じ）。
- `lab.describe` の SFNN：`L1` に H1、`L2` に H2 を入れる。`nnbin_bytes` は None（LEB128 の大きさは重みの値で決まるため）。
- バケツ：`zone_of`（ArchNetwork と同じ関数）。`kingzone_ref.k3k3_bucket` と一致することを `test_bucket_numbering` で確認。

## 環境の注意

- ブランチ：ユーザーの指示どおり `main`（adccc47）から `claude/m10-sfnn` を作った。
- `download.pytorch.org` はプロキシで 403 のため、PyPI の torch 2.15（CUDA ビルド、GPU なし）を使った。cshogi は 1.1.1。
- キットの導入の途中で、zip を含めたコミットが失敗し、ブランチが一度 `main` と同じ commit で push された。その後、正しいコミット `03279de` で更新した（履歴の書き換えはなし）。

## コミット

- `03279de` Install kit M10
- `3e16175` Add SFNN architecture name and model (M10)
- `fe10119` Export SFNN nn.bin with layer stacks (M10)
- `8910ec3` Add SFNN preset and Colab lab notebook option (M10)
- 本報告のコミット（下の最終コミットを参照）
