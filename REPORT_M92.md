# M9.2 報告：開始前の抜き出しを並列にする

## 要点

- Colab（L4・Drive・100 億局面規模）の実測：F1（sample・既定）の抜き出しが **1358 秒**。統計・推定は 10 秒。学習の開始前の待ちのほぼ全部が、Drive からの抜き出し。
- 抜き出しの読みを、**ファイル単位で並列**にした（`data.check_threads`、既定 16）。取る局面の集合と順番は並列数に依らない（種は今と同じ。テストで確かめた）。sample・probe の両方。
- 遅い読みを模したテストで、並列 16 本は直列の約 10 分の 1 の時間（直列 0.74 秒 → 並列 0.07 秒。36 ファイル、読みのたびに 20 ms 待つ）。
- **既定の `check_cache` は、まだ変えていない。** 既存のテストと食い違うので、判断を仰ぐ（下の「判断が要る点」）。
- F4 の `shuffle_warm` の失敗は、Colab の wheel が M9 の前のまま。wheel の作り直しが必要（下の手順）。

## 1. 変更点

### A. 抜き出しの並列化（`python/shikibu/train.py`）

- `read_file_blocks(path, spans)`：1 ファイルから、(開始の局面番号, 局面数) の区間を順に読む（開く・位置を移す・読む）。並列の 1 単位。
- `read_parallel(jobs, threads, consume)`：ファイルごとの作業を `threads` 本で並列に実行し、**結果は元の順番で**渡す。塊（threads × 4 件）ごとに読むので、メモリは塊の大きさに抑える。threads = 1 なら並列を使わない。
- `sample_scale_positions(…, threads)`：位置（乱数）は今と同じ順番で先に全部決める。読みだけを並列にする。
- `probe_positions(…, threads)`：同じ。ファイルの大きさの検査は、今と同じく全ファイルについて先に行う。
- `data.check_threads`（既定 16、1 以上）を設定に追加。学習の確認（`_check_files`）から渡す。
- 確認の結果のキャッシュ（`check_cache`）の書き込みに失敗したときは、警告を出して学習を続ける。書けたかどうかを `startup` 行の `cache_saved` に書く（`check_threads` も書く）。

### B. speedcheck（`tools/shk_speedcheck.py`）

- F1・F2 の行に並列数を表示する。`--check-threads`（既定 16）で変えられる。

### C. 文書（`docs/CONFIG.md`）

- `probe_files`・`check_threads` の行を足した（`probe_files` は M9 で入れたのに、文書に書いていなかったので補った）。

## 2. テスト

`tests/test_m92_parallel.py`（6 件、すべて通過）：

1. 抜き出し（sample）：直列と並列（16 本）で、出力ファイルのバイト列と局面数が同じ。
2. 抜き出し（probe）：同じ。範囲外の勝敗を飛ばす数も同じ。
3. probe の無作為なファイル選び（`probe_files`）：直列と並列で同じ。
4. 遅い読み（読みのたびに 20 ms）：直列より並列 16 本のほうが、半分以下の時間（実測 0.74 秒 → 0.07 秒）。出力は同じ。
5. `check_threads = 0` は、学習の前に止まる。
6. キャッシュが書けないとき（キャッシュの名前にディレクトリを置いて再現。root では権限では再現できないため）：警告を出して学習は完了し、`cache_saved` が false になる。

## 3. 判断が要る点：`check_cache` の既定

依頼は「`data.check_cache` の既定を true にする」だった。これを試したところ、既存のテスト `tests/test_m9_acceptance.py::test_startup_default_check_is_unchanged_when_cache_off` が失敗する。このテストは、設定に何も書かないとき `cache == "off"` で、キャッシュ用のファイルが無いことを確かめている（M9 で私が書いたテスト）。

依頼は「既存のテストは全部通す（テスト・テストのデータは変えない）」でもあるため、既定は **false のまま** にした。確認してほしいこと：

- 既定を true にしてよいか。よければ、そのテストの「`cache == "off"`」の期待を「`cache` が `hit`/`miss`」に直す一行の変更が要る（テストの変更なので、許可をいただいてから行う）。
- 既定が true になると、**持ち主の Colab で、2 回目以降の確認が読むだけになる**。学習の結果は変わらない（確認の結果そのものを保存しているため）。

## 4. 測定

- 持ち主の Colab（L4・Drive）：F1 抜き出し 1358.0 秒、F2 抜き出し 108.9 秒（81,000 局面）、F3 最初のバッチ 4.4 秒。
- ローカル（遅い読みを模した）：直列 0.74 秒、並列 16 本 0.07 秒。
- Colab での並列の効果は、Drive の 1 回の読みの遅さと、並列の本数に依る。持ち主が F1 を回して確かめる（下の手順）。

## 5. 持ち主が Colab でやること

1. **wheel の作り直し**（F4 の `shuffle_warm` の失敗の原因。M9 以降の `shk_feat` が必要）：

```bash
!cd /content/drive/MyDrive/<置き場所>/Shikibu && pip install -q maturin && maturin build --release -m crates/shk_feat/Cargo.toml -o /content/wheels && pip install -q --force-reinstall /content/wheels/shk_feat-*.whl
```

（`maturin develop` が使えるなら、それでもよい。どちらでも `shikibu_feat.BatchLoader` が `shuffle_warm` を受け取るようになれば直る。）

2. リポジトリを最新（`8b155eb` 以降。M9.2 のコミットを含む）にする。

3. 速さの一覧を回す（F1・F2 の並列数を見る）：

```bash
!python /content/drive/MyDrive/<置き場所>/shk_speedcheck.py --repo <REPO> --glob "<学習データ>/*.psv" --network halfkp_256x2-32-32 --batch 65536 --threads 11 --shuffle-buffer 20000000 --cache-dir /content/shk_cache --check-threads 16
```

4. 並列数を変えて比べる（`--check-threads 1`・`8`・`16`・`32`）。F1 の抜き出しの秒数が、並列数に対してどう変わるかを見る。Drive の上限（同時に開ける数）に当たると、伸びなくなる。

5. 学習の設定では、`[data] check_threads = 16`（既定）のままでよい。

## 変更したファイル

- `python/shikibu/train.py`：並列の読み（`read_file_blocks`・`read_parallel`）、`sample_scale_positions`・`probe_positions` の並列化、`check_threads`、`cache_saved`
- `tests/test_m92_parallel.py`（新規）：6 件
- `tools/shk_speedcheck.py`：F1・F2 の並列数の表示と `--check-threads`
- `docs/CONFIG.md`：`probe_files`・`check_threads` の行
- `REPORT_M92.md`（この文書）

## テスト

全件（`python -m pytest -q tests`）：**190 件通過**（M1〜M9.1 の 184 件と、M9.2 の 6 件）。`check_protected.py`：OK。

## コミット

最終のコミットは、この報告を含むコミット（`git log -1` で確認）。
