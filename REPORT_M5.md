# M5 報告：ドキュメント整備

## 結果

**完了。** 指定された7つの文書をそろえ、`README.md` を書き直しました。コードは一切変更していません（`.py`・`.rs`・`.toml`・`.ipynb` の差分は 0 件）。
中断の条件には当たりませんでした。ツール呼び出しはおよそ 80 回（上限 120 未満）です。

## テスト結果

作業の前後で確認しました。

```
$ python -m pytest -q tests
73 passed in 66.30s (0:01:06)

$ python check_protected.py
OK
```

- 作業の前にも同じ結果を確かめました（`73 passed in 62.27s`、`OK`）。
- 環境：Python 3.13、PyTorch 2.14.1（CPU で実行）、cshogi、numpy、pytest、maturin。仮想環境はリポジトリの外（作業用の一時フォルダ）に作りました。
- README の旧い手順は、`pip install cshogi pytest numpy maturin` だけでした。M3・M4 のテストは PyTorch も要るので、足りませんでした（確かめた：`No module named 'torch'`）。新しい README と `docs/OPERATIONS.md` の 8.1 にその旨を書きました。

## 手順0：作業ブランチ

`python/shikibu/export.py` は、作業ブランチ `ccr-d54f3f12-elu789` に既にありました（M4 の PR #6 の取り込みが入っていた）。取り込みは行っていません。

## 書いた文書

| 文書 | 中身 |
|---|---|
| `README.md`（書き直し） | Shikibu の概要と、Dajo・Nuenagon・Dainagon との関係、できること、構成、Colab と手元の PC の始め方（手元の手順は実際に実行して確かめた）、文書の一覧。 |
| `docs/GUIDE_COLAB.md` | Colab で学習を回す手順を、ノートブックの節の順に。Drive の配置、wheel の作成と再利用、設定と `SESSION_MAX_STEPS`、`mon.start`・`mon.watch`・`mon.stop`、`.shk` の書き出し、量子化の確認、エンジン用の正解データ、切れたときの再開。 |
| `docs/CONFIG.md` | 設定の全キー（[data]・[model]・[train]・[output]）。型・既定値・意味・設計書の節。検査の規則、勝敗欠損の4通りの組み合わせ（実行して確かめた）、再開の規則、検査されていない値と落とし穴。 |
| `docs/FORMATS.md` | PSV（40 バイト）、`.shk` のヘッダ（180 バイト）と本体の並び、フッタ、`checkpoint.pt` のキー、`log.jsonl` の行の種類と項目、`status.json`、エンジン用の正解データ（JSON）、ツールの出力。バイト位置と項目は、実際に書き出したファイルで確かめた。 |
| `docs/ARCHITECTURE.md` | Rust のクレートと Python のモジュールの役割、データの流れ、ローダーの係と枠、コピー係とキャッシュ、再開の仕組み、周回ごとのファイル順、2 パスのシャッフル、学習の1ステップ、進捗の表示、設計書 §S1 との違い。 |
| `docs/OPERATIONS.md` | `mon.watch()` を止めないこと、切れたあとの再開（何が消え、何が残るか）、二重起動の危険（実際に再現した）、教師データの性質、引き分けと eval_scale の関係（実験で確かめた）、速さの実測（出どころを明記）、エラーの対処の表。 |
| `docs/DEVELOPMENT.md` | 変更禁止ファイルと `check_protected.py`、キット（zip）による更新の手順、テストの実行、マイルストーン M1〜M5 の経緯、未解決の課題（報告書の引き継ぎを集めたもの）、この作業で見つけたコードの気になる点。 |

各文書はコミットごとに分けました。

| コミット | 内容 |
|---|---|
| `cdf3936` | `docs/CONFIG.md` |
| `f94a543` | `docs/FORMATS.md` |
| `309f257` | `docs/ARCHITECTURE.md` |
| `2bd850d` | `docs/OPERATIONS.md` |
| `eb32072` | `docs/GUIDE_COLAB.md` |
| `5d954f2` | `docs/DEVELOPMENT.md` |
| `0c257d4` | `README.md` |
| （本コミット） | `docs/GUIDE_COLAB.md` の修正、`REPORT_M5.md` |

各コミットの前に `python check_protected.py` を実行し、すべて `OK` でした。すべて `origin/ccr-d54f3f12-elu789` に push しています。

## 確かめたこと

いずれも、この開発環境（Linux・CPU・小さなデータ）で実際に実行したものです。

- CLI のヘルプ：`tools/shk_tools.py`（と `shuffle`）、`python -m shikibu.train`・`export`・`quantsim`、`tools/make_engine_golden.py`、`bench/bench_loader.py`。
- `shk_tools stats`・`estimate-scale`・`shuffle`（12,000 局面の人工データ）。`shuffle` を空でないフォルダに出すと、エラーになることを確かめた。
- 学習：20 ステップの通し、`--resume` で 30 ステップまで続くこと、`checkpoint.pt` があるのに `--resume` を付けないと終了コード 1 で止まること。`log.jsonl`・`status.json`・`checkpoint.pt` のキーと項目。
- 勝敗欠損の4通り（`allow_missing_results`・`lam`・`eval_scale` の組み合わせ）が、仕様どおりに止まるか学習するか。
- 監視ツール `shk_monitor` の `start`・`stop`・再開（ローカルで実行。Colab の `/content` のパスだけ差し替えた）。50 ステップごとの保存で、90 ステップで止めると 50 から再開した（40 ステップ分が失われる）。
- `export`（書き出しと読み直し）、`quantsim`、`make_engine_golden`（50 局面）。ヘッダのバイト位置、本体とファイルの大きさ（v0：7,698,300 バイト）、CRC32 の検出（1 バイトの破損で拒否）、正解データの `shk_sha256` の一致。
- `log_every_steps = 0` で `ZeroDivisionError` が出ること。
- 二重起動：1 つ目の学習が動いている最中に、別のプロセスから `start` すると、2 つの学習が同時に動き、2 つ目は `--resume` で保存点から再開すること。
- 引き分けと eval_scale：真の値 600 の人工データで、引き分けの割合 0%・30%・60% の推定値は 605・1,617・3,345。
- 勝敗と評価値が無関係なデータでは、推定値が上限 5,000 に張り付き、警告が出ること（4,999.8）。
- `scale_sample = 5000` の指定で、実際に使った局面数は 4,999 だった（整数の割り算の端数。害は無い）。
- `maturin build --release -o <フォルダ>` で wheel が作れること（ノートブックの節 3 と同じコマンド）。別の Python 版の名前の wheel は、`pip` が「対応していない wheel」として拒否すること。

## 確かめられなかったこと

以下は、Colab でしか確かめられないか、持ち主の情報にもとづくものです。文書の中でも、そう書いています。

- Colab の上での動作：GPU の経路（`device = "auto"`、固定メモリ経由の転送）、ノートブックの各セル（Drive のマウント、rustup、`git clone`）、`mon.watch()` の表示（IPython とグラフ）。
- ランタイムの切断の挙動：切断後に裏のプロセスが残るか、どのセルの実行中が「操作中」とみなされるか。文書には、持ち主の報告として書いた。
- 速さ：Colab L4 の約 100 万局面/秒と、Drive のコピー約 78MB/秒は、持ち主の実測です。この環境では再現していません。開発環境の性能（`perf_check.py --profile dev`）は、再計測していません（REPORT_M2.5 の値を引用）。
- 教師データの中身：Drive にある 4 ファイル・約 19.6 億局面・シャッフル済み・勝敗ありは、持ち主の情報です。水匠10 の評価値の尺度が違うかどうかも、確かめていません。
- 実データでの eval_scale：引き分けの現れ方は人工データで確かめただけで、実際の教師データでは確かめていません。
- 再開時の「設定の値が無視される」ことと、`save_every_minutes = 0` の挙動、`train.main` が捕まえない例外：コードを読んだ結果であり、実行はしていません（文書には「コードの読み」と明記）。
- エンジン用の正解データ：`cshogi` の版が違うと局面が変わる可能性は、確かめていません。

## 文書を書く中で見つけた、コードの気になる点（直していません）

詳しくは `docs/DEVELOPMENT.md` の 5.5 に書きました。主なものは次のとおりです。

1. **二重起動（確かめた）**：`shk_monitor.start` は、このプロセスで起動した学習しか覚えていない。カーネルの再起動の後に `start` すると、動いている学習と同じ出力フォルダで、もう1つ学習が始まる。
2. **`log_every_steps = 0`（確かめた）**：`ZeroDivisionError` のトレースバックで止まる。`validate()` で検査していない。
3. **`save_every_minutes = 0`（コードの読み）**：0 にすると毎ステップ保存になる。「保存しない」の意味ではない。
4. **`shk_tools shuffle` の出力先（確かめた）**：空でないフォルダだと `ValueError` のトレースバックが出る。ほかのコマンドは「エラー:」で整えて表示している。
5. **`train.main` の例外（コードの読み）**：`ValueError`・`RuntimeError`・`FileExistsError` しか捕まえない。壊れた checkpoint による `KeyError` などはトレースバックで止まる。
6. **再開の規則（コードの読み）**：再開すると `eval_scale` の設定は無視され、勝敗欠損の判定はやり直さない。ファイル数が同じで中身の違う `train_glob` は見つからない。
7. **`.gitignore` に `.venv/` が無い**：README の旧い手順は `.venv` を勧めていた。
8. **テストのコメント（変更禁止）**：`tests/test_m3_acceptance.py` の冒頭に「先頭の 1 ファイル」と書かれているが、M4 で全ファイルからの抽出に変わった。
9. **ノートブックの取り込むブランチ**：`BRANCH = "claude/magical-wright-5hukjw"` に固定。ブランチを消すと取得が失敗する。wheel は、Python の版を見ずに一番新しいものを選ぶ。
10. **設計書との違い**：§S1 の `shk_loader`・`shk_tools` のクレート、`features.py`、`networks/` のフォルダは作られていない。検証用シャードの自動除外（§S3.1 の 4）と評価値フィルタ（§S6）も未実装（`docs/ARCHITECTURE.md` の 10）。

## 次の担当への引き継ぎ

- `docs/DEVELOPMENT.md` の 5 を参照。特に、教師ごとの eval_scale（水匠10 を混ぜるとき）、検証用データの用意、GPU の経路の確認（Colab）、二重起動の防止を優先してください。
- ドキュメントに書いた Colab の手順は、次に Colab で1回回したときに、実際の手順と合っているかを確かめてください（`docs/GUIDE_COLAB.md`）。
- この作業ではコードを直していません。上の 1〜6 は、直すかどうかを次の担当が決めてください。

## 作業の記録

- コミットは上の表のとおり（`docs/` の 6 文書と `README.md`）。
- コミットメッセージには、`Claude-Session` 行を入れています。モデル名を含む `Co-Authored-By` 行は、システムの指示（コミットにモデル名を入れない）に従って省きました。
