# M8 報告：学習の高速化

## 結果

**完了**（中断はなし）。CPU での確認は全件通過。**CUDA（GPU）の速さは、この環境では確かめていない**（後述）。

- 手順 1（キットの導入）は完了。SHA-256 の照合、`KIT.json` の各ファイルの照合を経て、`Install kit M8` でコミット（`15f5970`）。`check_protected.py` は `OK`。
- 参考実装 `reference/m8_proto.diff` は、そのまま当たった（`git apply --check` 成功）。その後、forward ごとの計算の回数だけ直した（下の「仕様の解釈」）。
- 作業ブランチ：指示の「`main` から作る」は、この時点で `claude/charming-archimedes-mfqzbh` が `main`（`1d631b3`）と同じ commit を指していたので、そのブランチで作業した。

## テストの結果

```
$ python -m pytest -q tests
151 passed in 247.40s (0:04:07)

$ python check_protected.py
OK
```

内訳は M1〜M7 の 142 件と、M8 の 9 件（`tests/test_m8_acceptance.py`）。M8 の 9 件だけの実行も通った（`9 passed in 10.41s`）。

## 2 の重ね方（読み込みと GPU の計算）

- 学習のステップは、次のバッチを読んで送り始めてから、今のバッチの計算を積む。順番は `読む(k+1) → 計算(k)`。GPU が k−1 の計算を進めている間に、CPU が k+1 を読める。
- CUDA では、転送専用のストリームと、固定メモリの作業領域 2 面（交互に使う）を使う。送り終わりは event で示し、計算の前に `wait_event` で待つ（`Prefetcher` in `train.py`）。毎バッチの `pin_memory()`（これが端末の同期を起こしていた）はやめた。
- 先読みの位置は、バッチと一緒に `(epoch, file_perm, loader_state)` で持ち歩く。checkpoint には「学習したバッチの位置」だけを書く。先読みしたまま止めても、`--resume` は次のバッチから正しく続く（M3 の再開のテストが通ることで確かめた）。
- 決定的モードの再現性：転送の順番は変えたが、計算の順番と値は変えていない（`test_deterministic_runs_identical` が通る）。

## CPU での速さ（参考）

- 条件：この環境の CPU（論理 4 コア）、`torch_threads = 4`、`num_threads = 2`、バッチ 4096、60 ステップ。データは 6 万局面（PSV 2 ファイル）。学習の `log.jsonl` の `positions_per_sec` を、ステップ 6 以降で平均した。
- 変更前は `15f5970`、変更後は `426576f`（同じデータ・同じ設定で、同じ最後の損失になることを確認）。

| 構造 | 変更前 | 変更後 | 最後の損失（両方） |
|---|---|---|---|
| `halfkp_256x2-32-32_f` | 17,096 局面/秒 | 13,721 局面/秒（約 20% 遅い） | 0.469 |
| `raypair0_128x2-16-16` | 57,833 局面/秒 | 60,053 局面/秒（約 4% 速い。ばらつきの範囲に近い） | 0.5811 |

- halfkp が CPU で遅くなった理由は、まだ確かめていない。仮説は、forward ごとに「約 12.5 万行 × 256」の W_eff を作る（読み込みの行数は減るが、表の作成と逆伝播の分が増える）ため。CPU はメモリの速さに律速されやすいので、GPU では結果が違う見込み。**これは仮説。** 目的は GPU の速さなので、CPU の数字はここまでの参考。
- fused AdamW は CPU では使わない（従来どおり）。

## 持ち主が Colab で速さを測るためのセル

`tools/shk_speedcheck.py` は、学習と同じ 1 ステップ（base 入力・先読み・CUDA のとき fused AdamW）で、A 読み込み・B 転送・C 計算（fused あり／なし）・D 全部込みを測る。`<REPO>` と `<学習データ>` を書き換えて、Colab のセルで実行する。

```bash
!python <REPO>/tools/shk_speedcheck.py --repo <REPO> --glob "<学習データ>/*.psv" --network halfkp_256x2-32-32_f --batch 65536 --threads 11
```

比べるときは、同じ構造で `D` の行を見る（M7 までの「全部込み」と同じ意味）。`raypair0_128x2-16-16` は `--network raypair0_128x2-16-16` にする。

## 3 の fused AdamW

- `build_optimizer(..., fused=True)` は、`device.type == "cuda"` のときだけ使う。CPU は従来どおり（`fused` を渡さない）。
- checkpoint の `optimizer` の書き方は変えていない。fused の状態が従来と同じ形で保存されるかは、CUDA で確かめていない（Colab の確認項目）。

## 作ったファイル・変えたファイル

| ファイル | 役割 |
|---|---|
| `python/shikibu/networks.py` | `_table`・`_bag`（参考実装の `_eff` などを使う）。W_eff を forward ごとに 1 回だけ作る。`INPUT_VERSION`、`forward(..., base=True)` |
| `python/shikibu/train.py` | `INPUT_VERSION` でローダーを開く、`base=True`、`build_optimizer(fused=)`、`Prefetcher`（転送と先読み）、位置の持ち歩き（`_fetch`・`loader_state`） |
| `tools/shk_speedcheck.py` | 学習と同じ 1 ステップで D を測る形に直した。C3（表で計算）は削除（特徴の数が一定という前提があり、M8 の base 入力では成り立たない） |
| `docs/DEVELOPMENT.md` | M8 の行を追加 |
| `REPORT_M8.md` | この報告 |
| `tests/`・`reference/`・`PROTECTED.sha256`・`check_protected.py` | キットの導入で入った（変更禁止。手は入れていない） |

コミット（順に）：`Install kit M8`、`Factorize in the weights for base input (M8)`、`Overlap the next batch transfer with the step and use fused AdamW on CUDA (M8)`、`Measure the training step in shk_speedcheck (M8)`、`Add the M8 row to the development log`、（この報告）。

## 仕様の解釈で選んだこと

- **W_eff を forward ごとに 1 回**：参考実装は `_bag` の中で視点ごとに W_eff を作る。同じ表を 2 回（psqt も含むと 3〜4 回）作るので、`_table` で 1 回に直した。出力も勾配も同じ（`test_base_input_matches_factorized` が通る）。
- **先読みの位置**：先読みしたバッチの位置を checkpoint に書くと、再開で 1 バッチ飛ぶ。そのため、学習したバッチの位置だけを書く。
- **CPU の経路**：先読みは CPU でも行う（計算の順番の検査が CPU のテストで効くようにするため）。転送は CPU では従来の同期の `to_device` のまま。
- **shk_speedcheck の C3 の削除**：仕様は「D を学習と同じ処理で測る」なので、C3 は外した。

## 実施できなかったこと・未確認のこと

- **CUDA でのテスト**：この環境には GPU が無い（`torch.cuda.is_available()` は False）。`Prefetcher` の CUDA の経路（専用ストリーム・固定メモリ・event・`record_stream`）と、fused AdamW は、**実行していない**。テストは CPU だけ。Colab で `shk_speedcheck.py` を実行して、最初に確かめてほしい。
- **GPU の速さ**：計っていない（Colab が必要）。CPU の halfkp は遅くなった（上）ので、GPU で本当に速くなるかは、持ち主の計測で確かめること。
- **PyTorch の導入**：`download.pytorch.org` は proxy の policy で拒否されたため、PyPI の torch（CUDA 版の wheel）を使った。この環境では CUDA は使えない（上）。
- **本番データでの比較**：未実施。数学は変えていないが、fused AdamW の数値は CPU 版と僅かに違うことがある（CUDA でだけ使う）。

## 次の担当への引き継ぎ

1. **Colab で CUDA の経路を動かす**：`shk_speedcheck.py` の D と、fused の checkpoint の保存・`--resume`（止めて再開して、止めずに回した結果と一致するか）。
2. **CPU の halfkp の遅さ**：W_eff の作成を減らす（例：視点の 2 回を 1 回にまとめる、または表を使わずに因数分解の重みを別の表に分ける）。GPU で同じ問題が出るかを先に見ること。
3. **M7 からの積み残し**（`REPORT_M7.md` の引き継ぎ）は、この作業では触っていない。
