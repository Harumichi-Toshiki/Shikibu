# M1 報告：特徴抽出（`shikibu_feat`）

## 結果

**完了。** `python -m pytest -q tests` は全件通過。

## テスト結果

```
$ python -m pytest -q tests
..................................                                       [100%]
34 passed in 26.17s
```

内訳：`num_features` 2件、`test_golden` 24件（§F7 の全局面・全視点・v0/v1）、`test_decoder_vs_cshogi` 1件、`test_random_vs_reference` 2件、`test_psv_path_matches_sfen_path` 1件、`test_perspective_symmetry` 2件、`test_range_and_bound` 2件。
落ちたテスト：なし。スキップ：なし。

## 保護ファイルの確認

```
$ python check_protected.py
OK
```

## 作ったファイル

| ファイル | 役割 |
|---|---|
| `crates/shk_feat/Cargo.toml` | クレート定義（lib 名 `shikibu_feat`、pyo3 0.24） |
| `crates/shk_feat/Cargo.lock` | 依存関係の固定 |
| `crates/shk_feat/src/lib.rs` | pyo3 の公開関数 4 つ（`num_features`、`extract_sfen`、`decode_psv`、`extract_psv`）と引数検査 |
| `crates/shk_feat/src/position.rs` | 局面の表現、SFEN の読み取り・書き出し、PackedSfen・PSV のデコーダ |
| `crates/shk_feat/src/raypair.rs` | RayPair 特徴量（§F2）。`reference/raypair_ref.py` と同じ規則 |
| `README.md` | ビルド・テスト手順と API 一覧 |
| `tests/`、`reference/`、`PROTECTED.sha256`、`check_protected.py`、`.gitignore` | `main` の `shikibu_starter.zip` から verbatim に取り出した（ハッシュ検査 OK） |

## ビルド手順

```
python -m venv .venv && . .venv/bin/activate
pip install cshogi pytest numpy maturin
maturin develop --release -m crates/shk_feat/Cargo.toml
python -m pytest -q tests
```

`maturin develop` は仮想環境が無いと失敗する。

## 実装の注意点

- **PackedSfen の形式**：ビット順（下位から）、手番・玉位置（各7bit）・盤上の駒のハフマン符号・持ち駒の符号を、cshogi のリポジトリ（PyPI の sdist 内 `src/position.cpp`）に書かれた符号表どおりに読んでいる。デコーダ本体は自分で書き直した。テストは cshogi の `to_psfen` と照合しているので、符号表の転記違いはテストで検出される。
- **マス番号**：PackedSfen のマス番号 `s` は `y = s % 9`、`x = 8 − s / 9`（`1一 = 0`）。
- **PSV の手数**：`decode_psv` の SFEN の手数はレコードの `gamePly` を使う（0 のときは 1）。
- **エラー処理**：`perspective`・`version` が範囲外、PSV の長さが違う、符号が不正、持ち駒が上限超えの場合は `ValueError`。
- **玉の検査**：玉が盤上に無い・2枚ある場合の検査はしていない（§F2.0 の4は任意）。

## 次の担当への引き継ぎ

- **速度は未対応（M2 の範囲）**：`extract_sfen` は SFEN を毎回パースして `Vec` を作る。符号の読み取りは線形探索。正しさを優先した実装。
- **ベンチマーク未実施**：`bench/` は未作成。
- **`main` と本ブランチの関係**：テスト一式は `main` の `shikibu_starter.zip`（コミット `793c8c9`）から取った。`main` 自体は未変更。
- **コミットの分け方**：指示では小さく分けるとあったが、crate は 1 コミットにまとめた（`96b0888`）。凍結ファイルは `15386be` の 1 コミット。
- **符号表の出典**：デコーダの符号表とビット順は cshogi の公開ソースを参照して確認した。cshogi は MIT ライセンスで、本リポジトリに取り込んだコードはない。

## 作業の記録

- 凍結ファイルのコミット：`15386be`
- Rust クレートとREADME のコミット：`96b0888`
- 本レポートのコミット：（このファイルを含むコミット）
