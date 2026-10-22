# M6 報告：HalfKP の学習と、やねうら王の nn.bin の書き出し（比較実験用）

## 結果

**完了。** キットの取り込み（手順1）から、特徴量・ネットワーク・学習・書き出し・文書まで、すべて実装してテストが通りました。中断条件には当たりませんでした。

ただし、**やねうら王・Nuenagon で実際に nn.bin を読ませる確認は、この環境ではしていません**（エンジンがないため）。nn.bin の正しさは、キットの参照実装（`tests/halfkp_ref.py`・`tests/halfkp_nnbin.py`）との照合で確かめています。

## テストと検査

```
maturin develop --release -m crates/shk_feat/Cargo.toml   # 成功（警告なし）
python -m pytest -q tests
100 passed in 232.88s (0:03:52)                           # M1〜M4 73 件 + M5.1 15 件 + M6 12 件
python check_protected.py
OK
```

件数の内訳（`--collect-only` で確認）：M1 34、M2 15、M3 13、M4 11、M5.1 15、M6 12。

## 特徴量の速さ

`bench_extract` を、同じ PSV（ランダムな局面 5,000 件を循環させて 100 万局面）で測った値。3 回のうち最速。両視点の特徴抽出まで含む（解読は含む、読み込みは除く）。

| 版 | 1 スレッド | 4 スレッド |
|---|---|---|
| 0（RayPair_v0） | 0.15 M 局面/秒（6.59 s） | 0.55 M 局面/秒 |
| 2（HalfKP） | 0.31 M 局面/秒（3.25 s） | 1.21 M 局面/秒 |
| 3（HalfKP 因数分解つき） | 0.24 M 局面/秒（4.13 s） | 0.94 M 局面/秒 |

- RayPair_v0 は、変更前のコミット（`d0cf8cb`）と並べて測りました（1 スレッド 6.51 s → 6.59 s、4 スレッド 1.80 s → 1.81 s）。RayPair の経路の速さは変わっていません。
- 測った環境：論理コア 4（コンテナ内の Linux）、Python 3.13.16、PyTorch 2.14.1（CUDA 版の wheel を CPU で実行）、Rust 1.97.0 の release ビルド。

## 書き出した nn.bin と float モデルの評価値のずれ

`test_exported_eval_matches_float` と同じ手順（同じ設定で 60 step 学習し、乱数の局面 200 件で比べる）を、値を表示する別スクリプトで再現しました。テスト自体は閾値（平均 20 cp 以下、最大 150 cp 以下）を確かめるだけで、値は表示しません。

| ネットワーク | FV_SCALE | 平均のずれ | 最大のずれ |
|---|---|---|---|
| HalfKP_256Network | 13（eval_scale 600） | 11.6 cp | 44.0 cp |
| HalfKPf_256Network | 13（eval_scale 600） | 9.6 cp | 29.6 cp |

nn.bin のサイズは、どちらも 64,217,077 バイト（HalfKP と HalfKPf は同じ形で、因数分解は書き出すときに畳み込むため）。

## 作ったファイル・変えたファイル

**作ったもの**

- `crates/shk_feat/src/halfkp.rs`：HalfKP の特徴抽出（版 2）と因数分解つき（版 3）。`tests/halfkp_ref.py` と同じ式で、盤の持ち方（`board[y][x]`）からマス番号 `(8−x)×9+y` に変換する。
- `crates/shk_feat/src/feature.rs`：版の振り分け。0・1 は RayPair、2・3 は HalfKP。
- `python/shikibu/nnbin.py`：nn.bin の並び・ハッシュ・構造の文字列の書き込みと、読み直し。
- `configs/colab_halfkp.toml`：HalfKP のひな形（既定は `HalfKPf_256Network`、`ft_width = 256`）。
- `REPORT_M6.md`：この報告。

**変えたもの**

- `crates/shk_feat/src/lib.rs`：モジュールの登録、版の許容範囲を 0〜3 に、`extract_sfen`・`extract_psv`・`num_features` の振り分け。
- `crates/shk_feat/src/loader.rs`：`features_of` を版で振り分け。RayPair の経路（`Pieces` を1局面で1回だけ作る）は変えていない。
- `python/shikibu/networks.py`：`HalfKP_256Network`・`HalfKPf_256Network` を登録。計算式は RayPair と同じ。
- `python/shikibu/export.py`：`export_nnbin`（畳み込み・FV_SCALE・量子化・書き出し・読み直し）、CLI の分岐（`--fv-scale`、`FV_SCALE=<n>` の表示）。`export_shk` は HalfKP を拒否するようにした（RayPair の書き出しは変えていない）。
- `notebooks/colab_train.ipynb`：§8 は HalfKP のとき nn.bin を書いて FV_SCALE を表示。§9（quantsim）は HalfKP では対象外と表示。TEMPLATE の説明に HalfKP のひな形を追加。
- `README.md`・`docs/CONFIG.md`・`docs/FORMATS.md`（§2b に nn.bin の形式）・`docs/GUIDE_COLAB.md`・`docs/DEVELOPMENT.md`・`docs/ARCHITECTURE.md`・`docs/OPERATIONS.md`：HalfKP の使い方、件数、ファイル一覧、エラーの対処を追加。
- 手順1で入れたもの（キットから）：`PROTECTED.sha256`、`docs/DESIGN.md`（v1.7）、`tests/test_m6_acceptance.py`、`tests/halfkp_ref.py`、`tests/halfkp_nnbin.py`。
- 削除：`shikibu_kit_m6.zip`、`shikibu_kit_m51.zip`。

## 利用者向け：HalfKP を学習して、やねうら王・Nuenagon で使うまで

1. `configs/colab_halfkp.toml` をコピーし、`<…>` の目印（学習データと出力先）を書き換える。既定は因数分解つきの `HalfKPf_256Network`。因数分解なしの `HalfKP_256Network` にしたいときは `network` を書き換える。
2. 学習する：`PYTHONPATH=python python -m shikibu.train --config <設定ファイル>`（Colab ならノートブックの設定セルに `TEMPLATE = "configs/colab_halfkp.toml"` を指定する）。
3. nn.bin を書き出す：`PYTHONPATH=python python -m shikibu.export --checkpoint <出力フォルダ>/checkpoint.pt --out nn.bin`。標準出力の `FV_SCALE=<整数>` を控える。
4. やねうら王・Nuenagon で、評価関数ファイルに `nn.bin` を指定し、USI オプション `FV_SCALE` に控えた値を設定する（省略すると既定の 16 になり、評価値の尺度が合わない）。

## 判断したこと（キットの仕様の解釈）

- **作業ブランチ**：指示は `main` から始めること。ただしローカルの `main` は古いコミット（M4 時点）を指していたので、`origin/main`（`aca796a`、M5.1 まで入っている）と同じ内容の、指定の作業ブランチ `claude/brave-hypatia-wl6ibu` で作業した。ローカルの `main` は触っていない。
- **`FV_SCALE` の上限**：指定された FV_SCALE が `s = eval_scale × FV_SCALE / 8128 > 1` になるときは、エラーにした。仕様には書かれていないが、出力が範囲を超えて黙ってクリップされるのを防ぐため。既定の値では起きない。
- **§9（quantsim）の HalfKP での扱い**：quantsim は `.shk` 専用のため、HalfKP では省略した。nn.bin の書き出しのときに、読み直して全配列を照合している。
- **PyTorch の入れ方**：`download.pytorch.org` は proxy で拒否された（403）ため、PyPI の torch（CUDA 依存を含む）を入れた。CPU で実行している。

## 手順の途中で起きたこと

- 最初の検証用コマンドに含めた `rm -rf` が、安全チェックで止められて実行されなかった（削除していない）。以後の展開はすべて新しい空のディレクトリに対して行い、`rm` は使わなかった。
- `pip install` の最初の試行は、PyTorch の大きな wheel のダウンロードで何度か止まった。仮想環境は `/home/user/.venv_shk`（リポジトリの外）。

## 次の担当への引き継ぎ

1. **実エンジンでの確認**：やねうら王と Nuenagon で、書き出した nn.bin が読み込めて、評価値が学習の値 × eval_scale にほぼ一致することを確かめる。この環境では未確認。
2. **比較実験の本番**：同じデータ・同じ損失で RayPair と HalfKP を学習して比べる。本番データでの学習は、Colab でまだ回していない。HalfKP の学習の速さも、GPU では未計測（CPU の学習はテストの 60 step だけ）。
3. **ノートブックの HalfKP の経路**：Colab で実行して確かめていない（構文の確認だけ）。
4. **`docs/COLAB.md` は変更禁止**のため、HalfKP の説明は入れていない。必要なら次の設計の改訂で扱う。
5. **エンジン側の評価値の扱い**：参照実装どおり、Shikibu の書き出しはクリップしない。やねうら王は ±29999 でクリップするので、探索に使うときの扱いは Nuenagon の決まりに従う。
6. **RayPair の `--resume`**：学習ループとチェックポイントの形式は変えていないので、既存の checkpoint からの再開は従来どおり。M3 のテストに再開の検査がある。
