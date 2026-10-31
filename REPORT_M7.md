# M7 報告：評価関数の構造の実験室

## 結果

**完了**（中断はなし）。

- M6 は、この作業ブランチの時点で実装済みだった（`REPORT_M6.md`、`tests/test_m6_acceptance.py` が通る）。**M6 の分は、この作業では実装していない。**
- 手順 1（キットの導入）は、SHA-256 の照合、`KIT.json` の各ファイルの照合を経て完了。`check_protected.py` は `OK`。
- 取り込み直後の確認：キット導入の直後は、PyTorch が入っておらずテストを回せなかった（PyTorch は後から入れた）。最終の全件実行（下）で、M1〜M6 の 100 件も通ることを確かめた。

## テストの結果

```
$ python -m pytest -q tests
142 passed in 413.11s (0:06:53)

$ python check_protected.py
OK
```

内訳は M1〜M5.1 の 88 件、M6 の 12 件、M7 の 42 件。

## 参考実装（`reference/m7_proto.diff`）の使い方

差分は `git apply` で当てられなかった（M6 の実装と、`networks.py`・`export.py`・`lib.rs`・`loader.rs` などが衝突）。次のように取り込んだ。

- **そのまま使った**：`train.py` の差分（`git apply --include` で当たった）。
- **ほぼそのまま使った**：`networks.py`（名前の解析、`ArchNetwork`、`zone_of`、k3k3 の組の選び方。コメントは元の § 番号を残すように書き直した）、`lab.py`（関数の形と戻り値はそのまま。`export` の名前と、`nn.bin` の大きさの計算は今の `nnbin.py` に合わせた）、`configs/colab_lab.toml`（ひな形の形をそのまま使い、`val_path` などを揃えた）。
- **仕様どおりに書き直した**：
  - `halfk.rs` は、M6 の `halfkp.rs` を置き換える形にした（参考実装は `halfk.rs` を新設して `halfkp.rs` を残していた）。版 2〜11 を 1 つの表（`kind`）で扱う。
  - `kingzone.rs` を独立させた（参考実装は `halfk.rs` に入れていた）。
  - `feature.rs` は、版 0〜13 の振り分けと `king_squares` にした。
  - `nnbin.py` は、M6 の HalfKP 専用の定数をやめ、特徴・幅・層を引数に取る形にした（参考実装は `nnbin.py` の代わりに `export.py` に書いていた）。
  - `export.py` は、`nnbin` を使う形に合わせた。
  - ノートブックは、設定の説明と実行手順を書き直した（参考実装のものは使っていない）。
- **参考実装から外したもの**：`raypair.rs` の `features`・`features_into`（未使用になったので削除。RayPair の計算は変えていない）。M6 の `Pieces` の扱いは、ローダーでは常に作る形にした。

## 特徴量の抽出の速さ（`bench_extract`）

- 環境：Linux、論理 4 コア（計測は 1 スレッド）、release ビルド。入力は 2 万局面の PSV を循環して 20 万局面。
- 版 0（RayPair_v0）：約 11 万局面/秒（4 回の繰り返しで 110〜116k。変更前の同じ計測も 110〜116k で、差は無い）
- 版 2（HalfKP）：約 22 万局面/秒（1 回）
- 版 10（HalfKA_hm2）：約 24 万局面/秒（1 回）
- 版 12（RayPair_k）：約 7.4 万局面/秒（1 回）。区間 C（玉の周り）の分だけ遅い
- 版 13 は計測していない（版 12 と同じ経路）

## 学習の速さ（CPU・参考）

- 条件：CPU（4 スレッド）、バッチ 4096、40 ステップ。ローダーを含む学習の全体。ログの経過時間の差から、10 ステップあたりを換算した。
- `halfkp_256x2-32-32_f`：約 0.44 秒/ステップ（約 9,300 局面/秒）
- `halfkahm2_512x2-16-32_f`：約 0.64 秒/ステップ（約 6,400 局面/秒）
- `raypair0_128x2-16-16`：約 0.26 秒/ステップ（約 16,000 局面/秒）
- GPU の速さは計っていない。

## 書き出した nn.bin と float のモデルのずれ（`test_exported_eval_matches_float`）

テストの閾値は、平均 20 以下・最大 150 以下（cp）。作業用の表示で測った値：

| 構造 | 平均（cp） | 最大（cp） |
|---|---|---|
| `halfkahm2_64x2-16-32_f` | 12.4 | 30.4 |
| `halfkp_64x2-8-32` | 6.4 | 29.3 |

## 作ったファイル・変えたファイル

| ファイル | 役割 |
|---|---|
| `crates/shk_feat/src/halfk.rs` | 新規。HalfKP・HalfKA1/2・HalfKA_hm1/2（版 2〜11、因数分解つき含む）。`halfkp.rs` を置き換えた |
| `crates/shk_feat/src/kingzone.rs` | 新規。玉の周りのブロック（区間 C、版 12・13） |
| `crates/shk_feat/src/feature.rs` | 版 0〜13 の振り分け、`side_features`、`king_squares` |
| `crates/shk_feat/src/halfkp.rs` | 削除（`halfk.rs` に統合） |
| `crates/shk_feat/src/lib.rs` | 版 0〜13 の受け付け、`king_squares_sfen` の追加 |
| `crates/shk_feat/src/loader.rs` | バッチに `"kings"`（int64、(n, 2)）を追加。局面の駒の一覧は両視点で使い回す |
| `crates/shk_feat/src/raypair.rs` | 未使用になった `features`・`features_into` を削除（計算は変えていない） |
| `python/shikibu/networks.py` | 構造の名前（`parse_arch`・`arch_of`・`build`）、`ArchNetwork`（psqt・k3k3）、`clip_`・`decay_parameters` |
| `python/shikibu/train.py` | 構造の名前の受け付け、`ft_width` 0 の既定値、検証データの除外、`kings` の受け渡し、クリップと weight decay の一般化 |
| `python/shikibu/nnbin.py` | nn.bin の並び・ハッシュを、特徴と幅・層で作る汎用版に書き直した（`write`・`read`・`file_size`） |
| `python/shikibu/export.py` | `export_nnbin`（halfk… の任意の幅、FV_SCALE・EDITION）、`export_shk`（raypair0・1 の任意の幅）、書けないものは `ValueError`、CLI の出力 |
| `python/shikibu/lab.py` | 新規。`PRESETS`・`describe`・`make_config`・`write_config`・`compare_runs`・`export_run` |
| `notebooks/shikibu_lab.ipynb` | 新規。構造を選ぶ → 裏で順番に学習 → 比べる → 書き出す（日本語の説明つき。出力は消してある） |
| `configs/colab_lab.toml` | 新規。実験室のひな形（`<…>` の目印つき） |
| `README.md` | 構造の名前の読み方、実験室の使い方、比べ方の注意、やねうら王・Nuenagon での使い方 |
| `docs/CONFIG.md` | `model.network`（構造の名前）、`ft_width` 0、ひな形の一覧 |
| `docs/FORMATS.md` | nn.bin を一般化した形で書き直し、`.shk` の注記 |
| `docs/GUIDE_COLAB.md` | 実験室の節を追加 |
| `docs/DEVELOPMENT.md` | M7 の行、積み残し（5.6） |
| `docs/ARCHITECTURE.md` | 新しいモジュールの表の行 |
| `reference/m7_proto.diff`・`tests/`・`docs/DESIGN.md` など | キットの導入で入った（変更禁止。手は入れていない） |

## 利用者向け：比べて、書き出して、使うまで（手順の要点）

1. Colab でノートブック `notebooks/shikibu_lab.ipynb` を開き、設定セルの `<…>` を書き換える。`STRUCTURES` に比べる構造の名前を並べる（例：`halfkp_256x2-32-32_f`、`halfkahm2_512x2-16-32_f`）。
2. 節 6〜7 で、構造以外を同じにした設定を作って、構造ごとに裏で学習する。節 8 で進み具合を見る。
3. 節 9 で、val_loss を並べる（強さは対局で確かめる）。節 10 で、良かった構造の `ARCH` を書いて書き出す。
4. 出力の `nn.bin` を、やねうら王の評価関数のフォルダに置く。表示された `FV_SCALE` を、エンジンの `FV_SCALE` に設定する。`EDITION` の名前で、やねうら王を `YANEURAOU_EDITION` 付きでビルドする。

## 実施できなかったこと・未確認のこと

- **エンジンでの確認**：やねうら王・Nuenagon で、新しい幅・層の `nn.bin` を実際に読ませていない。テストは、やねうら王の静的評価値と参照の計算が 1 点も違わないことを、幅 256・層 32-32 などで確かめているだけ。
- **Colab での実行**：ノートブックは、構文・出力が空であること・テスト（`test_notebook_and_template`）の確認のみ。Colab では動かしていない。
- **GPU での k3k3 のメモリ**：バッチ 65536 では CPU でも試していない。計算は、全部の組の出力（局面数 × 9 × 出力の幅）を作ってから選ぶので、幅 L1 が大きいと中間が大きくなる（L1 = 16 なら約 38MB）。
- **書き出しの範囲**：psqt・k3k3 の構造と、`raypair0k`・`raypair1k` は書き出せない（仕様どおり `ValueError`）。
- **本番データでの比較**：未実施。

## 仕様の解釈で選んだこと

- 玉のない局面：バッチの `"kings"` を全版で作るため、ローダーは玉のない局面で `ValueError` を出すようになった（RayPair の版 0・1 にも及ぶ）。正しい局面（玉が両方ある）では変わらない。
- `model.ft_width` の既定値を 0 にした。従来の名前（ft_width を書かない設定）は 128 として扱う。checkpoint の `config` には、設定どおり 0 が残る。
- k3k3 の計算：全部の組の出力を作ってから `gather` で選ぶ（重みは局面ごとに複製しない）。
- `describe` の `max_active`：halfkp は 38、halfka… は 40、raypair は 208、raypair…k は 256。
- `export_run` の理由（書けないとき）：特徴ごとに、書けない理由を日本語で返す。

## 次の担当への引き継ぎ

1. **エンジンでの確認を最優先に**：幅 512 の HalfKA_hm2（`halfkahm2_512x2-16-32_f`）の nn.bin を、やねうら王でビルドして読ませる。`EDITION` と `FV_SCALE` の使い方をそのまま試す。評価値が学習の値 × eval_scale にほぼ一致するかを見る。
2. **Colab で実験室を最後まで回す**：GPU でのメモリ（特に k3k3 とバッチ 65536）と、比較の結果を確かめる。ノートブックは未実行。
3. **psqt・k3k3・`raypair…k` の書き出し**：やねうら王の形式に無いので、設計書の改訂で書き出しの形を決める必要がある。
4. **比較の本番**：同じデータ・同じ学習量で構造を比べ、対局で強さを確かめる。val_loss は強さの代わりにはならない。
5. **ローダーの玉の検査**：玉のない局面を、RayPair でも拒否するようになった（上の「仕様の解釈」）。壊れたデータで学習が止まるので、気になるなら検査の仕方を見直す。
6. **速さ**：版 12・13 の抽出は約 7 万局面/秒で、基準（20 万）に届かない（`docs/DEVELOPMENT.md` の 5.1 の候補を参照）。
7. **M6 の残り**（前の報告の引き継ぎ）：HalfKP の本番学習（GPU）と、やねうら王での確認は、まだ。

---

作業の記録：コミットは 5 つ（`Install kit M7`、`Add HalfK versions 4-11 and king-zone features 12-13 to shk_feat`、`Accept structure names for networks and training`、`Export nn.bin for any HalfK structure and .shk for any RayPair width`、`Add the structure lab, its notebook and config template`、`Document structure names, the structure lab, and engine use`）。`origin/claude/shikibu-m7-eval-structure-s90ebi` に push 済み（この報告書のコミットを含む。push の結果は最後の確認で示す）。
