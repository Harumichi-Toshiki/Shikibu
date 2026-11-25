# REPORT M11：DL の学習（NNUE の教師データで、方策と価値の 2 つの頭を持つ DL）

設計書 v1.9 §S14（DL1）の前半に当たる実装報告です。キット `shikibu_kit_m11.zip`（`KIT.json` の SHA-256 を照合して導入）の受け入れテストを満たすように作りました。

## 1. 結果の要約

- 受け入れテスト `tests/test_m11_acceptance.py`：**22 件すべて通過**（4000 局面）。大きい乱数局面（`SHIKIBU_M11_N=20000`）でも、詰めた局面・指し手の番号・展開・hcpe 変換の比較が通ることを確かめた。
- 追加の確認 `tests/test_m11_extra.py`：**12 件通過**（変更禁止ではない。M9.1 の `test_m91_triton.py` と同じ扱い）。
- 既存の M1〜M10 のテスト：**197 件通過**（M11 の追加の前に、Rust の変更込みで確認）。
- 全体：TOTAL_PLACEHOLDER
- `python check_protected.py`：`OK`。
- GPU では動かしていません（この環境に GPU が無い）。GPU の速さ・メモリ・bf16／fp16 の実行、`torch.compile` の実行は未確認です（§6）。

## 2. ブランチとコミット

ブランチ `claude/m11-dl`（`main` から作成。`main` の先頭は、キット zip を載せたコミット `c017d52`）。

1. `Install kit M11`：キットの 3 ファイル（`PROTECTED.sha256`・`reference/dainagon_ref.py`・`tests/test_m11_acceptance.py`）を導入し、zip を削除。
2. 以降のコミットは `crates/shk_feat`・`python/shikibu`・`configs`・`notebooks`・`tools`・`tests/test_m11_extra.py`・`docs`・`README.md`・`REPORT_M11.md`。

## 3. 変えたファイル

| 区分 | ファイル | 内容 |
|---|---|---|
| Rust | `crates/shk_feat/src/dl.rs`（新） | 詰めた局面（盤 81・持ち駒 14・王手）、指し手の番号（2187）、`dl_record`、`dl_of`（バッチ）。単体テスト 4 件 |
| Rust | `crates/shk_feat/src/loader.rs` | `Output`（`Features`／`Dl`）と `Batch` を追加。`BatchLoader(..., output="features")`。`features` の経路は変えていない |
| Rust | `crates/shk_feat/src/lib.rs` | `dl_record` の登録 |
| Python | `python/shikibu/dl_features.py`（新） | `expand`（GPU で一括、Dainagon とビット単位で同じ）、`hl_gauss`、`from_hcpe`（検証用） |
| Python | `python/shikibu/dl_networks.py`（新） | 構造の名前、登録表 `BLOCKS`（S・L・F・T・H）、`register_block`、`parse_dl`、`build_dl`、`describe_dl`、`PRESETS`、重み減衰の組 `param_groups` |
| Python | `python/shikibu/train_dl.py`（新） | 学習ループ（`python -m shikibu.train_dl`）。損失・検証・保存と再開・`max_minutes`・fp16 の GradScaler |
| Python | `tools/shk_speedcheck_dl.py`（新） | 読み込み・展開・学習の 1 ステップを、プリセットごとに測る |
| Python | `tools/shk_monitor.py` | `model.network` が `dl_` で始まれば `train_dl` を起動。DL の検証行の表示 |
| 設定 | `configs/colab_dl.toml`（新）、`notebooks/colab_dl.ipynb`（新） | 既定値は指示どおり。`<…>` の目印つき |
| テスト | `tests/test_m11_extra.py`（新） | §5 |
| 文書 | `README.md`・`docs/ARCHITECTURE.md`・`docs/CONFIG.md`・`docs/FORMATS.md`・`docs/GUIDE_COLAB.md`・`docs/DEVELOPMENT.md`・`docs/DL_BLOCKS.md`（新） | 指示の文書＋「新しいブロックの足し方」 |

## 4. CPU で測った速さ（この環境：4 コア、データは合成の PSV）

持ち主の GPU の値は未計測です。ここの数字は CPU の参考値で、学習の速さの判断には使わないでください。

| 段階 | 結果 | 条件 |
|---|---|---|
| 読み込み（`BatchLoader(output="dl")`） | 約 49 万局面/秒 | 作業係 1 スレッド、12 万局面、3 回の最良 |
| 同（RayPair v0、比較） | 約 20.5 万局面/秒 | 同条件 |
| 同（HalfKP v2、比較） | 約 46.7 万局面/秒 | 同条件 |
| 読み込み（DL、2 スレッド） | 約 92〜97 万局面/秒 | 短い計測（0.1 秒）なので参考 |
| 展開（`expand`、CPU） | 約 1.2〜1.4 万局面/秒 | バッチ 512 |
| 学習の 1 ステップ（CPU、fp32、バッチ 512） | A 253、B 264、X2 151 局面/秒 | 既定のスレッド、`compile`・channels_last なし |

**読み取り**：DL の作業係は、HalfKP v2 とほぼ同じ速さで、RayPair v0 よりは速い。指示の「HalfK 系より速いはず」は、この計測では成り立っていません（ほぼ同じ）。Colab の 11 スレッドなら読み込みは数百万局面/秒の見込みで、目標の 5 万局面/秒（GPU の学習）より十分速い。律速は GPU 側になる見込みです。

## 5. 設計の選択と、確かめたこと

1. **PSV の指し手の読み方**。やねうら王の 16 ビットは、下位 7 ビットが移動先、次の 7 ビットが移動元（打つときは駒の種類 1〜7）、bit 14 が打つ、bit 15 が成る。cshogi の `move16_from_psv` と、移動先が盤内で移動元が盤内（または打つ種類が 1〜7）の全符号について一致することを確かめた（`test_psv_move_codes_match_reference_exhaustively`、先手番・後手番の 2 局面）。cshogi は範囲外の符号で異常終了するため、範囲外は −1 とした。
2. **盤の座標**。既存の M1 のデコーダ（`Position`）をそのまま使い、手番側から見た座標に直す。チャンネル番号は、駒の種類の順（§F1.3）と Dainagon の番号が同じなので、そのまま使える。
3. **展開の大駒の利き**。方向ごとに「元の駒から手前がすべて空き」を、占有の累積積で求める。方向の集合は 180° 回転で閉じるので、座標の回転の前後で同じ結果になる（展開は座標を回さない）。
4. **損失**。指示の式のとおり。方策は `label_smoothing` 付きの CE、価値は HL-Gauss の CE、勝敗は BCE。価値の目標は `sigmoid(score / eval_scale)`。
5. **検証の value_mse**。PSV は `eval_scale`、hcpe は `score / 600`（指示どおり）。`policy_acc` は合法手で絞らない（Dainagon と同じ）。
6. **注意の計算**。`scaled_dot_product_attention` に相対位置バイアスを加法マスクとして渡す（F・T とも）。最初は 81×81 の行列を明示的に作っていたが、CPU でバッチ 4096 がメモリ不足で落ちたので変えた。数式は同じ（M11.0.1 で Dainagon の F と比べて確かめた。§9）。
7. **ログの形**。各行は `type` と `kind` の両方に同じ値（`shk_monitor` と今の学習器が `type` を読み、受け入れテストは `kind` を読むため）。
8. **train.py の部品の再利用**。`load_config`・`check_placeholders` の一部・`train_paths`・`epoch_perm`・`lr_factor`・`Status`・`JsonlLog`・`sample_scale_positions`・`eval_scale_setting`・`seed_everything`・`rng_state`・`restore_rng`・`fmt_sec`・`_on_term`・`TrainLock` を import。`train.py` は変更していない。ただし DL の設定の既定値・検査は `train_dl.py` に別に置いた（キーが違うため）。
9. **オプティマイザの口**。`train_dl.make_optimizer` の 1 か所。未知の名前は `validate` で止まる。
10. **既定の設定の動き**。`BatchLoader` の既定の `output` は `"features"` で、従来の経路は変えていない（既存の全テストが通ることで確認）。

## 6. 未確認・既知の問題（持ち主に確かめてほしいこと）

- **（M11.0.1 で解消）F・T は Dainagon と照合して直した（§9）。S・L と頭は設計担当が Dainagon と照合し、同じ計算と確認済み。** 学習そのものは新しく始めるので影響は限定的です。
- **メモリ（X2 が心配）。** CPU の fp32 でバッチ 256 の 1 ステップのピーク増分から見積もると、1 局面あたり A 約 6.5 MB、B 約 7.1 MB、X2 約 9.7 MB。bf16 の GPU ではこの 6 割ほどの見込み。**バッチ 4096 で X2 は L4（24 GB）に入らない可能性がある。** 入らなければ 2048 か 1024 にして、`total_steps`・`max_steps` を合わせて調整してください（`docs/GUIDE_COLAB.md` の 8.2）。実測で決めてください。
- **GPU の速さは未計測。** `tools/shk_speedcheck_dl.py` を Colab で実行して測ってください。
- **未確認の経路**：`amp = "fp16"`（GradScaler。CPU では拒否することだけ確かめた）、`compile = true`（CPU では未実行。既定は true なので、Colab で最初に確かめてください）、`channels_last` の GPU 側、CUDA の固定メモリ経路（`DLPrefetcher` の CUDA 分岐）。
- **hcpe の検証**は、合成の hcpe で `from_hcpe` と参照の一致を確かめただけです。Dainagon のテストセットでは動かしていません。
- **shk_monitor の表示**は、CPU の短い学習の出力（train・val・save・end）で表示まで確かめました。Colab のノートブックは実行していません。
- **Rust の速さ**は指示の目安（HalfK より速い）に届かず、ほぼ同じでした（§4）。作業係の `dl_of` は `compact` の中で王手の判定を全駒に対して行うので、ここを直せば速くなる見込み（未着手）。
- **ONNX の書き出し（M11.1）は未着手。** `shift9` は固定の depthwise conv（buffer）にしてあるので、その準備はできている。入出力の名前と形は、持ち主から受け取ってから決める。

## 7. 引き継ぎ

- **M11.1（ONNX）**：今の Dainagon の ONNX の入出力の名前と形を受け取る。`Shift9` は buffer なので書き出しやすい。注意の `scaled_dot_product_attention` は ONNX の対応を確かめる必要がある（未確認）。
- **Muon など**：`train_dl.make_optimizer` に分岐を足すだけ。重み減衰の組は `dl_networks.param_groups` が決めるので、Muon 用の組はそこで分ける。
- **持ち主の作業**：（1）L4 で `tools/shk_speedcheck_dl.py` を実行し、プリセットごとの局面/秒とメモリを測る。（2）バッチサイズを決める（X2 は特に）。（3）`total_steps`・`max_steps` を実測の局面/秒に合わせる。（4）S・L・F を Dainagon のコードと照合する。（5）検証用の PSV（学習に使わないもの）を用意し、`val_path` に指定する。

## 8. 実行したコマンド（再現用）

```bash
python -m pytest -q tests/test_m11_acceptance.py      # 22 件
python -m pytest -q tests/test_m11_extra.py           # 12 件
python -m pytest -q tests                             # 全体
python check_protected.py                             # OK
python tools/shk_speedcheck_dl.py --repo . --glob "<学習データ>/*.psv" --presets A,B,X2 --batch 4096 --threads 11
```

## 9. M11.0.1：F・T を Dainagon と同じ計算にする

設計担当が Dainagon の `network_chunagon_v35.py` と照合し、F（GatedTransformerBlockRelPos）と、同じ注意を使う T が違うと分かった。直した内容：

| 項目 | 前（M11） | 後（M11.0.1、Dainagon と同じ） |
|---|---|---|
| 位置の埋め込み | 無い | `file_emb (1,C,1,9)`・`rank_emb (1,C,9,1)`（乱数 ×0.02）を、トークンにする前に入力へ足す |
| qkv | `Linear(C, 3C)`（bias あり） | `Linear(C, 3C, bias=False)` |
| ゲート | 注意の出力に `sigmoid(Linear(h))`（要素ごと）。FFN にゲート無し | スカラーの `gate_attn`・`gate_mlp`（初期値 0.1）。注意の出力と FFN の出力に掛けてから残差に足す |
| 相対位置バイアス | 切断正規分布（0.02） | ゼロ初期化 |
| 名前 | `attn.qkv`・`attn.out`・`attn.gate` など | Dainagon と同じ（`file_emb`・`rank_emb`・`norm1`・`norm2`・`qkv`・`proj`・`ffn.0`・`ffn.2`・`gate_attn`・`gate_mlp`・`relpos_bias`） |

変えなかったもの：LayerNorm の位置（注意の前に `norm1`、FFN の前に `norm2`）、`proj`（bias あり）、FFN の形（F は `Linear(C,4C)`・GELU・`Linear(4C,C)`、bias あり）、`scale = hd^-0.5`、相対位置の番号（`(dy+8)×17+(dx+8)`）。

T は F と同じ注意・位置の埋め込み・スカラーのゲートで、FFN だけ GLU（`Linear(C, 2·ffn·C)` を 2 つに分け、`GELU(ゲート) × 値`、`Linear(ffn·C, C)`）。

macs：F は `81 × (4C² + 8C²) + 2·81²·C`、T は `81 × (4C² + 3·ffn·C²) + 2·81²·C`（ゲートの Linear が無くなった分だけ減った）。

### 確かめたこと（`tests/test_m11_extra.py`）

- `test_F_matches_dainagon`（C=64・heads=2、C=320・heads=10）：Dainagon の F のコードをテストの中にそのまま置き、同じ重みを入れて、乱数の入力で出力を比べた。最大の差は 1e-5 未満（通過）。参照側の `relpos_bias` は確かめるために乱数にし、ゲートは 0.7・−0.4 にした（Shikibu 側へ同じ値を写す）。
- `test_T_matches_glu_reference`（同じ 2 通り）：FFN だけ GLU にした参照と比べた（通過）。
- `test_F_T_init_values`：`relpos_bias` が全部 0、`gate_attn`・`gate_mlp` が 0.1、`qkv` に bias が無い、`file_emb`・`rank_emb` の形。
- `test_F_T_macs_formula`：macs の式。
- `test_no_decay_groups`：位置の埋め込み・ゲート・相対位置バイアス・LN・バイアスは減衰しない組に入る。

### テストの結果

- `tests/test_m11_acceptance.py`：22/22 通過（F・T を含む網の macs の見積もり、学習の数歩も含む）。
- `tests/test_m11_extra.py`：17/17 通過。
- 全体（`python -m pytest -q tests`）：235 件通過、1 件失敗。失敗は `tests/test_m51_acceptance.py::test_configs_are_templates`（M5.1 の変更禁止のテスト）。`configs/colab_dl.toml` が DL 専用のキーを持つため、RayPair の学習器が「ひな形」のエラーより前に未知のキーで止まる。テストも学習器も変えていない（持ち主の判断待ち）。

