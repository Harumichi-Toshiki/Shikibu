# M9.1 報告：L4 の測定を受けて（速さの道具の直し・第1層の Triton カーネル）

## 要点

- M9 の第1層のオプション（`ft_forward = "sparse"`・`ft_grad = "sparse"`・`lazy_adam`・`fast`）は、L4 では既定より**遅かった**。使わない（`docs/SPEED_M9.md` に注意を書いた）。
- 速さの道具は、各行を独立にし（失敗した行は理由を出して次へ）、時間の内訳（P 行：`torch.profiler`）を足した。F（開始前の待ち）は最後に回し、始める前に表示する。
- 第1層の専用カーネル（`train.ft_kernel = "triton"`）を書いた。**GPU では動かしていない**（この環境に GPU が無い）。解釈モードで、`torch` の計算と一致することは確かめた。速さは持ち主が E5 の行で測る。
- 既定の設定（`ft_kernel = "torch"`）の学習の結果は変えていない。

## 1. 持ち主の測定の読み方

HalfKP 256（因数分解なし）では、E0（既定）157.5、E1 157.3 万局面/秒。sparse の順伝播は既定と同じ。E2（順伝播・勾配 sparse＋遅延型）は 52.4 と 3 倍遅い。E3（遅延型のみ）は 119.6。bf16（E4）は効かない。

HalfKA_hm 512（因数分解つき）では、E0 が 84.4、E1 が 44.5 万局面/秒。E1 の次の行（E2）で、逆伝播が 10 GiB を確保しようとしてメモリ不足になった。

見立て（設計担当）：時間の大半は `embedding_bag` の順伝播・逆伝播の中。M9 の sparse の順伝播は、因数分解つきで大きな中間データ（バッチ × 特徴 × 幅）を作る疑い。これは後で P 行で確かめる。

## 2. A：速さの道具の直し（`tools/shk_speedcheck.py`）

- 各行を `row()` で囲む。失敗したら `失敗（型: 理由）` を 1 行出し、`gc.collect()` と `torch.cuda.empty_cache()` をして、次の行へ進む。
- E 行に **E5（`ft_kernel = "triton"`）** を足した。
- **P 行**：既定（E0）の 1 ステップを `torch.profiler` で 5 ステップ測る。順伝播・勾配の初期化・逆伝播・オプティマイザ・クリップの各段階の時間（`record_function` で区切る）と、自分の時間の上位 15 件、0 埋め・コピー系（`zero_`・`fill_`・`copy_`）の合計を出す。CUDA では GPU の時間、CPU では CPU の時間。
- **F 行は最後**。始める前に「ここから開始前の時間を測る（数分〜数十分かかることがある）」と出す。F1〜F4 も各行を独立にした。
- `--skip-profile` で P を飛ばせる。

CPU で小さく回した結果（2 ステップだけの動作確認で、速さの意味は無い）：

- 失敗した行（E5）は `RuntimeError: 0 active drivers` と出て、次の行へ進んだ。
- P 行は、CPU では `aten::fill_`（0 埋め）が約 32%、`aten::div`・`aten::sqrt`（Adam の計算）が約 29%。GPU では内訳が違うはず。

## 3. B：第1層の専用カーネル（Triton）

設定：`train.ft_kernel = "torch"`（既定）・`"triton"`。`"triton"` のときは `ft_forward`・`ft_grad` が `"dense"` でなければならない（検査で止まる）。

中身（`python/shikibu/ft_triton.py`）：

- **順伝播**：局面（手番側・相手側の 2 本）ごとに 1 プログラム。有効な特徴の行を、バイアスから始めて足す。因数分解つきでは、本体の行 `r` と、共通の行 `n + r mod P` を足す。`W_eff` も、大きな中間データも作らない。
- **逆伝播**：出力の勾配を、有効な特徴の行に `tl.atomic_add`（fp32）で足し込む。因数分解の共通の行も同じ。勾配は今と同じ**密な形**（重みと同じ大きさ）で返すので、オプティマイザ（fused AdamW）は今のまま。バイアスの勾配は両側の和。
- Triton は必要になったときだけ import する。GPU が無いときは、解釈モード（`TRITON_INTERPRET=1`）でだけ動く（遅い）。Trainer は、使えないときに学習の前に止まる。
- 入力の形（特徴の並び・オフセット・空の局面）は、今の `BatchLoader` の出力に合わせた。空の局面は和が 0。

## 4. 正しさの確かめ（解釈モードで動かした）

`tests/test_m91_triton.py`（5 件、すべて通過。解釈モードで実際に Triton のカーネルが動いた）：

1. 因数分解つき（`halfkp_32x2-8-8_f`）：順伝播の出力と、重み・バイアスの勾配が、`torch` の計算と一致（許容誤差 1e-5）。
2. 因数分解なし（`halfkp_32x2-8-8`）：同じく一致。
3. 空の局面を含む、幅 64 の因数分解つき：一致。
4. 解釈モードが無いときは、学習の前に `TRITON_INTERPRET` の理由で止まる。
5. **学習の全体**（`train.main`、3 ステップ）：`torch` と `triton` で、保存された重み（checkpoint）がすべての名前で一致（許容誤差 1e-4）。

**確かめていないこと**：GPU での動き（コンパイル・メモリ・速さ）。原子的な加算の順序による、GPU での丸めの差（解釈モードでは一致した）。

## 5. 持ち主が Colab で回す手順

1. 速さの一覧（E5 と P を含む）：

```bash
!python /content/drive/MyDrive/<置き場所>/shk_speedcheck.py --repo <REPO> --glob "<学習データ>/*.psv" --network halfkp_256x2-32-32 --batch 65536 --threads 11 --shuffle-buffer 20000000 --cache-dir /content/shk_cache
```

（HalfKA_hm 512 なら `--network halfkahm2_512x2-16-32_f` など、設定と同じ構造の名前にする。）

2. 見る行：
   - `E0`（既定）と `E5`（Triton）の万局面/秒。E5 が E0 より速ければ、`ft_kernel = "triton"` を候補にする。
   - `P` の「上位 15 件」と「0 埋め・コピー系の合計」。順伝播（`1 順伝播`）・逆伝播（`3 逆伝播`）・オプティマイザ（`4 オプティマイザ`）のどこが重いかを見る。
   - 失敗した行があれば、その理由（メモリ不足など）。
3. 設定：`[train] ft_kernel = "triton"`（他の `ft_*` は既定のまま）。学習の損失の推移（`log.jsonl` の `loss`）が既定と同じ程度か確かめる。

## 6. C：片付け

- `docs/SPEED_M9.md`：L4 の測定値と、M9 の sparse の注意（既定より遅い。使わない）を書いた。`ft_kernel` の行を足した。
- M9 の `ft_forward = "sparse"`・`ft_grad = "sparse"`・`lazy_adam` は残した（既定は変えていない）。

## テスト

全件（`python -m pytest -q tests`）：**184 件通過**（M1〜M9 の 179 件と、M9.1 の 5 件）。`check_protected.py`：OK。

## 変更したファイル

- `python/shikibu/ft_triton.py`（新規）：Triton のカーネルと、autograd の関数
- `python/shikibu/networks.py`：`FT_KERNEL`（既定 `"torch"`）と、`triton` のときの第1層の経路
- `python/shikibu/train.py`：`train.ft_kernel` の設定と検査、Trainer での利用可否の確認
- `tools/shk_speedcheck.py`：行ごとの失敗の扱い、E5、P（内訳）、F を最後に
- `tests/test_m91_triton.py`（新規）：5 件（解釈モード）
- `docs/SPEED_M9.md`：L4 の測定値と注意、`ft_kernel`
- `REPORT_M91.md`（この文書）

## コミット

最終のコミットは、この報告を含むコミット（`git log -1` で確認）。
