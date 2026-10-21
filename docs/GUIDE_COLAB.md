# Colab で学習を回す手順

初めて Colab で Shikibu の学習を回すときの、順番どおりの手順です。
切れたときの対処や注意は `docs/OPERATIONS.md`、設定の意味は `docs/CONFIG.md` にあります。

ノートブックは `notebooks/colab_train.ipynb` です。以下の「節 1」「節 3」などは、そのノートブックの見出しの番号を指します。

> この手順のうち、Colab の上で動く部分は、この環境では実行していません（`docs/OPERATIONS.md` の 10 を参照）。ノートブックのコマンドは、同じ内容を手元の Linux で実行して確かめています。

---

## 0. 用意するもの（Google Drive）

Drive に次の4つを置きます。

| 場所（`MyDrive` からの相対パス） | 中身 | 無いとき |
|---|---|---|
| `DainagonProject/匠生用/データセット/psv/PonkotsuWCSC35_unique_*.psv` | 学習用の PSV（4 ファイル） | **学習できない**（設定の `train_glob` と一致する場所に置く） |
| `Shikibu/` | リポジトリ | 節 2 が GitHub から取得する |
| `shikibu_wheels/` | `shk_feat-*.whl`（Rust の拡張） | 節 3 が作って保存する（初回だけ時間がかかる） |
| `shikibu/runs/<設定名>/` | 学習の出力 | 学習を始めると自動で作られる |

- 学習データのパスは、`configs/colab_v0.toml`（と `colab_v1.toml`）の `train_glob` に書かれています。このパスを変えるなら、設定のほうを直します。
- PSV の中身（シャッフル済み、勝敗あり）は、`docs/OPERATIONS.md` の 5 を参照してください。

---

## 1. 最初の1回の準備

1. Colab で `notebooks/colab_train.ipynb` を開く。
2. **ランタイム → ランタイムのタイプを変更** で、ハードウェアアクセラレータを **GPU（L4）** にする。
3. 上の表の4つが Drive にあることを確かめる。

ここから先は、ノートブックの節を上から順に実行します。

---

## 2. 節の順番と、やること

### 節 1：Google Drive をマウントする

- 初回は認証画面が出るので、表示に従って許可します。
- 再接続のたびに実行します（`docs/OPERATIONS.md` の 2.2）。

### 節 2：リポジトリを取得（または更新）する

- `REPO`（Drive の `MyDrive/Shikibu`）があれば、`git pull` で更新します。無ければ GitHub から `git clone` します。
- `BRANCH` で、取り込むブランチを決めます。今の既定は `claude/magical-wright-5hukjw`（M4 の成果）です。
  - 別のブランチを使うときは、`BRANCH` を書き換えます。`REPO` の中で未コミットの変更があると、`checkout` や `pull` が失敗することがあります。

### 節 3：wheel（`shikibu_feat`）をインストールする

- Drive に `shk_feat-*.whl` があれば、それを入れます（数秒）。
- 無ければ、Rust を入れて `maturin build --release` で wheel を作り、`shikibu_wheels/` に保存してから入れます（初回だけ）。
- 節の最後に、`torch` の版、CUDA が使えるか、GPU の名前、論理コア数、`shikibu_feat` の次元数（v0 は 15000）を表示します。**ここで GPU が「なし」と出たら、ランタイムの種類を確かめてください。**

Rust のソースを変えたときは、`shikibu_wheels/` の wheel を消してから、節 3 を実行し直します。

wheel の名前には Python の版が入ります（例：`cp313`）。Colab の Python の版が変わると、古い wheel は入りません。確かめた：`cp312` の名前の wheel を `pip` に渡すと、「対応していない wheel」として拒否されます。そのときは、wheel を消して作り直します。

※ 節 3 は保存された wheel のうち一番新しいものを選ぶだけで、Python の版は見ません（`docs/DEVELOPMENT.md` の 5.4）。

### 節 4：設定を選ぶ（このセッションの量を決める）

```python
CONFIG_NAME = "colab_v0"   # "colab_v0"（RayPair_v0）か "colab_v1"（RayPair_v1）
SESSION_MAX_STEPS = None   # None なら設定の max_steps（179000）。数値なら通算の更新回数の上限
```

- `SESSION_MAX_STEPS` は、**通算**の更新回数の上限です。1回のセッションで区切るときは、その時点の通算の数を入れます。例：最初のセッションは `60000`、次は `120000`。
- `total_steps`（学習率の減衰の長さ）は変えないので、区切って続けても学習率は同じ曲線をたどります。
- この節は、`/content/run_config.toml` に書き出した設定を表示します（リポジトリの設定は変わりません）。学習用の PSV の数と、出力先、checkpoint の有無が表示されます。

### 節 5：学習を裏で開始する（`mon.start`）

```python
import sys
sys.path.insert(0, os.path.join(REPO, "tools"))
import shk_monitor as mon

mon.start(CONFIG, repo=REPO)
```

- 学習は裏のプロセスで始まり、このセルはすぐ終わります。
- `checkpoint.pt` があれば、自動で `--resume` を付けて続きから回します。無ければ最初から回します。
- 止めるときは `mon.stop()` を実行します。最後の保存より後の分は失われます。

### 節 6：進み具合を表示する（`mon.watch`）

```python
mon.watch(CONFIG)
```

- 15 秒ごとに、段階（データの確認・準備・学習中）、進捗、残り時間、速さ、損失のグラフを表示します。
- 表示の■（停止）は、表示だけを止めます。学習は続きます。もう一度このセルを実行すると表示が戻ります。
- **学習の間は、このセルを動かし続けてください。** 別のセルを実行したら、すぐにこのセルを実行し直します（`docs/OPERATIONS.md` の 1）。

表示される段階の意味は次のとおりです。

| 表示 | 意味 |
|---|---|
| 準備中：データの確認（勝敗の記録・局面の抜き出し） | 新規の学習の最初。全学習ファイルから局面を抜き出して、勝敗を確かめる |
| 準備中：eval_scale を推定中 | `eval_scale` が `"auto"` のとき |
| 準備中：Drive から先読み中 | `cache_dir` へのコピー（ファイルの大きさを表示する） |
| 準備中：シャッフルバッファを充填中 | `shuffle_buffer` を満たしている |
| 学習中 | 通常の学習 |
| 完了 | `max_steps` に達して、保存して終わった |
| 停止（手動） | `mon.stop()` で止めた |
| 異常終了（終了コード …） | 学習のプロセスが失敗した。最後の出力が表示される |

### 節 7：`.shk`（重みファイル）を書き出す

- 学習の途中の `checkpoint.pt` からでも書き出せます（保存は一時ファイルから改名するので、書き出し中に壊れることはありません）。
- 出力先は `MyDrive/shikibu/runs/<設定名>/<設定名>.shk` です。
- 書き出したあと、同じファイルを読み直して、ヘッダ・CRC32・全テンソルが一致するかを確かめています。一致しなければ、エラーで止まります。
- 終了コードが 0 なら、書き出しは成功です。

### 節 8：量子化の確認（`quantsim`）

- 整数の計算（エンジンが使う量子化、§E4）と、float32 の重みとの差を表示します。
- 表示される「z の差」「評価値の差（cp）」「損失」を控えます。
- **注意**：節 8 は、学習に使ったファイルの先頭で行います。本当の確認には、学習に使わなかった局面（検証用）を `--psv` で指定します（検証用の PSV は今は無い）。

---

## 3. 学習が終わったら

1. `MyDrive/shikibu/runs/<設定名>/` に、次のファイルがあることを確かめる。
   - `<設定名>.shk`：エンジンに渡す重み
   - `checkpoint.pt`・`log.jsonl`・`status.json`
2. `status.json` の `phase` が `done` であることを確かめる（`done` なら `max_steps` に達して、保存して終わっている）。
3. エンジン（Nuenagon）に渡すときは、`.shk` を `eval/nuenagon.shk`（既定の場所。§E2.2）に置くか、USI オプション `EvalFile` で指定する。
4. エンジン用の正解データ（N3 で使う）は、次の節の手順で作ります。

---

## 4. エンジン用の正解データを作る

Colab でも手元の PC でも作れます。`shikibu_feat`（wheel）と `cshogi` が要ります。

```
python tools/make_engine_golden.py --shk <設定名>.shk --out golden.json --n 1000 --seed 1
```

- `--n` は局面数、`--seed` は局面を作る乱数の種です。同じ種でも、`cshogi` の版が違うと局面が違う可能性があります（確かめていません）。
- 出力の JSON には、`.shk` の SHA-256（`shk_sha256`）が入ります。エンジン側で、同じ `.shk` を使っているかを確かめられます（形式は `docs/FORMATS.md` の 6）。
- `cp` はクリップしていません。エンジン側で、詰みの値と重ならないように収めてください。

手元の PC で動かすには、`README.md` の「手元の PC」の手順で仮想環境を作ってください。

---

## 5. 途中で切れたとき

1. `docs/OPERATIONS.md` の 2（切れたあとの再開）に従う。
2. 要点：
   - 節 1（Drive）→ 節 3（wheel）→ 節 2（リポジトリ）→ **節 4（設定。これを飛ばさない）** → 節 5（`mon.start`。自動で再開）→ 節 6（`mon.watch`）の順に実行する。
   - 最後の保存より後の分は失われる（既定では最大約 30 分）。

---

## 6. 設定を変えたいとき

- 節 4 の `SESSION_MAX_STEPS` で、このセッションの区切りを変えられます（学習率は変わりません）。
- そのほかの設定（バッチの大きさ、学習率、スレッド数など）は、Drive のリポジトリの `configs/colab_v0.toml` を直します。節 4 を実行すると、その内容で `/content/run_config.toml` が作り直され、節 5 で反映されます（`/content/run_config.toml` を直接書き換えても、節 4 を実行すると消えます）。
  - **途中で変えてよいもの**：`max_steps`（通算の上限）、`save_every_minutes`、`progress_every_sec`、`num_threads`、`torch_threads`。
  - **途中で変えてはいけないもの**：`network`（`checkpoint.pt` と合わなくなる）、`lam`、`train_glob` のファイル数、`eval_scale`（無視される。変えたいときは別の出力フォルダで始める）。
  - 詳しくは `docs/CONFIG.md` の「再開（`--resume`）のときの規則」を参照してください。
- 別の設定で最初から学習するときは、`out_dir` を別のフォルダにします。同じフォルダに `checkpoint.pt` が残っていると、`--resume` なしでは止まります。

