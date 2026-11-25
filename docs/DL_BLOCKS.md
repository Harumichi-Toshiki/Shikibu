# DL のブロックの足し方（M11）

DL（`python/shikibu/dl_networks.py`）のブロックは、1 文字の登録表 `BLOCKS` で持ちます。
新しいブロックを試すときは、クラスを書いて `register_block` に渡すだけで、構造の名前（例：`dl_192_SHHHHT_H_HH`）の中で使えます。
名前の読み取り・計算量の数え方（`describe_dl`）・テストは、ブロックの種類を知りません。

---

## 1. クラスの約束

```python
import torch.nn as nn

class MyBlock(nn.Module):
    def __init__(self, channels, act, opts):
        # channels：入出力のチャンネル数 C。act：活性化のクラス（ReLU か SiLU。使うときは act()）。
        # opts：構造の名前のオプション {"stem", "ffn", "hd", "vh"}（省略時の既定で埋まっている）。
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = act()

    def forward(self, x):
        # x：(B, C, 9, 9) → 同じ形（B, C, 9, 9）を返す。
        return self.act(self.bn(self.conv(x)) + x)

    @classmethod
    def macs(cls, channels, opts):
        # 1 局面の順伝播の掛け算の回数（conv・linear・注意の行列積を数える。BN・活性化・シフトは数えない）。
        return channels * channels * 81
```

- `macs` は**クラスメソッド**で、`channels` と `opts` だけから決まる値を返します（重みの値には依らない）。
- 学習するパラメータは普通の `nn.Parameter`・`nn.Module` で持ちます。`describe_dl` の `params` は、作った網の `requires_grad` のパラメータを数えるので、書く必要はありません。
- 重み減衰を掛けないものは、パラメータに `p._no_decay = True` と付けます（BN・LN の重み、バイアス、相対位置バイアスの例は `dl_networks.py` を参照）。`param_groups` がこの印を見ます。

## 2. 登録する

```python
from shikibu import dl_networks as N

N.register_block("Z", MyBlock)     # 1 文字の大文字（"0" は使えない）。すでに登録されていれば ValueError
N.parse_dl("dl_64_SZ_Z_0")         # 名前が読める
N.describe_dl("dl_64_SZ_Z_0")      # macs・params が出る
N.build_dl("dl_64_SZ_Z_0")         # 網ができる
```

- 登録は、そのプロセスの中だけに効きます。学習の設定（TOML）で使うには、学習器を動かすプロセスの中で登録するコードが要ります。
  恒久的に使うなら、`dl_networks.py` の `BLOCKS` に 1 行足して、そのコミットで使ってください（既存の名前の意味は変わりません）。
- 登録した文字は、`parse_dl`・`build_dl`・`describe_dl`・学習の設定の `model.network` でそのまま使えます。

## 3. 試す

テストが、登録と使い方の例を見せています（`tests/test_m11_acceptance.py` の `test_register_block`）。自分で試すには次のように書きます。

```python
import torch
from shikibu import dl_networks as N

N.register_block("Z", MyBlock)
net = N.build_dl("dl_64_SZ_Z_0")
net.train()
p, value_logits, result_logit = net(torch.randn(2, 55, 9, 9))   # p (2, 2187)・value_logits (2, 128)・result_logit (2, 1)
(p.sum() + value_logits.sum()).backward()                       # 逆伝播が通る
print(N.describe_dl("dl_64_SZ_Z_0"))                            # {"canonical", "params", "macs"}
```

比べるときは、`describe_dl` の `macs` で計算量をそろえてから、同じデータ・同じ学習量で学習の損失（`log.jsonl` の `loss`・`loss_p`・`loss_v`）を比べてください。

## 4. 注意

- `opts["hd"]` を使う注意ブロックは、幅 C が `hd` で割り切れないと `parse_dl` が `ValueError` にします（ブロックの名前に `F` か `T` があるとき）。
- 注意の 81 トークン（9×9）は、`F`・`T` の `_Attention` を使い回せます。相対位置バイアスは `(heads, 289)` の表と `rel_index` です。
- 出力の形（`(B, C, 9, 9)`）を変えるブロックは使えません（頭が前提にしています）。
- `shift9` は学習するパラメータを持たず、`Shift9` という `Module` としても使えます（`kernel` は buffer。ONNX に書き出す準備のため）。
- ネットワークの学習の速さは、ブロックの計算量（`macs`）だけでは決まりません。GPU の実測（`tools/shk_speedcheck_dl.py`）で確かめてください。
