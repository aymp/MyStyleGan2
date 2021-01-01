# StyleGAN2で笑顔度編集
[ハッカソンで作ったやつ](https://github.com/masato-ikegawa/torys)([スライド](https://docs.google.com/presentation/d/1BPJ_adOObROC7nR7FzXzrFfDPqFiVNdBcm3I3A0IVfQ/edit?usp=sharing))の機械学習部分だけもってきた自分用のメモみたいなもの
Dockerfile周り自分の研究室PCの環境に合わせたものになってる

こちらの2つの記事に書いてあることを組み合わせただけです↓

[新垣結衣はGANの潜在空間に住んでいるのか？ | cedro-blog](http://cedro3.com/ai/search-for-yui/)

[StyleGAN2を使って顔画像の編集をやってみる | cedro-blog](http://cedro3.com/ai/edit-new-image/)

## 使い方
1. sample/picに編集したい顔入れる

2．my_edit_new_edit.py実行

3．my/ディレクトリに各生成物

笑顔度近づけるターゲットの画像の潜在変数の探索はcreate_hashikan_dataset.py
