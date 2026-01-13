仕様目標

# GoogleColabで動くか実験
研究室に行ける時間は限られるので、Googlecolabでcuda使って動作問題ないか確かめる。
1. Googlecolab開く
2. このリポジトリをクローン
3. Algonauts2023のデータのGoogleDriveからsubj01だけ解凍する
4. このリポジトリのコードを実行することで、Algonauts2023　subj01だけで学習(する動作確認、本番は研究室のLinux)
5. 性能や　生成画像の確認ができるようにする。

# 研究室での実行
1. python環境構築(元のReadmeに書いてたコマンドを使う)
2. このリポジトリをクローン
3. Googlecolabで試したコードを実行。


# コード改変計画
- ニューラルネットの入力を、ボクセルから、Algonauts2023のデータ対応にする。transformer brain encorderのモデルの出力の次元数とかを参照できそう。
- ニューラルネットの学習時のデータの扱いのやつをAlgonauts2023のやつに対応させる。できるだけわかりやすいほうがいい。