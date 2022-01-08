# Chat_Bot

## 作品名
ツイートされた内容に対して，漫画「ピンポン」に登場するキャラクターが返答するプログラムを作成しました．

![car](img/box_car.jpg)

(実行中の画面)

## 概要
このプログラムは，つぶやきを得ると，作成した一問一答集からそのつぶやきに近い質問文を見つけ，対応する解答文を返す．
この時，つぶやきと質問文に形態素解析を行って文章をベクトル化し，つぶやきと質問文間の類似度をすべて計算し，
最も類似度が高い質問文に対応する解答文を返している．形態素解析にJanomeを使用し，文章のベクトル化にDoc2Vecを使用し，
類似度はコサイン類似度を使用した．

チャットボットの相手として，漫画「ピンポン」に登場するペコというキャラクターと対話が行えるように，一問一答集を作成した．

参考資料：
- 「Python+Doc2Vecで似た意味を持つ文章を調べる」データサイエンス情報局.2019/10/05.
< https://analysis-navi.com/?p=2293 >(2021/02/10アクセス)
- 「[gensim]Doc2Vecの使い方」Qiita. 2017/12/12.
< https://qiita.com/asian373asian/items/1be1bec7f2297b8326cf >(2021/02/10 アクセス)


## 最適化問題
走行距離を目的関数とし，目的関数を最大化する．
目的関数の値は，シミュレーションによって評価される．

設計変数は以下の20の要素からなる．

0~1: タイヤの半径

2~3: タイヤの配置

4~5: タイヤの密度

6~17: 車体の各頂点座標...(x,y)のセットが6個

18: 車体の密度

19: タイヤの数(0~2)

すべて0~1で正規化されている．
 
## UNDX
Unimodal Normal Distribution Crossover (UNDX)は，交叉を主探索オペレータとして考える．
方法として，3つの親から正規乱数を使用して2つの子個体を生成する．
3つの親のうち，2つの親を結ぶ直線を主探索軸とし，その周辺に子個体が生成される．

詳しいUNDXでの子個体生成式を以下に示す．UNDXの実装は[Ono 2003]を参考にした．

![undx_shiki](img/undx_offspring.jpg)

 [Ono 2003]Ono, Isao, Hajime Kita, and Shigenobu Kobayashi. "A real-coded genetic algorithm using the unimodal normal distribution crossover." Advances in evolutionary computing. Springer, Berlin, Heidelberg, 2003. 213-237.
                                                                                               
## プログラムの構成  
基本的には遺伝的アルゴリズムの流れになっている．

1. 個体集団をランダムに生成．
2. 個体情報から車を設計し，シミュレーションから個体の評価
3. 交叉(UNDX)の適用
4. 突然変異の適用
5. 次世代に残す個体の選択
6. 2~5の操作を収束するまで繰り返す．


## 環境
- ファイル構造
    - src
       - run.py...プログラムを実行するためのもの
       - setting.py...パラメータの値を設定する
       - classes.py...車の設計からシミュレーションまでを記述している

- 実行環境
  - Python 3.7 or 3.6
  - box2d==2.3.10
  - pygame
  - matplotlib

- プログラムの動かし方
  - 実行環境を作成
  - cd [run.pyなど3つのプログラムがある場所]
  - python run.py  を入力
  
## 補足
- 交叉や突然変異は，classes.pyのundx()関数やmutation()関数に記述している．
- 遺伝的アルゴリズムの部分は，主にnext_generation関数に記述している．
- 
