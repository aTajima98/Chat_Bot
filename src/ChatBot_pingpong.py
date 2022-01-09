import re
import libmstdn
import csv
#modelの読み込み
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
model = Doc2Vec.load("jawiki.doc2vec.dbow300d.model")
#形態素解析
from janome.tokenizer import Tokenizer
#cos類似度
import numpy as np
import random


#Mastodon　ホスト
#注意："https://"を付けず、ホスト名のみを記述する
HOST = "memphis.ibe.kagoshima-u.ac.jp"

#Mastodon APIアクセストークンを取得して以下のTOKENに代入する
#Mastodonのサイトで登録を行う。ログイン後、設定→開発→新規アプリ
TOKEN = "4cbdcbb0607a080ee47cbf47479690b2eb4a21c7803aceb7d69f1fc87dec5a5c"

def remove_html_tags( content):
    return re.sub("<[^>]*>","",content).strip()

def is_to_me(status,my_id):
    for mention in status["mentions"]:
        if mention["id"] == my_id:
            return True
    return False

#形態素解析
def sep_by_janome(text):
    t=Tokenizer()
    tokens=t.tokenize(text)
    docs=[]
    for token in tokens:
        docs.append(token.surface)
    return docs

#返答を求める
def generate_reply(status,my_name):
    received_text = remove_html_tags(status["content"])
    toot_from = status["account"]["username"]

    # 文章の入力
    text = received_text[14:]
    # 形態素解析
    tokens = sep_by_janome(text)
    input_vec = model.infer_vector(tokens)

    # ベクトル化
    document_vecs = []
    for d in documents:
        document_vecs.append(model.infer_vector(sep_by_janome(d)))

    # 類似度計算
    v1 = np.linalg.norm(input_vec)
    cos_sim = []
    for v2 in document_vecs:
        cos_sim.append(np.dot(input_vec, v2) / (v1 * np.linalg.norm(v2)))
    doc_sort = np.argsort(np.array(cos_sim))[::-1]
    cos_sort = sorted(cos_sim, reverse=True)

    for i in range(rank_size):
        print(documents[doc_sort[i]],cos_sort[i],"//",answers[doc_sort[i]])

    # 類似度低い時
    # 「よくわかりません」集を使う。
    un = 5
    if cos_sort[0] < 0.7:
        un = random.randint(1, 5)
        if un == 1:
            return "何？"
        if un == 2:
            return "ケロケロ。"
        if un == 3:
            return "ふうん。"
        if un == 4:
             return "あらまっ…"
        if un == 5:
             return "周りくどいのね、オイラにはサッパリ…"

    #同じくくりの質問まとめる
    i=0
    count=1
    while cos_sort[i]-cos_sort[i+1]<0.02:
        count+=1
    #チョイス
    if not count==1:
        rand_choice=random.randint(0,count)
        print(answers[doc_sort[rand_choice]])
        return answers[doc_sort[rand_choice]]

    print(answers[doc_sort[0]])
    return answers[doc_sort[0]]


#変数置き場
#文章の読み込み
documents=[]
answers=[]
#ランキング上位５表示
rank_size=5

#Main

api = libmstdn.MastodonAPI(mastodon_host=HOST, access_token=TOKEN)

account_info = api.verify_account()
my_id = account_info["id"]
my_name = account_info["username"]
print("Started bot, name:{}, id:{}".format(my_name,my_id))

#一問一答集の読み込み
with open("peko.csv", "r", encoding="utf-8") as f:
    rd = csv.reader(f)
    for i in rd:    #空白行スキップする
        if not len(i[0]) == 0:
            documents.append(i[0])
            answers.append(i[1])
    del documents[0]
    del answers[0]
    f.close()

#学習済みモデルの使用
model = Doc2Vec.load("jawiki.doc2vec.dbow300d.model")
#質問集でーたからモデルの学習（2行）
#trainings = [TaggedDocument(words = data.split(),tags = [i]) for i,data in enumerate(documents)]
#model=Doc2Vec(documents=trainings,dm=1,size=300,window=8,min_count=10,workers=4)

stream = api.get_user_stream()
for status in stream:
    if is_to_me(status,my_id):
        received_text = remove_html_tags(status["content"])
        toot_id = status["id"]
        toot_from = status["account"]["username"]
        print("received from {}:{}".format(toot_from,received_text))
        reply_text = "@{} {}".format(toot_from,generate_reply(status,my_name))
        api.toot(reply_text,toot_id)
        print("post to {}:{}".format(toot_from,reply_text))
