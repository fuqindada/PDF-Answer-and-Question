import sys
import torch
from sentence_transformers import SentenceTransformer

# 加载预训练的句子嵌入模型
model = SentenceTransformer(sys.argv[1]) # ./bge-m3
# 定义句子列表
sentences_1 = ["据传，菜煎饼起源于13世纪中期，当时明军与元军在峄州展开激战，当地人民死伤惨重。后来，从山西洪洞一带移民至此的民众，"
               "仅靠官府发放的半斤粮食无法充饥，便将五谷掺水，用石磨研磨成浆糊，放在铁板上，用竹片摊成“薄纸”，并大量包装蔬菜、野菜、草根和树叶，以此充饥。"]
sentences_2 = ["菜煎饼是山东鲁南地区的一种大众食品，制作原料主要有面粉、杂粮、鸡蛋等，老少兼宜，俗称“中国热狗”，流行于枣庄、济宁、临沂、徐州等鲁南地，后传布周围省市。"
               "上个世纪七十年代，枣庄农村的生活还是很匮乏的，老百姓的主食以煎饼为主，煎饼的主要原料是地瓜干，条件好一点的可稍放点小麦，刚烙煎饼时鏊子凉，"
               "需把鏊子烧热擦些油才容易把煎饼从鏊子上揭下来，这样烙出的煎饼就很厚，稍等一会儿，煎饼凉了又板又硬，很难下咽。"
               "因此我们枣庄人把烙煎饼时前几张和后几张煎饼称为滑鏊子煎饼或滑塌子。这样的煎饼很难下咽，但丢了又可惜，精明的母亲们就将大白菜，"
               "土豆丝，粉条，豆腐切碎加点猪油，放上辣椒面，花椒面和盐，做成了所谓的菜煎饼，这样一来不但滑鏊子煎饼解决了，并且做出的煎饼还特别好吃，"
               "这样一传十、十传百，于是菜煎饼就在农村各家各户传开了！"]
sentences_3 = ["八十年代末期，农村土地实行了联产承包责任制已有多年，农民在农忙季节忙耕种，农闲时便有了剩余时间，有的农村妇女就到街上摆地摊卖菜煎饼挣点零花钱。"
               "一辆三轮车，一盘小饼鍪，一个蜂窝炉，一个切菜板，几样时令蔬菜，食客现场点菜，业主现场烙制，简简单单的营生，成为枣庄街头一道风景。"
               "许许多多的农村人多了一个贴补家用的挣钱机会，人们生活也多了一道风味小吃。到了九十年代末期，就连一些男人也走上了街头卖起了菜煎饼。"]
sentences_4 = ["1993年5月，山东省劳动厅在枣庄举办特级厨师培训班，聘请江苏省淮安商业技工学校一行6人赴枣庄讲学，这六人当中有校领导、高级讲师、特级厨师，途中经台儿庄区招待所午餐，"
               "席上菜肴丰盛，但惟有“菜煎饼”被其六人齐呼：“天下第一美食”。"]
sentences_5 = ["山东菜煎饼如何做呢？首先要热锅，放油（油要多），下豆腐中火翻炒至金黄，放入之前切好的粉条，继续翻炒几分钟，加入适量的盐。"
               "再放入切好的韭菜，翻炒几下搅匀即可（千万不可炒过了，韭菜要生生的），撒味精出锅。将煎饼摊开，用勺子舀上适量的韭菜馅儿，用勺背整匀。好了之后，可以将两边向中间折叠形成长方形，一张煎饼就做好了。"]
# 获取句子的嵌入向量表示、
sentences_embeddings_1 = torch.from_numpy(model.encode(sentences_1, normalize_embeddings=True))
sentences_embeddings_2 = torch.from_numpy(model.encode(sentences_2, normalize_embeddings=True))
sentences_embeddings_3 = torch.from_numpy(model.encode(sentences_3, normalize_embeddings=True))
sentences_embeddings_4 = torch.from_numpy(model.encode(sentences_4, normalize_embeddings=True))
sentences_embeddings_5 = torch.from_numpy(model.encode(sentences_5, normalize_embeddings=True))
# 合并所有的句子嵌入表示
all_sentences_embeddings = torch.cat(
    [sentences_embeddings_1, sentences_embeddings_2, sentences_embeddings_3, sentences_embeddings_4, sentences_embeddings_5],
    dim=0)

# 定义查询句子
queries_1 = ["菜煎饼的制作原料有哪些？"]
queries_2 = ["菜煎饼的组成是什么？"]
queries_3 = ["做菜煎饼需要什么？"]
# 获取查询句子的嵌入向量表示
queries_embeddings_1 = torch.from_numpy(model.encode(queries_1, normalize_embeddings=True))
queries_embeddings_2 = torch.from_numpy(model.encode(queries_2, normalize_embeddings=True))
queries_embeddings_3 = torch.from_numpy(model.encode(queries_3, normalize_embeddings=True))

# 计算查询句子与所有句子的相似度
similarity_queries_1_sentences = queries_embeddings_1 @ all_sentences_embeddings.T
similarity_queries_2_sentences = queries_embeddings_2 @ all_sentences_embeddings.T
similarity_queries_3_sentences = queries_embeddings_3 @ all_sentences_embeddings.T

# 打印numpy size
print("sentences_vector dimension:", sentences_embeddings_1.size()) # [1, 1024]
print("sentences_vector dimension:", all_sentences_embeddings.size()) # [5, 1024]
print("sentences_vector dimension:", queries_embeddings_1.size()) # [1, 1024]
# 打印相似度结果（几个问题都是在问制作原料，从字面来看，我们预期三个查询与sentences_2 和 sentences_5 的相似度较高）
print("Query 1 Similarity:", similarity_queries_1_sentences)
print("Query 2 Similarity:", similarity_queries_2_sentences)
print("Query 3 Similarity:", similarity_queries_3_sentences)