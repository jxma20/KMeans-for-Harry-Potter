from TFIDF_Advanced import *
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 处理Harry Potter文档中的换行符，并保存到process_Harry_Potter.txt中
file = open("Harry Potter 1 - Sorcerer's Stone.txt", encoding='gb18030', errors='ignore')
data = file.read().split('\n')
data = list(set(data).difference(set([data[i] for i in range(len(data)-1, -1, -1) if data[i] is ''])))
file.close()
print(len(data))
# [print(i) for i in data]

p = TfIdf(corpus_filename="corpus.txt", stopword_filename=None, DEFAULT_IDF=1.5)
p.merge_corpus_document("动作.txt")
p.merge_corpus_document("环境.txt")
p.merge_corpus_document("语言.txt")

n_corpus = len(p.term_num_docs)

for line in data:
    p.add_input_document(line)


tfidf = []
for doc in data:
    tfidf.append(np.array(p.get_str_keywords(doc)))

tfidf = np.array(tfidf)
print("docs num is:", len(p.term_num_docs))
# print(f"the term num docs is: {p.term_num_docs}")
# [print(i) for i in tfidf]
print(f"the tfidf shape is {tfidf.shape}")

# 对term-doc矩阵进行pca降维
pca = PCA(n_components=n_corpus)
pca.fit(tfidf)
sum = 0
maxindex = 0
for i, element in enumerate(pca.explained_variance_ratio_):
    if sum > 0.95:
        maxindex = i
        break
    sum += element

print(f"pca结果，前{maxindex}个主成分保留")

newX = pca.transform(tfidf)[:, : maxindex]
print(f"pca降维后term-docs矩阵的维数:{newX.shape}")

# 处理完成之后使用kmeans进行聚类
n_clusters = 5
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(newX)
res = cluster.labels_  # 各个样本的结果
element, counts = np.unique(res, return_counts=True)  # 对其进行计数，得到各个标签的数量
print(f"res content :{element}")
print(f"counts of cluster is {counts}")
print(f"res shape is :{res.shape}")

