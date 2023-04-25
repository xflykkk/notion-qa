import numpy as np
import faiss

# 生成随机向量数据
dimension = 64
num_vectors = 10000
data = np.random.random((num_vectors, dimension)).astype('float32')

# 创建Faiss索引
index = faiss.IndexFlatL2(dimension)
index.add(data)

# 查询向量
query_vector = np.random.random((1, dimension)).astype('float32')
k = 10  # 搜索最近的5个向量
distances, indices = index.search(query_vector, k)

print("查询结果：")
print("距离：", distances)
print("索引：", indices)
