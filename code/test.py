# import numpy as np
# from sklearn.random_projection import SparseRandomProjection

# # 假设你有一个包含多个n*k数组的batch_array
# # batch_array 的形状为 (batch, n, k)
# batch_array = np.random.rand(10, 100, 50)  # 示例数据

# # 创建 SparseRandomProjection 对象
# sparse_random_projector = SparseRandomProjection()

# # 对每个 n*k 数组进行变换
# transformed_batch_array = []
# for array in batch_array:
#     # 将 n*k 数组转换成矩阵
#     matrix = array.reshape(array.shape[0], -1)
#     # 进行变换
#     transformed_matrix = sparse_random_projector.fit_transform(matrix)
#     # 将变换后的矩阵重新组织成原始形状
#     transformed_array = transformed_matrix.reshape(array.shape[0], -1)
#     # 将变换后的数组加入列表
#     transformed_batch_array.append(transformed_array)

# # 将变换后的数组列表组装回 batch
# transformed_batch_array = np.array(transformed_batch_array)

# # 现在 transformed_batch_array 的形状为 (batch, n, 新的特征维度)
import numpy as np

# 定义函数f()，接受一个n*m的向量，返回一个n*k的向量
def f(input_vector):
    # 这里假设进行了某种变换操作
    # 这里的示例只是将输入向量的每一行复制k次
    k = 3
    return np.tile(input_vector, (1, k))

# 假设有一个b*n*m的输入向量
b = 5
n = 4
m = 2
input_vector = np.random.rand(b, n, m)

# 对每个b*n*m的子向量应用函数f()，得到b*n*k的向量
output_vectors = np.zeros((b, n, f(input_vector[0]).shape[1]))
for i in range(b):
    output_vectors[i] = f(input_vector[i])

# 整合变换后的向量，形成一个b*n*k的向量
final_output = output_vectors

# 打印结果
print(final_output.shape)  # 输出结果为(b, n, k)
