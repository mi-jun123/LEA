import numpy as np


def calculate_weights(matrix):
    """
    计算判断矩阵的权重
    :param matrix: 判断矩阵
    :return: 权重向量
    """
    n = matrix.shape[0]
    # 计算每列的和
    column_sums = np.sum(matrix, axis=0)
    # 归一化矩阵
    normalized_matrix = matrix / column_sums
    # 计算每行的平均值作为权重
    weights = np.mean(normalized_matrix, axis=1)
    return weights


def consistency_check(matrix):
    """
    进行一致性检验
    :param matrix: 判断矩阵
    :return: 是否通过一致性检验，一致性比例
    """
    n = matrix.shape[0]
    # 计算矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # 获取最大特征值
    max_eigenvalue = np.max(eigenvalues).real
    # 计算一致性指标 CI
    ci = (max_eigenvalue - n) / (n - 1)
    # 平均随机一致性指标 RI
    ri_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    ri = ri_dict[n]
    # 计算一致性比例 CR
    cr = ci / ri
    if cr < 0.1:
        return True, cr
    else:
        return False, cr


# 示例6x6判断矩阵
matrix = np.array([
    [1, 1/2, 5, 5, 5, 5],
    [2, 1, 7, 7, 7, 7],
    [1/5, 1/7, 1, 1/2, 1/3, 1/3],
    [1/5, 1/7, 2, 1, 1/2, 1/2],
    [1/5, 1/7, 3, 2, 1, 2],
    [1/5,1/7, 3, 2, 1/2, 1]
])
# 计算权重
weights = calculate_weights(matrix)
print("权重向量:", weights)

# 进行一致性检验
pass_check, cr = consistency_check(matrix)
if pass_check:
    print("通过一致性检验，一致性比例 CR =", cr)
else:
    print("未通过一致性检验，一致性比例 CR =", cr)    