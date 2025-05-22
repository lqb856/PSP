from dataloader import *

import numpy as np
import faiss
import os

def gen_knn_graph(k, data, path):
    """
    使用 Faiss 构建 KNN 图并按指定格式保存为二进制文件。
    
    参数:
        k (int): 要查找的邻居数量
        data (np.ndarray): 数据矩阵，shape = (n_samples, dim)
        path (str): 输出文件路径
    """
    n, dim = data.shape
    
    # 构建 Faiss 索引：IVF4096,PQ8x4fs 是一种高效的量化索引结构
    faiss_index = faiss.index_factory(dim, "IVF4096,PQ8x4fs", faiss.METRIC_L2)

    # 训练索引（如果数据量较小，可能需要调整聚类中心数量）
    print("开始训练索引...")
    faiss_index.train(data)

    # 添加向量到索引
    print("开始添加数据到索引...")
    faiss_index.add(data)

    # 查询所有点的 k+1 个最近邻（包含自己）
    print(f"开始查询 {k+1} 个最近邻...")
    faiss_index.nprobe = 128
    distances, indices = faiss_index.search(data, k + 1)

    # 去除每个点本身（第一个邻居就是它自己）
    indices = indices[:, 1:]  # shape: (n, k)

    # 保存为 IndexMips::Load_nn_graph 可读取的格式
    print(f"开始写入文件: {path}")
    with open(path, 'wb') as f:
        # 写入 k 值
        f.write(np.uint32(k))

        # 写入每个节点的邻居信息
        for i in range(n):
            # 写入节点 id（被忽略）
            f.write(np.uint32(i))
            # 写入 k 个邻居 id
            neighbors = indices[i].astype(np.uint32)
            f.write(neighbors.tobytes())

    print(f"KNN 图已成功保存至 {path}")



if __name__ == '__main__':

    K = 100
    path = "/home/lqb485508/dataset/LAION/LAION_base_imgemb_10M.fbin"

    # load data
    base = read_fbin(path)
    print(base.shape)

    gen_knn_graph(K, base, f"{K}_nn_graph.bin")

