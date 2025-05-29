# SwinSparseMDT
SwinParseMDT: Brain Tumor Diagnosis Model

#数据结构
├── Train_kaggle.py                 # 主训练脚本
├── traine.py                # 包含训练和测试函数
├── Build_model.py           # 构建模型的函数
├── datasets/                # 包含训练和测试数据集
│   ├── Training/
│   └── Testing/
├── SparseSwinMDT              # 模型

稀疏Swin Transformer：对输入图像进行分层注意力建模，同时通过稀疏注意力机制降低计算量。
多路径软决策树（Multipath Soft Decision Tree）：结合多路径信息进行特征增强，提高分类稳定性。
支持4类分类任务：包括：胶质瘤（glioma）、脑膜瘤（meningioma）、脑垂体瘤（pituitary tumor）和非肿瘤（no tumor）。

环境要求
Python ≥ 3.8
PyTorch ≥ 1.12
CUDA 11+（建议使用GPU加速）
