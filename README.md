# nlp_project

### Env
PyTorch2.2.0-Ubuntu 22.04

实现了一个简易逻辑的notebook，经验以向量形式存在库里。
用的时候查最相似的一条经验辅助做题（以后可以改成top k）。
如果距离大于DISTANCE_THRESHOLD，那就不采用经验,直接做。
DISTANCE_THRESHOLD这个参数待实验，现在是瞎写的。

测试流程：
运行学习脚本:python step1_learning.py
运行测试脚本:python step2_testing.py
