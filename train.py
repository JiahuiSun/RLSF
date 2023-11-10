# reward模型设计，跟policy保持一致吧；输入是s和a，s用BEV表示，a用向量表示，分别用CNN和MLP处理吧
# 采集大量的数据保存起来；只要把原始数据保存下来，剩下的我都可以再处理
    # 样本保存成：(tau1, ap1) (tau2, ap2)
# 怎么构成对比样本？随便两两组合？

from dataset import test

test()