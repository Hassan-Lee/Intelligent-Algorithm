# GSA（Genetic Simulated Annealing）

## 算法过程

1. 生成初始种群（N个）

2. N个点产生L个新解（在某温度下）

3. metropolis接收准则舍弃一些点

4. 计算每个点的适应值，选择N个点作为生存集（GA）（锦标赛）

5. 如果第四步中适应值最大的点被舍弃了，那么就随机舍弃一个点，又把它加回来。
(这一步在代码中处理为：先把适应值最大的点加入，再选择N-1个点和这个点一起作为生存集)

6. 降温，重复以上过程

## 文件说明

+ GSA-python-1-Song.py:
+ GSA-python-2-Song.py:
+ GSA-python-3-Song.py:
+ GSA-python-4-Song.py:for循环调参，但这个文件中选择生存集的方法错误，因此调的参意义也不大。
+ GSA-python-5-Song.py:
+ GSA-python-6-Song.py:一个父类和三个子类（以Rastrigrin为例）。效果最好的是fastGSA。
+ GSA-python-7-Song.py:三个函数的数值实验（未调参）。直接运行是Griewangk的，如果要运行其他函数，只需要注释Griewangk的三行代码，取消其他函数的三行代码的注释即可。