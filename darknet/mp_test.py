from multiprocessing import Pool, cpu_count

# 計算數值平方的函數
def f(x):
    return x*x

cpus = cpu_count()
# 建立含有 4 個工作者行程的 Pool
input = [i for i in range(1, cpus+1)]
with Pool(processes=cpus) as p:
    # 以 map 平行計算數值的平方
    print(p.map(f, input))