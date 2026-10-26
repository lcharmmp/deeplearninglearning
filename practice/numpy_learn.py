import numpy as np
# #dtype规定数据类型对象
# student=np.dtype([('name','S20'),('age',np.int8)])
# a=np.array([('lee',122),('yre',20)],dtype=student)
# print(a[0]['age'])
# print(a['name'])
# print(f'a.itemsize={a.itemsize}')

# #axis轴，‘沿着0/1轴’
# a=np.array([[[1,2,3],[6,2,4]],[[1,1,1],[2,3,3]]])
# print(a)
# print(f'a.shape=\n{a.shape}')
# print(f'a.size={a.size}')
# print(f'a.ndim={a.ndim}')
# print(f'a.sum(axis=1)=\n{a.sum(axis=1)}')
# print(f'a.sum(axis=0)=\n{a.sum(axis=0)}')

# b=np.arange(12)
# print(f'b={b}')
# b=b.reshape(3,-1)
# print(f'b={b}')

# a=np.zeros((3,2),dtype=[('x',int),('y',int)])
# print(a)
# b=np.zeros_like(a, dtype=int)
# print(b)

rng = np.random.default_rng(42)
a=rng.integers(0,10,(2,3))
b=rng.integers(0,10,(3,4))
print(f'a=\n{a}\nb=\n{b}')
c=a@b
print(f'c=\n{c}')