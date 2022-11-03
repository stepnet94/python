import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
from scipy import stats

a = np.array([1,2,3,4,5,6]) # создали массив
print(a.dtype) # целочисленные элементы int64
b = np.array([1,2,"3",4,True]) # не можеть быть разнородных элементов, все равно приведется все к одному - строке
print(b.dtype)
print(a[0]) # обращение к первому элементу
a[1] = 456 # запись по определенному индексу
c = a[[1, 1, 1, 1]] # новый список, состоящий только из эдементов по индексу 1 из списка а
d = a[[True, False, False, True, False, True]] # вернет только те значения, у которых True [1 4 6]
print(d)
print("==================================")
e = a.reshape(3,2) # создание двумерного массива - 3 строки, 2 столбца
print(e)
print("==================================")
t = np.array([1,2,3,4,5,6], 'float64') # явно указываем тип эдементов массива
print(t) # [1. 2. 3. 4. 5. 6.]
print(t.dtype)
print("==================================")

w = np.array([[456,564], [123, 645], [323, 553]]) # создание двумерного массива - 3 строки, 2 столбца
print(w)



# Расчёт моды, медианы и среднего с помощью библиотек numpy и scipy
sample = np.array([185, 175, 170, 169, 171, 175, 157, 172, 170, 172, 167, 173, 168, 167, 166,
              167, 169, 172, 177, 178, 165, 161, 179, 159, 164, 178, 172, 170, 173, 171])

print('mode:', stats.mode(sample))
print('median:', np.median(sample))
print('mean:', np.mean(sample))
print("==================================")

# Расчёт моды, медианы и среднего с помощью библиотеки pandas
sample1 = pd.Series([185, 175, 170, 169, 171, 175, 157, 172, 170, 172, 167, 173, 168, 167, 166,
              167, 169, 172, 177, 178, 165, 161, 179, 159, 164, 178, 172, 170, 173, 171])

print('mode1:', sample1.mode())
print('median1:', sample1.median())
print('mean1:', sample1.mean())

print(f'Range: {np.ptp(sample)} is equal max - min: {np.max(sample)- np.min(sample)}')
print(f'Standard deviation: {np.std(sample, ddof=1):.2f}')
plt.boxplot(sample, showfliers=1)
plt.show()


# ''' Считается, что значение IQ (уровень интеллекта) у людей имеет нормальное распределение
# со средним значением равным 100 и стандартным отклонением равным 15 (M = 100, sd = 15).
# Какой приблизительно процент людей обладает IQ > 125?

mean = 100
std = 15
IQ=125
# sf - Survival function = (1 - cdf) - Cumulative distribution function
print(f"Только у {(stats.norm(mean, std).sf(IQ))*100:.2f}% людей, IQ>{IQ}")

# расчет 99% доверительного интервала для среднее = 10, стандартное отклонение = 5, размер выборки = 100
p = 0.99
mean = 10
std = 5
n = 100

se = std / math.sqrt(n)
alpha = (1-p)/2
sigma = stats.norm().isf(alpha)
сonfidence_interval = mean - sigma*se, mean + sigma*se
print('[%.2f; %.2f]' % сonfidence_interval)



