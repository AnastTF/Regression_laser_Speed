#импортируем библиотеки pandas, nu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel("значение скорости для АМг3М толщиной 1мм.xls")
print(data.shape)#количество колонок и столбцов таблицы
print(data.head())#выыодит первые 5 строк таблицы
#построим точки на графике
#data.plot(x='Количество кадров', y='Значение рабочей скорости, м/с', style='o')
#plt.title('Зависимость скорости режущего инструмента на рабочем ходе от числа кадров для АМг3М, 1мм')
#plt.xlabel("Количество кадров УП")
#plt.ylabel("Von(факт),м/с")
#plt.show()
#plt.grid()
x=data.iloc[:,:-1].values#столбец количество кадров

#x=data['Количество кадров']
y=data.iloc[:,1].values#столбец значений скорости рабочего хода режущего инструмента
#y=data['Значение рабочей скорости, м/с']
#вычислим коэффициент корреляции
coeffKorr=data['Количество кадров'].corr(data['Значение рабочей скорости, м/с'])
print("coeffKorr=", coeffKorr)
#разделяем данные на тестовые и обучающие
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)#разбиваем данные на 80% - обучающие, 20% - тестовые
print(x_train)

from sklearn.linear_model import LinearRegression
model=LinearRegression()#вычисляем оптимальные значения весов
z=model.fit(np.log(x_train),y_train)
print(z.intercept_)#значение перехвата
print(z.coef_)#значение наклона(коэффициент при х)

z_train = model.predict(np.log(x_train))# Для сравнения качества делаем прогноз и на тех данных, на которых обучались
z_test = model.predict(np.log(x_test))
#делаем прогнозы по модели по тестовым данным
y_pred=model.predict(np.log(x_test))
#сравниваем фактические выходные значения для x_test с прогнозируемыми значениями
df=pd.DataFrame({'Actual':y_test,'Predicted': y_pred})
print(df)

#оценка алгоритма
#найдем коэффициент детерминации
r_sq=model.score(np.log(x_train),y_train)
print('r_sq=', r_sq)
#найдем значение средней абсолютной ошибки-MAE, среднеквадратичной ошибки - MSE и средняя квадратная ошибка - RMSE
from sklearn import metrics
print('Mean Abcolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
x_pr=[]
#значения для построения регрессирующей функции на графике
for row in x_train:
    for i in row:
        x_pr.append(i)
x_pr.sort()
x_pr=np.array(x_pr).reshape((-1,1))#переводим в двумерный массив
z_pr = model.predict(np.log(x_pr))

lab=str(round((*z.coef_),3))+' *ln(x)+ '+str(round(z.intercept_,3))
#ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ   
plt.plot(x_train, model.predict(np.log(x_train)),color='green',marker='o',ls='')#marker-маркер точек, ls - стиль линии
plt.plot(x_test, model.predict(np.log(x_test)),color='blue',marker='o',ls='')
plt.plot(x_pr, z_pr,color='red',label=lab)
plt.title("Зависимость скорости режущего инструмента"+"\n"+"на рабочем ходе от числа кадров для АМг3М, 1мм")
plt.xlabel("Количество кадров УП")
plt.ylabel("Von(факт),м/с")
plt.grid()
plt.legend()
