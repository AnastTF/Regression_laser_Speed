#����������� ���������� pandas, nu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel("�������� �������� ��� ���3� �������� 1��.xls")
print(data.shape)#���������� ������� � �������� �������
print(data.head())#������� ������ 5 ����� �������
#�������� ����� �� �������
#data.plot(x='���������� ������', y='�������� ������� ��������, �/�', style='o')
#plt.title('����������� �������� �������� ����������� �� ������� ���� �� ����� ������ ��� ���3�, 1��')
#plt.xlabel("���������� ������ ��")
#plt.ylabel("Von(����),�/�")
#plt.show()
#plt.grid()
x=data.iloc[:,:-1].values#������� ���������� ������

#x=data['���������� ������']
y=data.iloc[:,1].values#������� �������� �������� �������� ���� �������� �����������
#y=data['�������� ������� ��������, �/�']
#�������� ����������� ����������
coeffKorr=data['���������� ������'].corr(data['�������� ������� ��������, �/�'])
print("coeffKorr=", coeffKorr)
#��������� ������ �� �������� � ���������
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)#��������� ������ �� 80% - ���������, 20% - ��������
print(x_train)

from sklearn.linear_model import LinearRegression
model=LinearRegression()#��������� ����������� �������� �����
z=model.fit(np.log(x_train),y_train)
print(z.intercept_)#�������� ���������
print(z.coef_)#�������� �������(����������� ��� �)

z_train = model.predict(np.log(x_train))# ��� ��������� �������� ������ ������� � �� ��� ������, �� ������� ���������
z_test = model.predict(np.log(x_test))
#������ �������� �� ������ �� �������� ������
y_pred=model.predict(np.log(x_test))
#���������� ����������� �������� �������� ��� x_test � ��������������� ����������
df=pd.DataFrame({'Actual':y_test,'Predicted': y_pred})
print(df)

#������ ���������
#������ ����������� ������������
r_sq=model.score(np.log(x_train),y_train)
print('r_sq=', r_sq)
#������ �������� ������� ���������� ������-MAE, ������������������ ������ - MSE � ������� ���������� ������ - RMSE
from sklearn import metrics
print('Mean Abcolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
x_pr=[]
#�������� ��� ���������� �������������� ������� �� �������
for row in x_train:
    for i in row:
        x_pr.append(i)
x_pr.sort()
x_pr=np.array(x_pr).reshape((-1,1))#��������� � ��������� ������
z_pr = model.predict(np.log(x_pr))

lab=str(round((*z.coef_),3))+' *ln(x)+ '+str(round(z.intercept_,3))
#������������ �����������   
plt.plot(x_train, model.predict(np.log(x_train)),color='green',marker='o',ls='')#marker-������ �����, ls - ����� �����
plt.plot(x_test, model.predict(np.log(x_test)),color='blue',marker='o',ls='')
plt.plot(x_pr, z_pr,color='red',label=lab)
plt.title("����������� �������� �������� �����������"+"\n"+"�� ������� ���� �� ����� ������ ��� ���3�, 1��")
plt.xlabel("���������� ������ ��")
plt.ylabel("Von(����),�/�")
plt.grid()
plt.legend()
