"""
El "Conjunto de datos sobre emisiones y atributos de vehículos" contiene información completa sobre varios vehículos fabricados en el año 
2000. Incluye detalles como marca, modelo, clase de vehículo, tamaño del motor, número de cilindros, tipo de transmisión y tipo de 
combustible. . Además, el conjunto de datos proporciona rangos de consumo de combustible y emisiones de CO2, ofreciendo información sobre 
el impacto ambiental de cada vehículo. El conjunto de datos abarca una amplia gama de tipos de vehículos, desde compactos hasta medianos, 
e incluye tanto modelos convencionales como de alto rendimiento. Con esta información, los analistas e investigadores pueden estudiar las 
tendencias en las características de los vehículos, la eficiencia del combustible y las emisiones. Este conjunto de datos sirve como un 
recurso valioso para comprender el panorama automotriz e informar debates sobre sostenibilidad ambiental y políticas de transporte."""

"""Regresión lineal
La regresión lineal modela la relación entre variables dependientes (y) e independientes (x) a través de una línea recta o un plano. Asume 
una relación lineal y apunta a minimizar las diferencias al cuadrado entre los valores observados y predichos. Ampliamente utilizado para 
la predicción y la comprensión en diversos campos como la economía, las finanzas y las ciencias.

1. Recopilación de datos
Importar bibliotecas y leer conjuntos de datos"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings


fc = pd.read_csv("FuelConsumption (1).csv")
print(fc)
plt.style.use('dark_background')
#2. Análisis exploratorio de datos
#En EDA realizamos tareas como limpiar, describir y modificar los datos en consecuencia.
print(fc.head(10))  #Muestra las primeras 5 filas de los datos

print(fc.shape) #dimensión de los datos

print(fc.info())  #info sobre los datos

print(fc.describe())  #estadísticas de los datos

print(fc.columns)   #leyendo las columnas

fc.columns
cfc = fc[['ENGINE SIZE','CYLINDERS','FUEL CONSUMPTION','COEMISSIONS ']]
print(cfc.head())

#considerando solo las columnas importantes y almacenando en cfc varaiable

#3. Visualización de datos
#La visualización de datos implica presentar datos en formatos gráficos o pictóricos, lo que ayuda a comprender patrones y relaciones 
# dentro de los datos.
visual = cfc[['ENGINE SIZE','CYLINDERS','FUEL CONSUMPTION','COEMISSIONS ']]
print(visual.hist())
plt.show()

#visualizando los datos en histograma
for i in cfc[['ENGINE SIZE','CYLINDERS','FUEL CONSUMPTION','COEMISSIONS ']]:
    plt.scatter(cfc[i],cfc['COEMISSIONS '],color = 'green')
    plt.xlabel(i)
    plt.ylabel("Emisión\n")
    plt.show()
#visualizar los datos en un diagrama de dispersión

#4.Entrenamiento de modelos
#En el entrenamiento de modelos dividimos los datos en tren (80%), prueba (20%) y entrenamos el modelo.
mask = np.random.rand(len(cfc))<0.80
train = cfc[mask]
test = cfc[~mask]
# sumergir el conjunto de datos en entrenamiento (80%) y prueba (20%)
#Aplicar regresión lineal en datos de entrenamiento
#calcular coeficiente e intersección
coefficient=[]
intercept=[]
regress_model = {}
for i in ['ENGINE SIZE','CYLINDERS','FUEL CONSUMPTION']:
    reg = linear_model.LinearRegression()
    train_x = np.asanyarray(train[[i]])
    train_y = np.asanyarray(train[['COEMISSIONS ']])
    reg.fit(train_x, train_y)
    regress_model[i] = reg
    print("Relación entre {} y {}".format(i, "'coemisión'"))
    print("Coeficiente :", reg.coef_)
    print("Interceptar :", reg.intercept_)
    coefficient.append(reg.coef_)
    intercept.append(reg.intercept_)
    print('\n')
    
#Utilice el parámetro de regresión para modelar una ecuación lineal
j=0
for i in train[['ENGINE SIZE','CYLINDERS','FUEL CONSUMPTION']]:
    plt.scatter(train[i], train['COEMISSIONS '],  color='blue')
    x=train[i].values
    print(x.shape)
    y=coefficient[j][0]*x + intercept[j]
    print(y.shape)
    l=len(y)
    y=np.reshape(y,(l,))
    plt.plot(x, y, '-r')
    plt.xlabel(i)
    plt.ylabel("Emisión\n")
    plt.show()
    j=j+1
    
#5.Rendimiento del modelo
#La eficacia de un modelo predictivo para capturar patrones con precisión y realizar predicciones confiables sobre datos invisibles.
#Evaluar el rendimiento del modelo en datos de prueba
#calculando Error absoluto medio, suma residual de cuadrados, puntuación R2

for i in train[['ENGINE SIZE','CYLINDERS','FUEL CONSUMPTION']]:
    test_x = np.asanyarray(test[[i]])
    test_y = np.asanyarray(test[['COEMISSIONS ']])
    test_y_ = regress_model[i].predict(test_x)
    print("Error de ajuste entre {} y {}".format(i,"'CO2EMISSIONS'"))
    print("Error absoluto medio: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Suma residual de cuadrados (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("Puntuación R2: %.2f" % r2_score(test_y_ , test_y) )
    print('\n')
    

my_data= pd.read_csv("FuelConsumption (1).csv")
my_data.dropna(inplace=True)
my_data
plt.style.use('dark_background')

Lbl=LabelEncoder()
my_data.MAKE=Lbl.fit_transform(my_data.MAKE)
my_data.MODEL=Lbl.fit_transform(my_data.MODEL)
# my_data.VEHICLE CLASS=Lbl.fit_transform(my_data.VEHICLE CLASS)
my_data.TRANSMISSION=Lbl.fit_transform(my_data.TRANSMISSION)
my_data.FUEL=Lbl.fit_transform(my_data.FUEL)
my_data["VEHICLE CLASS"]=Lbl.fit_transform(my_data["VEHICLE CLASS"])
my_data

my_data.info

sns.histplot(data=my_data)
plt.show()

sns.pairplot(my_data[["Year","MAKE","MODEL","VEHICLE CLASS","ENGINE SIZE","CYLINDERS","TRANSMISSION","FUEL","FUEL CONSUMPTION"]], hue='FUEL CONSUMPTION', palette='husl')
plt.show()

my_data.hist(figsize=(20,10),bins = 50)

plt.figure(figsize=(20,10))
sns.heatmap(my_data.corr(),annot=True,cmap="coolwarm")
plt.show()

sns.jointplot(data=my_data, x="FUEL CONSUMPTION", y="Year", height=5, ratio=2, marginal_ticks=True)
plt.show()

sns.stripplot(data=my_data, x="FUEL CONSUMPTION", y="MODEL")
plt.show()

X=my_data.drop(["FUEL CONSUMPTION"],axis=1)
print(X)

y=my_data["FUEL CONSUMPTION"]
y=pd.DataFrame(y)
print(y)

#Escalador estándar para datos
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X= scaler.fit_transform(X)
print(X)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
y= scaler.fit_transform(y)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=33, shuffle =True)

reg_moduel=RandomForestRegressor(n_estimators=150,random_state=33)
reg_moduel.fit(X_train,y_train)
#Calcular detalles
print('La puntuación del tren de regresor forestal aleatorio es : ' ,  reg_moduel.score(X_train, y_train))
print('La puntuación de la prueba del regresor forestal aleatorio es : ' , reg_moduel.score(X_test, y_test))

warnings.filterwarnings("ignore")
data = pd.read_csv("FuelConsumption (1).csv")
data.head(2)
plt.style.use('dark_background')
data.info()

data.describe()

# Seleccionar solo columnas numéricas
numeric_data = data.select_dtypes(include=[np.number])

# Matriz de correlación
corr = numeric_data.corr()

# Trazar el mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.show()

# Distribución del consumo de combustible
plt.figure(figsize=(10, 6))
sns.histplot(data['FUEL CONSUMPTION'], bins=20, kde=True, color='skyblue')
plt.title('Distribución del consumo de combustible\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('El consumo de combustible\n')
plt.ylabel('Frecuencia\n')
plt.show()

# Diagrama de caja del consumo de combustible por clase de vehículo
plt.figure(figsize=(12, 8))
sns.boxplot(x='VEHICLE CLASS', y='FUEL CONSUMPTION', data=data, palette='Set2')
plt.title('Consumo de combustible por clase de vehículo\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Clase de vehículo\n')
plt.ylabel('El consumo de combustible\n')
plt.xticks(rotation=45)
plt.show()

# Contar gráfico de clase de vehículo
plt.figure(figsize=(10, 6))
sns.countplot(y='VEHICLE CLASS', data=data, order=data['VEHICLE CLASS'].value_counts().index, palette='pastel')
plt.title('Conteo de vehículos por clase de vehículo\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Contar\n')
plt.ylabel('Clase de vehículo\n')
plt.show()

# Gráfico de barras del consumo medio de combustible por marca
plt.figure(figsize=(12, 8))
mean_fuel_consumption_by_make = data.groupby('MAKE')['FUEL CONSUMPTION'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=mean_fuel_consumption_by_make.values, y=mean_fuel_consumption_by_make.index, palette='muted')
plt.title('Consumo medio de combustible por marca (Top 10)\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Consumo medio de combustible\n')
plt.ylabel('Hacer\n')
plt.show()

# Diagrama de caja del tamaño del motor por tipo de transmisión
plt.figure(figsize=(12, 6))
sns.boxplot(x='TRANSMISSION', y='ENGINE SIZE', data=data, palette='husl')
plt.title('Tamaño del motor por tipo de transmisión\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Transmisión\n')
plt.ylabel('Tamaño de la maquina\n')
plt.show()

# Gráfico de dispersión del consumo de combustible frente al tamaño del motor coloreado por tipo de transmisión
plt.figure(figsize=(10, 6))
sns.scatterplot(x='ENGINE SIZE', y='FUEL CONSUMPTION', hue='TRANSMISSION', data=data, palette='Set2')
plt.title('Consumo de combustible versus tamaño del motor por tipo de transmisión\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tamaño de la maquina\n')
plt.ylabel('El consumo de combustible\n')
plt.show()

# Distribución del tamaño del motor por clase de vehículo
plt.figure(figsize=(12, 8))
sns.boxplot(x='VEHICLE CLASS', y='ENGINE SIZE', data=data, palette='Set2')
plt.title('Distribución del tamaño del motor por clase de vehículo\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Clase de vehículo\n')
plt.ylabel('Tamaño de la maquina\n')
plt.xticks(rotation=45)
plt.show()

# Gráfico de barras del consumo medio de combustible por tipo de transmisión
mean_fuel_consumption_by_transmission = data.groupby('TRANSMISSION')['FUEL CONSUMPTION'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=mean_fuel_consumption_by_transmission.index, y=mean_fuel_consumption_by_transmission.values, palette='coolwarm')
plt.title('Consumo medio de combustible por tipo de transmisión\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de transmisión\n')
plt.ylabel('Consumo medio de combustible\n')
plt.xticks(rotation=45)
plt.show()

# Diagrama de caja del tamaño del motor por tipo de combustible
plt.figure(figsize=(10, 6))
sns.boxplot(x='FUEL', y='ENGINE SIZE', data=data, palette='Set3')
plt.title('Tamaño del motor por tipo de combustible\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de combustible\n')
plt.ylabel('Tamaño de la maquina\n')
plt.show()

# Gráfico de enjambre de consumo de combustible por recuento de cilindros
plt.figure(figsize=(10, 6))
sns.swarmplot(x='CYLINDERS', y='FUEL CONSUMPTION', data=data, palette='muted')
plt.title('Consumo de combustible por cantidad de cilindros\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Recuento de cilindros\n')
plt.ylabel('El consumo de combustible\n')
plt.show()