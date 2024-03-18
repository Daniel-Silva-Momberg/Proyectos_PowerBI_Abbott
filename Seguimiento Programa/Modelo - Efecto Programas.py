# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:47:05 2024
@author: Daniel Silva Momberg
Title  : Efecto Programas
"""


# =============================================================================
# 1.LIBRERÍAS
# =============================================================================
import os
os.chdir("C:\\Users\\Daniel\\Desktop\\ABBOTT\\Proyectos Power BI\\Seguimiento Programa")

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm

# =============================================================================
# 2.DATOS
# =============================================================================
df = pd.read_csv("df_long.csv", sep=";", decimal=",")

mr = pd.read_excel("MercadoRelevante_IQVIA.xlsx")
mr = mr[["Cód_IMS","División"]].drop_duplicates()
mr.rename(columns={"Cód_IMS":"IMS"},inplace = True)

df = pd.merge(df,mr, how ="left", on = "IMS")

ventas = df[(df["TIPO"]=="Sell out") & (df["U_Medida"]=="Unidades")]
ventas = pd.pivot_table(ventas
                        ,values='Valor'
                        ,index=["Fecha","División"]
                        #,columns=
                        ,aggfunc='sum'
                        #,margins=True
                        #,margins_name='Total'
                        ,fill_value=0)

ventas.rename(columns={"Valor":"Demanda"},inplace=True)

programas = df[(df["TIPO"]!="Sell out") & (df["U_Medida"]=="Cupón")]
programas = pd.pivot_table(programas
                        ,values='Valor'
                        ,index=["Fecha","División"]
                        ,columns=["TIPO","CADENA"]
                        ,aggfunc='sum'
                        #,margins=True
                        #,margins_name='Total'
                        ,fill_value=0)

#programas = programas.reset_index()






df_rrss = df[(df["TIPO"]=="RRSS") & (df["U_Medida"]=="Cupón")]
df_rrss["Valor"] = df_rrss["Valor"] * -1
df_rrss = pd.pivot_table(df_rrss
                    ,values='Valor'
                    ,index=["Fecha","División"]
                    ,columns=["CADENA"]
                    ,aggfunc='sum'
                    #,margins=True
                    #,margins_name='Total'
                    ,fill_value=0)
df_rrss_columnas = df_rrss.columns.tolist()
df_rrss_columnas = ["RS_" + str(elemento) for elemento in df_rrss_columnas]
df_rrss.columns = df_rrss_columnas
df_rrss = df_rrss.reset_index()
df_rrss["ID"] = df_rrss["Fecha"].astype(str) + df_rrss["División"].astype(str) 


df_sm = df[(df["TIPO"]=="SM") & (df["U_Medida"]=="Cupón")]
df_sm["Valor"] = df_sm["Valor"] * -1
df_sm = pd.pivot_table(df_sm
                    ,values='Valor'
                    ,index=["Fecha","División"]
                    ,columns=["CADENA"]
                    ,aggfunc='sum'
                    #,margins=True
                    #,margins_name='Total'
                    ,fill_value=0)
df_sm_columnas = df_sm.columns.tolist()
df_sm_columnas = ["SM_" + str(elemento) for elemento in df_sm_columnas]
df_sm.columns = df_sm_columnas
df_sm = df_sm.reset_index()
df_sm["ID"] = df_sm["Fecha"].astype(str) + df_sm["División"].astype(str) 


df_ges = df[(df["TIPO"]=="GES") & (df["U_Medida"]=="Cupón")]
df_ges["Valor"] = df_ges["Valor"] * -1
df_ges['CADENA'] = df_ges['CADENA'].apply(lambda x: "GES_" + str(x) if pd.notna(x) else x)
df_ges = pd.pivot_table(df_ges
                    ,values='Valor'
                    ,index=["Fecha","División"]
                    ,columns=["CADENA","TIPO II"]
                    ,aggfunc='sum'
                    #,margins=True
                    #,margins_name='Total'
                    ,fill_value=0)
df_ges = df_ges.reset_index()
df_ges["ID"] = df_ges["Fecha"].astype(str) + df_ges["División"].astype(str)

 
df_otros = df[(df["TIPO"]=="OTROS") & (df["U_Medida"]=="Cupón")]
df_otros["Valor"] = df_otros["Valor"] * -1
# maestro_programas = pd.read_excel("Maestro_Programas.xlsx")
# maestro_programas = maestro_programas[['TIPO I', 'Tipo II']].drop_duplicates()
# maestro_programas.dropna(inplace =True)
# maestro_programas.rename(columns={"Tipo II":"TIPO II"},inplace=True)
# df_otros = pd.merge(df_otros,maestro_programas,how="left", on="TIPO II")
df_otros.isna().sum()
df_otros['CADENA'] = df_otros['CADENA'].apply(lambda x: "Otros_" + str(x) if pd.notna(x) else x)
df_otros = pd.pivot_table(df_otros
                    ,values='Valor'
                    ,index=["Fecha","División"]
                    ,columns=["CADENA","TIPO I"]
                    ,aggfunc='sum'
                    #,margins=True
                    #,margins_name='Total'
                    ,fill_value=0)
df_otros = df_otros.reset_index()
df_otros["ID"] = df_otros["Fecha"].astype(str) + df_otros["División"].astype(str)

columnas_a_eliminar = ["Fecha","División"]

datos = pd.merge(df_rrss, df_sm.drop(columnas_a_eliminar, axis=1), how='left', on = "ID")
datos = pd.merge(datos, df_ges.drop(columnas_a_eliminar, axis=1), how='left', on = "ID")
datos = pd.merge(datos, df_otros.drop(columnas_a_eliminar, axis=1), how='left', on = "ID")


ventas = df[(df["TIPO"]=="Sell out") & (df["U_Medida"]=="Unidades")]
ventas = pd.pivot_table(ventas
                        ,values='Valor'
                        ,index=["Fecha","División"]
                        #,columns=
                        ,aggfunc='sum'
                        #,margins=True
                        #,margins_name='Total'
                        ,fill_value=0)

ventas.rename(columns={"Valor":"Demanda"},inplace=True)
ventas = ventas.reset_index()
ventas["ID"] = ventas["Fecha"].astype(str) + ventas["División"].astype(str) 
datos = pd.merge(datos, ventas[["ID","Demanda"]], how='left', on = "ID")

del datos["ID"]
del datos["Fecha"]


# datos = pd.merge(ventas, programas, left_index=True, right_index=True, how='inner')
# datos = datos.loc[:, (datos != 0).any(axis=0)] #Elimina columnas 100% con 0
# datos = datos[(datos != 0).any(axis=1)] #Elimina filas 100% con 0
# datos = datos.loc[:, (datos != 0).sum() >= 0.7*len(datos)] #Elimina columnas 70% con 0


#datos.fillna(0.000000000001,inplace=True)
#datos.loc[:, datos.columns != 'Demanda'] = datos.loc[:, datos.columns != 'Demanda'] * -1

# datos = datos.reset_index()
# del datos["Fecha"]


datos = datos.fillna(0)
y = "Demanda"
x = datos.drop(y, axis=1)
y = datos.loc[:,"Demanda"]

x.hist(bins = 30, figsize=(20,20), color = 'r')


# Variables dummy
# *****************************************************************************
# Crear variables dummy
variables_dummy = pd.get_dummies(datos['División'])#, prefix='')
del datos["División"]

# TRANSFORMACIÓN LOGARITMICA
# *****************************************************************************
#Variables numéricas


for columna in datos.columns:
    if (datos[columna] != 0).all():
        datos[columna] = np.log(datos[columna])

#log (ln)

#datos[numeric_cols] = np.log(datos[numeric_cols])
constante = 1e-10  # Puedes ajustar este valor según sea necesario
for columna in datos.columns:
    if (datos[columna] == 0).any():
        datos[columna] = np.log(datos[columna] + constante)
        
# Concatenar las variables dummy al DataFrame original
datos = pd.concat([datos, variables_dummy], axis=1)
del datos["Women's Health & Urology"]

        
# =============================================================================
# Regresión Lineal Múltiple: Statsmodels
# =============================================================================
datos.dropna(inplace=True)    
datos.index = range(len(datos))
# Modelo 1
#---------------
# Variables
y = "Demanda"
x = datos.drop(y, axis=1)
y = datos.loc[:,"Demanda"]



x = sm.add_constant(x, prepend=True)
modelo = sm.OLS(endog=y, exog=x)
modelo = modelo.fit()
print(modelo.summary())

r2_1 = modelo.rsquared
modelo.f_pvalue

# r_squared_1 = modelo.rsquared
# f_values_1 = modelo.f_pvalue
        
        

# =============================================================================
# OUTLIERS
# =============================================================================


# Calcular la distancia de Cook
influence = modelo.get_influence()
cooks_distance = influence.cooks_distance
d_t = 4/(len(datos.index)-len(x.columns)-1)


#Gráfico distancia de cook
#-------------------------
sns.set(style='whitegrid')
plt.figure(figsize=(20, 10))
plt.scatter(range(len(cooks_distance[0])), cooks_distance[0], marker='o', color='b', s=30)
plt.axhline(y=d_t, color='r', linestyle='--', label='Umbral')
plt.title("Distancia de Cook"
          ,fontsize=30,color='#808080', loc='left', weight='bold', pad=20, y=1.05)
plt.xlabel(f'ID', fontsize=20, color='#808080')
plt.ylabel(f'Distancia de Cook', fontsize=20, color='#808080')
plt.xticks(fontsize=20, color='#808080', rotation=0)
plt.yticks(fontsize=20, color='#808080')
#Fondo
plt.gca().set_facecolor('#FFFFFF')
# eliminar el borde del gráfico
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
# eliminar las líneas de fondo
plt.grid(False)
# Mostrar y guardar el gráfico
plt.savefig("Distancia de cook.jpg")
plt.show()






# Crear un DataFrame con las distancias de Cook
#----------------------------------------------
outliers = pd.DataFrame({'Observation': range(len(cooks_distance[0])), 'Cook\'s Distance': cooks_distance[0]})
outliers = outliers[outliers["Cook\'s Distance"] > d_t]

# Eliminar outliers
datos = datos.drop(outliers.index)
datos.index = range(len(datos))


# =============================================================================
# NUEVO MODELO
# =============================================================================
# Variables
y = "Demanda"
x = datos.drop(y, axis=1)
y = datos.loc[:,"Demanda"]

x = sm.add_constant(x, prepend=True)
modelo = sm.OLS(endog=y, exog=x,)
modelo = modelo.fit()
print(modelo.summary())

r2_2 = modelo.rsquared
modelo.f_pvalue




# =============================================================================
# MULTICOLINEALIDAD
# =============================================================================
# #Eliminar la constante
# #del x["const"]

# def vif(x):
#     var_pred_labels = list(x.columns)
#     num_var_pred = len(var_pred_labels)
    
#     result = pd.DataFrame(index = ['vif'], columns = var_pred_labels)
#     result = result.fillna(0)
    
#     for ite in range(num_var_pred):
#         x_features = var_pred_labels[:]
#         y_feature = var_pred_labels[ite]
#         x_features.remove(y_feature)
        
#         x_i = sm.add_constant(datos[x_features], prepend=True)
#         y = datos[y_feature]
        
#         modelo = sm.OLS(endog=y, exog=x_i,)
#         modelo = modelo.fit()
        
#         result[y_feature] = 1/(1 - modelo.rsquared)
#     result = result.T
#     result = result.reset_index()
#     result.rename(columns={"index":"var"},inplace= True)
#     return result


# # Gráficamente
# #-------------
# del x["const"]
# vif = vif(x.copy(deep = True))
# vif = vif.sort_values(by='vif')
# data_list = vif.values.tolist()
# data_list.insert(0, vif.columns.tolist())


# # Configurar el estilo de seaborn
# sns.set_theme()
# # Crear el gráfico de barras horizontal con paleta invertida
# plt.figure(figsize=(20, 10))
# sns.barplot(data=vif, x='vif', y='var', palette='coolwarm')  # Invertimos la paleta
# plt.axvline(x=4, color='red', linestyle='--', label='x = 4')
# plt.title("INFLACIÓN DE VARIANZA (VIF)"
#           ,fontsize=30,color='#808080', loc='left', weight='bold', pad=20, y=1.05)
# plt.xlabel(f'VIF', fontsize=20, color='#808080')
# plt.ylabel(f'Variable explicativa', fontsize=20, color='#808080')
# plt.xticks(fontsize=20, color='#808080', rotation=0)
# plt.yticks(fontsize=20, color='#808080')
# #Fondo
# plt.gca().set_facecolor('#FFFFFF')
# # Mostrar el gráfico
# plt.savefig("Multicolinealidad.jpg")
# plt.show()



# # # Nuevo modelo
# # #-------------
# # vif_not = vif[vif["vif"]>=4]
# # vif = vif[vif["vif"]<4]
# # x_new = vif["var"].tolist()
# # x = x[x_new]


# # # Volver ajustar el modelo
# # x = sm.add_constant(x, prepend=True)
# # modelo = sm.OLS(endog=y, exog=x,)
# # modelo = modelo.fit()
# # print(modelo.summary())



# =============================================================================
# TEST DE NORMALIDAD 
# =============================================================================
# Obtener los residuos del modelo
residuals = modelo.resid
predictions = modelo.predict(x)

# Test de Hipótesis
#=======================
# Jarque-Bera
#------------------
from scipy import stats
jb_stat, jb_p_value = stats.jarque_bera(residuals)

# Imprimir los resultados
print("Estadístico de prueba:", jb_stat)
print("Valor p:", jb_p_value)

# Shapiro-Wilk test
#------------------
sw_stat, sw_p_value  = stats.shapiro(residuals)

print("Estadístico de prueba:", sw_stat)
print("Valor p:", sw_p_value)

# D'Agostino's K-squared test
#----------------------------
k2, p_value = stats.normaltest(residuals)
print(f"Estadítico= {k2}, p-value = {p_value}")


# Método Analítico
#=======================
# Calcular la asimetría y curtosis
skewness = pd.Series(residuals).skew()
kurtosis = pd.Series(residuals).kurtosis()

# Imprimir los resultados
print("Asimetría:", skewness)
print("Curtosis:", kurtosis)



# Representación gráfica
#=======================
# Forma 1: Histograma 
sns.set(style='whitegrid')
plt.figure(figsize=(20, 10))
# Crear un histograma utilizando Seaborn
import seaborn as sns
sns.histplot(
    data    = residuals,
    stat    = "density",
    kde     = True, #linea de distribución
    line_kws= {'linewidth': 1}, # ancho de la línea
    color   = "blue",
    alpha   = 0.1,
)

from scipy.stats import norm
# Calcular los parámetros de la distribución normal
mu, sigma = norm.fit(residuals)
# Generar valores para la curva normal teórica
x_hat = np.linspace(min(residuals), max(residuals), 100)
y_hat = norm.pdf(x_hat, mu, sigma)

# Agregar curva normal teórica al gráfico
plt.plot(x_hat, y_hat, 'k-', label='Curva normal teórica', color = "red")

# Mostrar la información de asimetría y curtosis en el gráfico
plt.title("Histograma de Residuos\nAsimetría: {:.2f}, Curtosis: {:.2f}".format(skewness, kurtosis)
          , fontsize=30,color='#808080', loc='left', weight='bold', pad=20, y=1)
plt.xlabel(f'', fontsize=20, color='#808080')
plt.ylabel(f'Density', fontsize=20, color='#808080')
plt.xticks(fontsize=20, color='#808080', rotation=0)
plt.yticks(fontsize=20, color='#808080')
#Fondo
plt.gca().set_facecolor('#FFFFFF')
# eliminar el borde del gráfico
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
# eliminar las líneas de fondo
plt.grid(False)

legend = plt.legend(loc='upper right', fancybox=True, shadow=True, ncol=3, fontsize=20)#, bbox_to_anchor=(0.5, 0))
legend.get_frame().set_facecolor('#FFFFFF')

# Mostrar el gráfico
plt.savefig("Normalidad_Histograma.jpg")
plt.show()




# Forma 2: Q-Q Plot
sns.set(style='whitegrid')
plt.figure(figsize=(30, 10))

sm.qqplot(
    residuals,
    fit   = True,
    line  = 'q',
    #ax    = axes[1, 1], 
    color = 'firebrick',
    alpha = 0.4,
    lw    = 2
)
plt.title("Q-Q Plot Residuals"
          , fontsize=16,color='#808080', loc='left', weight='bold', pad=20, y=1)

#Fondo
plt.gca().set_facecolor('#FFFFFF')
# eliminar el borde del gráfico
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
# eliminar las líneas de fondo
plt.grid(False)

# Mostrar el gráfico
plt.savefig("Normalidad_QQPlot.jpg")
plt.show()


# =============================================================================
# HOMOCEDASTICIDAD 
# =============================================================================
from statsmodels.compat import lzip
# Obtener los residuos del modelo
residuals = modelo.resid
# Obtener los residuos al cuadrado
residuals_squared = modelo.resid**2

predictions = modelo.predict(x)


# Realizar el test de Breusch-Pagan
name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
bp_test = sm.stats.diagnostic.het_breuschpagan(residuals_squared, x)
bp_results = lzip(name, bp_test)

# Imprimir los resultados del test
print("Breusch-Pagan Test:")
for result in bp_results:
    print(result[0], ":", result[1])

bp = round(result[1],3)

bp_results[1][1]

# Gráficamente
#-------------
# id
sns.set(style='whitegrid')
plt.figure(figsize=(20, 10))
plt.scatter(list(range(len(y))), residuals, edgecolors=(0, 0, 0), alpha=0.6)
plt.axhline(y=0, linestyle='--', color='red', lw=2)
plt.title("Residuos del modelo"
          ,fontsize=30,color='#808080', loc='left', weight='bold', pad=20, y=1.05)
plt.xlabel(f'id', fontsize=20, color='#808080')
plt.ylabel(f'Residuo', fontsize=20, color='#808080')
plt.xticks(fontsize=20, color='#808080', rotation=0)
plt.yticks(fontsize=20, color='#808080')
#Fondo
plt.gca().set_facecolor('#FFFFFF')
# eliminar el borde del gráfico
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
# eliminar las líneas de fondo
plt.grid(False)
# Mostrar el gráfico
plt.savefig("Homocedasticidad_id.jpg")
plt.show()

#Predicción
sns.set(style='whitegrid')
plt.figure(figsize=(20, 10))
plt.scatter(predictions, residuals, edgecolors=(0, 0, 0), alpha=0.5)
plt.axhline(y=0, linestyle='--', color='red', lw=2)
plt.title("Residuos del modelo vs predicción"
          ,fontsize=30,color='#808080', loc='left', weight='bold', pad=20, y=1.05)
plt.xlabel(f'Predicción', fontsize=20, color='#808080')
plt.ylabel(f'Residuo', fontsize=20, color='#808080')
plt.xticks(fontsize=20, color='#808080', rotation=0)
plt.yticks(fontsize=20, color='#808080')
#Fondo
plt.gca().set_facecolor('#FFFFFF')
# eliminar el borde del gráfico
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
# eliminar las líneas de fondo
plt.grid(False)
# Mostrar el gráfico
plt.savefig("Homocedasticidad_PrediccionResiduos.jpg")
plt.show()

# predicción vs real
sns.set(style='whitegrid')
plt.figure(figsize=(20, 10))
plt.scatter(y, predictions, edgecolors=(0, 0, 0), alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, color ='red')
plt.title("Valor predicho vs valor real"
          ,fontsize=30,color='#808080', loc='left', weight='bold', pad=20, y=1.05)
plt.xlabel(f'Real', fontsize=20, color='#808080')
plt.ylabel(f'Predicción', fontsize=20, color='#808080')
plt.xticks(fontsize=20, color='#808080', rotation=0)
plt.yticks(fontsize=20, color='#808080')
#Fondo
plt.gca().set_facecolor('#FFFFFF')
# eliminar el borde del gráfico
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
# eliminar las líneas de fondo
plt.grid(False)
# Mostrar el gráfico
plt.savefig("Homocedasticidad_PrediccionReal.jpg")
plt.show()





df_resultados = pd.DataFrame({'betas': modelo.params, 'pvalues': modelo.pvalues})
df_resultados.reset_index(inplace=True)

df_resultados.to_excel("Resultados.xlsx", index=False)

