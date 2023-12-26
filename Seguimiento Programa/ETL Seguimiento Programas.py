# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:41:04 2023
@author: Daniel Silva Momberg
Title  : ETL Seguimiento Programas
"""

# =============================================================================
# 1. LIBRERÍAS
# =============================================================================
import os
import pandas as pd
import pyodbc
from unidecode import unidecode


# =============================================================================
# 2.FUNCIONES
# =============================================================================

def encabezados(data,fila_encabezado, fila_datos):
    nuevos_encabezados = data.iloc[fila_encabezado]
    data = data[fila_datos:]  
    data.columns = nuevos_encabezados
    data = data.reset_index(drop=True)
    return data
    
def etl(df, c, tipo2 ):
    columnas_base = ["TIPO","CADENA","TIPO II","IMS","MARCA","PRESENTACION"]
    df_c = df.iloc[:,0:c].copy()
    df_c = encabezados(data= df_c,fila_encabezado=2,fila_datos=3)
    df_c.columns = [unidecode(column).upper() for column in df_c.columns]
    if tipo2 == 'GES':
        df_c.rename(columns={"ISAPRE":"TIPO II"}, inplace=True)
    elif tipo2 == 'RRSS' or tipo2 == 'SM':
        df_c["TIPO II"] = df_c["TIPO"]
    elif tipo2 == 'OTROS':
        df_c.rename(columns={"CLIENTE":"CADENA","PROGRAMA":"TIPO II"}, inplace=True)
    df_c = df_c[columnas_base]
    
    df_v = df.iloc[:,c:].copy()
    fechas = df_v.iloc[2,:]
    fechas = [fecha.strftime("%d-%m-%Y") for fecha in fechas]
    df_v.iloc[2,:] = fechas
    columnas = df_v.apply(lambda x: x[1] + '_' + x[2], axis=0)
    columnas = columnas.tolist()
    df_v.iloc[0,:] = columnas
    df_v = encabezados(data = df_v,fila_encabezado = 0 ,fila_datos=3)
    
    df_resultado = pd.concat([df_c,df_v],axis=1)
    return df_resultado

# =============================================================================
# 3.INPUT
# =============================================================================
# Espacio de trabajo
os.chdir("C:\\Users\\Daniel\\Desktop\\ABBOTT\\Proyectos Power BI\\Seguimiento Programa")

ges = pd.read_excel("BASE.xlsx", sheet_name="GES")
ges = etl(df = ges, c = 17, tipo2 = "GES")

rrss = pd.read_excel("BASE.xlsx", sheet_name="RRSS")
rrss = etl(df = rrss, c = 16, tipo2 = "RRSS")

sm = pd.read_excel("BASE.xlsx", sheet_name="SM")
sm = etl(df = sm, c = 15, tipo2 = "SM")

otros = pd.read_excel("BASE.xlsx", sheet_name="OTROS")
otros = etl(df = otros, c = 18, tipo2 = "OTROS")



# =============================================================================
# 4.OUTPUT
# =============================================================================

# 1: Base en formato wide
df_wide = pd.concat([ges,rrss, sm, otros], axis=0)

# 2: Base en formato long
df_long = df_wide.melt(id_vars=df_wide.iloc[:,0:6].columns.tolist(), 
                        value_vars=df_wide.iloc[:,6:].columns.tolist(), 
                        var_name='Columna', value_name='Valor')
df_long[["U_Medida", 'Fecha']] = df_long['Columna'].str.split('_', expand=True)
df_long.drop('Columna', axis=1, inplace=True)
df_long["Fecha"] = pd.to_datetime(df_long["Fecha"],format="%d-%m-%Y", dayfirst = True)




