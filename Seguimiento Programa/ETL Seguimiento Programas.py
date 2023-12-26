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
from datetime import datetime

# =============================================================================
# 2.FUNCIONES
# =============================================================================

def encabezados(data,fila_encabezado, fila_datos):
    nuevos_encabezados = data.iloc[fila_encabezado]
    data = data[fila_datos:]  
    data.columns = nuevos_encabezados
    data = data.reset_index(drop=True)
    return data
    
def etl(df, c, tipo2 , df_precios, tipo_base):
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
    
    # precios
    
    
    #venta bruta
    
    
    #if si es sell out
    # cálculo de cupon
    
    
    # if si es programa 
    
    df_v = df.iloc[:,c:].copy()
    fechas = df_v.iloc[2,:]
    fechas = [fecha.strftime("%d-%m-%Y") for fecha in fechas]
    df_v.iloc[2,:] = fechas
    columnas = df_v.apply(lambda x: x[1] + '_' + x[2], axis=0)
    columnas = columnas.tolist()
    df_v.iloc[0,:] = columnas
    df_v = encabezados(data = df_v,fila_encabezado = 0 ,fila_datos=3)
    
    df_resultado = pd.concat([df_c,df_v],axis=1)
    

    fechas_precio = df_precios.columns.tolist()
    fechas_precio.remove("IMS")
    fechas_precio = [elemento.replace('Precio_', '') for elemento in fechas_precio]


    df_resultado = pd.merge(df_resultado,df_precios, how = "left", on ="IMS")
    # fechas_precio.remove("IMS")
    # fechas_precio = [fecha.strftime('%d-%m-%Y') for fecha in fechas_precio]

    # Crear columnas resultantes
    for fecha in fechas_precio:
        columna_unidades = f'Unidades_{fecha}'
        columna_precio = f'Precio_{fecha}'
        columna_resultante = f'VB_{fecha}'

        # Multiplicar las columnas correspondientes
        df_resultado[columna_resultante] = df_resultado[columna_unidades] * df_resultado[columna_precio]
    
    if tipo_base == 'programa':
        return df_resultado
    elif tipo_base == 'sell out':
        return df_resultado



# =============================================================================
# 3.INPUT
# =============================================================================
# Espacio de trabajo
os.chdir("C:\\Users\\Daniel\\Desktop\\ABBOTT\\Proyectos Power BI\\Seguimiento Programa")
# DATOS
#******************************************************************************
df_precios = pd.read_excel("Precios.xlsx", sheet_name="Precios", header=1)
df_precios = df_precios.dropna(subset=['IMS'])
fechas_precio = df_precios.columns.tolist()
fechas_precio = [elemento for elemento in fechas_precio if isinstance(elemento, datetime) and elemento.year >= 2020]
fechas_precio = ["IMS"]+fechas_precio
df_precios = df_precios[fechas_precio]
df_precios["IMS"] = df_precios["IMS"].astype(int)
df_precios = pd.DataFrame(df_precios).set_index('IMS')
columnas_df_precios = df_precios.columns.tolist()
columnas_df_precios = ['Precio_' + fecha.strftime('%d-%m-%Y') for fecha in columnas_df_precios]
df_precios.columns = columnas_df_precios
df_precios.reset_index(inplace=True)

# DATOS
#******************************************************************************
ges = pd.read_excel("BASE.xlsx", sheet_name="GES")
ges = etl(df = ges, c = 17, tipo2 = "GES", df_precios= df_precios, tipo_base="programa")


rrss = pd.read_excel("BASE.xlsx", sheet_name="RRSS")
rrss = etl(df = rrss, c = 16, tipo2 = "RRSS", df_precios= df_precios, tipo_base="programa")

sm = pd.read_excel("BASE.xlsx", sheet_name="SM")
sm = etl(df = sm, c = 15, tipo2 = "SM", df_precios= df_precios, tipo_base="programa")

otros = pd.read_excel("BASE.xlsx", sheet_name="OTROS")
otros = etl(df = otros, c = 18, tipo2 = "OTROS", df_precios= df_precios, tipo_base="programa")






df_unidades = sm.copy()

df_unidades = pd.merge(df_unidades,df_precios, how = "left", on ="IMS")
fechas_precio.remove("IMS")
fechas_precio = [fecha.strftime('%d-%m-%Y') for fecha in fechas_precio]

# Crear columnas resultantes
for fecha in fechas_precio:
    columna_unidades = f'Unidades_{fecha}'
    columna_precio = f'Precio_{fecha}'
    columna_resultante = f'VB_{fecha}'

    # Multiplicar las columnas correspondientes
    df_unidades[columna_resultante] = df_unidades[columna_unidades] * df_unidades[columna_precio]







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




