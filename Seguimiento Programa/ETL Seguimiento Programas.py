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
    columnas_base = ["TIPO","CADENA","TIPO II","DIVISION","LINEA","IMS","MARCA","PRESENTACION"]
    df_c = df.iloc[:,0:c].copy()
    
    # c=16
    # df_c = sd.iloc[:,0:c].copy()
    # tipo2= "SD"
    # tipo_base = "sell out"
    # df_v = sd.iloc[:,c:].copy()
    
    df_c = encabezados(data= df_c,fila_encabezado=2,fila_datos=3)
    df_c.columns = [unidecode(column).upper() for column in df_c.columns]
    if tipo2 == 'GES':
        df_c.rename(columns={"ISAPRE":"TIPO II"}, inplace=True)
    elif tipo2 == 'RRSS' or tipo2 == 'SM':
        df_c["TIPO II"] = df_c["TIPO"]
    elif tipo2 == 'OTROS':
        df_c.rename(columns={"CLIENTE":"CADENA","PROGRAMA":"TIPO II"}, inplace=True)
    elif tipo2 == 'SD' or tipo2 == 'SI':
        df_c["TIPO II"] = df_c["TIPO"]
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
        
    
    
    
    fechas_precio = df_precios.columns.tolist()
    fechas_precio.remove("IMS")
    fechas_precio = [elemento.replace('Precio_', '') for elemento in fechas_precio]


    df_resultado = pd.merge(df_resultado,df_precios, how = "left", on ="IMS")
    #df_resultado["Unidades_01-10-2023"].sum()
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
        for fecha in fechas_precio:
            columna_unidades = f'Unidades_{fecha}'
            columna_precio = f'Precio_{fecha}'
            columna_resultante = f'Cupón_{fecha}'

            # Multiplicar las columnas correspondientes
            df_resultado[columna_resultante] = -0.03 * df_resultado[columna_unidades] * df_resultado[columna_precio]
        return df_resultado



# =============================================================================
# 3.INPUT
# =============================================================================
# Espacio de trabajo
os.chdir("C:\\Users\\Daniel\\Desktop\\ABBOTT\\Proyectos Power BI\\Seguimiento Programa")
# DATOS
#******************************************************************************
df_precios = pd.read_excel("Precios.xlsx", sheet_name="EXLAB")#, header=1)
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

sd = pd.read_excel("BASE.xlsx", sheet_name="Sell out Directo")
sd = etl(df = sd, c = 16, tipo2 = "SD", df_precios= df_precios, tipo_base="sell out")

si = pd.read_excel("BASE.xlsx", sheet_name="Sell out Indirecto")
si = etl(df = si, c = 16, tipo2 = "SI", df_precios= df_precios, tipo_base="sell out")


# =============================================================================
# 4.OUTPUT
# =============================================================================
maestro_programas = pd.read_excel("Maestro_Programas.xlsx")
maestro_programas = maestro_programas[['TIPO I', 'Tipo II']].drop_duplicates()
maestro_programas.dropna(inplace =True)
maestro_programas.rename(columns={"Tipo II":"TIPO II"},inplace=True)


# 1: Base en formato wide
df_wide = pd.concat([ges,rrss, sm, otros, sd, si], axis=0)
columnas_a_convertir = df_wide.iloc[:,8:].columns.tolist()
for columna in columnas_a_convertir:
    df_wide[columna] = pd.to_numeric(df_wide[columna], errors='coerce')  # 'coerce' para convertir no numéricos a NaN

cadena_lista = ["CRUZ_VERDE", "SALCO", "FASA"]
df_wide['CANAL'] = df_wide['CADENA'].apply(lambda x: 'DIRECTO' if x in cadena_lista else 'INDIRECTO')



df_wide = pd.merge(df_wide,maestro_programas, how = "left", on = "TIPO II")
micro = df_wide[(df_wide["TIPO"]=="OTROS")&(df_wide["TIPO I"].isnull())]["TIPO II"].unique().tolist()
def reemplazar_tipo(row):
    if row['TIPO II'] in micro:
        return "Spot Micro"
    else:
        return row['TIPO I']

df_wide['TIPO I'] = df_wide.apply(reemplazar_tipo, axis=1)
df_wide['TIPO I'] = df_wide['TIPO I'].fillna(df_wide['TIPO'])
orden_columnas = ["TIPO","TIPO I","TIPO II","CANAL","CADENA","DIVISION","LINEA","IMS","MARCA","PRESENTACION"] + columnas_a_convertir
df_wide = df_wide[orden_columnas]



df_wide["LINEA"] = df_wide["LINEA"].replace('Gastro', 'Gastroenterology')
df_wide["LINEA"] = df_wide["LINEA"].replace('Cardio', 'Cardiovascular')
df_wide["LINEA"] = df_wide["LINEA"].replace('Pain', 'Trauma & Pain')

df_wide["DIVISION"] = df_wide["DIVISION"].replace("Urology", "Women's Health & Urology")

df_wide[df_wide["DIVISION"]=="MARKET ACCESS"]["LINEA"].unique()
df_wide.loc[df_wide['MARCA'] == 'KLARICID', 'DIVISION'] = 'Respiratory & Dermatology'
df_wide.loc[df_wide['DIVISION'] == 'Market Access', 'LINEA'] = "Market Access"

df_wide.to_csv("df_wide.csv",index=False, encoding='utf-8-sig', decimal = ",", sep = ";")


# 2: Base en formato long
df_long = df_wide.melt(id_vars=df_wide.iloc[:,0:10].columns.tolist(), 
                        value_vars=df_wide.iloc[:,10:].columns.tolist(), 
                        var_name='Columna', value_name='Valor')
df_long[["U_Medida", 'Fecha']] = df_long['Columna'].str.split('_', expand=True)
df_long.drop('Columna', axis=1, inplace=True)
df_long["Fecha"] = pd.to_datetime(df_long["Fecha"],format="%d-%m-%Y", dayfirst = True)
df_long["TIPO"] = df_long["TIPO"].astype(str)
df_long["CADENA"] = df_long["CADENA"].astype(str)
df_long["TIPO II"] = df_long["TIPO II"].astype(str)
df_long["DIVISION"] = df_long["DIVISION"].astype(str)
df_long["LINEA"] = df_long["LINEA"].astype(str)
df_long["IMS"].fillna(0, inplace=True)
df_long["IMS"] = df_long["IMS"].astype(int)
df_long["MARCA"] = df_long["MARCA"].astype(str)
df_long["PRESENTACION"] = df_long["PRESENTACION"].astype(str)
df_long["Valor"] = df_long["Valor"].astype(float)
df_long["U_Medida"] = df_long["U_Medida"].astype(str) 
df_long.dtypes
df_long.to_csv("df_long.csv",index=False, encoding='utf-8-sig', decimal = ",", sep = ";")


df_long = df_long[df_long["Fecha"]!="2023-12-01"]



