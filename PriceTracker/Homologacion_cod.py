# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:38:38 2023
@author: Daniel Silva Momberg
Título : Homologación de Códigos
"""
# =============================================================================
# 1. LIBRERÍAS Y FUNCIONES
# =============================================================================
import os
import pandas as pd
import pyodbc
from fuzzywuzzy import fuzz


# Modificar la función match_medicamentos para devolver True si la similitud es mayor al umbral, de lo contrario False
def match_medicamentos(name1, name2, threshold=90):
    similarity = fuzz.partial_ratio(name1, name2)
    return similarity >= threshold

# Modificar la función get_matching_id para usar el umbral al obtener coincidencias
def get_matching_id(name1, df2, threshold=90):
    # Filtrar solo las filas donde la similitud es mayor al umbral
    valid_matches = df2[df2['PRODUCTO'].apply(lambda name2: match_medicamentos(name1, name2, threshold))]
    
    # Si hay al menos una coincidencia válida, devuelve el ID correspondiente de la mejor coincidencia
    if not valid_matches.empty:
        best_match_index = valid_matches['PRODUCTO'].apply(lambda name2: match_medicamentos(name1, name2, threshold)).idxmax()
        return df2.loc[best_match_index, 'PRODUCTO']
    else:
        return None  # Si no hay coincidencias válidas, devuelve None o un valor que tenga sentido en tu caso

# =============================================================================
# 2. DATOS
# =============================================================================

#1. ESPACIO DE TRABAJO
#********************************
os.chdir("C:\\Users\\Daniel\\Desktop\ABBOTT\\Proyectos Power BI\\PriceTracker")

# MERCADO RELEVANTE - USO INTERNO
#******************************************************************************
#query_mr = "select * from P_MERCADO_RELEVANTE_FALLAS"
#mr = pd.read_sql(query_mr, con)
mr = pd.read_excel("MercadoRelevante_IQVIA.xlsx")
mr = mr[["Cód_IMS","Presentación"]].drop_duplicates()
mr.rename(columns = {"Cód_IMS":"IMS","Presentación":"PRODUCTO"},inplace=True)
mr["PRODUCTO"] = mr["PRODUCTO"].str.upper()


# SOCOFAR
#******************************************************************************
scf = pd.read_excel("Precios_SOCOFAR.xlsx", sheet_name="Homologación")
scf = scf[["CÓDIGO SOCOFAR","DESCRIPTOR"]].drop_duplicates()
scf.rename(columns = {"CÓDIGO SOCOFAR":"COD","DESCRIPTOR":"PRODUCTO"},inplace=True)
scf["PRODUCTO"] = scf["PRODUCTO"].str.upper()
scf['mr'] = scf['PRODUCTO'].apply(lambda x: get_matching_id(x, mr, threshold=60))
scf.to_excel("scf_token_partial_ratio_2.xlsx")

# mr['scf'] = mr['PRODUCTO'].apply(lambda x: get_matching_id(x, scf))
# mr.to_excel("scf_token_partial_ratio.xlsx")


# GLOBAL PHARMA
#******************************************************************************
gph = pd.read_excel("Precios_GPH.xlsx")
gph = gph[["Código Barra","Producto 04  - 09"]].drop_duplicates()
gph.rename(columns = {"Código Barra":"COD","Producto 04  - 09":"PRODUCTO"},inplace=True)
gph["PRODUCTO"] = scf["PRODUCTO"].str.upper()
gph.dropna(inplace=True)
gph['mr'] = gph['PRODUCTO'].apply(lambda x: get_matching_id(x, mr, threshold=60))
gph.to_excel("gph_token_partial_ratio_2.xlsx")


gph2 = pd.read_excel("Precios_GPH.xlsx")
gph2 = gph2[["Código Barra","Producto 04  - 09"]].drop_duplicates()
gph2.rename(columns = {"Código Barra":"COD","Producto 04  - 09":"PRODUCTO"},inplace=True)
gph2["PRODUCTO"] = gph2["PRODUCTO"].str.upper()
gph2.dropna(inplace=True)
gph2['mr'] = gph2['PRODUCTO'].apply(lambda x: get_matching_id(x, mr, threshold=60))
gph2.to_excel("gph_token_partial_ratio_2.xlsx")























#%%
import pandas as pd
from fuzzywuzzy import fuzz

# Crear el primer DataFrame
data1 = {
    'ID': [1, 2, 3, 4, 5,6],
    'NombreMedicamento1': ['Aspirina 20', 'Paracetamol', 'Ibuprofeno', 'Amoxicilina', 'Omeprazol',"Movidol Caps L.P. 200 Mg X 30"]
}

df1 = pd.DataFrame(data1)

# Crear el segundo DataFrame
data2 = {
    'ID': [6, 7, 8, 9, 10,11],
    'NombreMedicamento2': ['Aspirin 2.0', 'Paracetamol', 'Ibuprofen', 'Amoxycillin', 'Omeprazole',"PROGENDO 200 MG X 30 CAPS"]
}

df2 = pd.DataFrame(data2)

# Realizar la coincidencia de columnas utilizando el algoritmo de Levenshtein
def match_medicamentos(name1, name2):
    return fuzz.partial_ratio(name1, name2)

# Función para obtener el ID correspondiente en df2
def get_matching_id(name1, df2):
    similarities = df2['NombreMedicamento2'].apply(lambda name2: match_medicamentos(name1, name2))
    best_match_index = similarities.idxmax()
    return df2.loc[best_match_index, 'ID']

df1['MejorCoincidencia'] = df1['NombreMedicamento1'].apply(lambda x: get_matching_id(x, df2))

# Imprimir el resultado
print(df1)



















=======
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:38:38 2023

@author: silvadx30
"""
# =============================================================================
# 1. LIBRERÍAS Y FUNCIONES
# =============================================================================
import os
import pandas as pd
import pyodbc

# Realizar la coincidencia de columnas utilizando el algoritmo de Levenshtein
def match_medicamentos(name1, name2):
    return fuzz.ratio(name1, name2)

# Función para obtener el ID correspondiente en df2
def get_matching_id(name1, df2):
    similarities = df2['PRODUCTO'].apply(lambda name2: match_medicamentos(name1, name2))
    best_match_index = similarities.idxmax()
    return df2.loc[best_match_index, 'PRODUCTO']

con = pyodbc.connect('Driver={SQL Server};'
                     'Server=WQ00408D;'
                     'Database=Programas_Fidelizacion;'
                     'Trusted_Connection=yes;')


# =============================================================================
# 2. DATOS
# =============================================================================

#1. ESPACIO DE TRABAJO
#********************************
os.chdir("C:\\Users\\silvadx30\\OneDrive - Abbott\\Proyectos Power BI\\PriceTraker\\Informe\\precios septiembre\\precios septiembre")

# MERCADO RELEVANTE - USO INTERNO
#********************************
#query_mr = "select * from P_MERCADO_RELEVANTE_FALLAS"
#mr = pd.read_sql(query_mr, con)
mr = pd.read_excel("C:\\Users\\silvadx30\\OneDrive - Abbott\\Proyectos Power BI\\PriceTraker\\Informe\\MercadoRelevante_IQVIA.xlsx")
mr = mr[["Cód_IMS","Presentación"]].drop_duplicates()
mr.rename(columns = {"Cód_IMS":"IMS","Presentación":"PRODUCTO"},inplace=True)
mr["PRODUCTO"] = mr["PRODUCTO"].str.upper()


# FARMA 7
#********************************
f7 = pd.read_excel("F7.xlsx")
f7 = f7[["BARCODE","DESCRIPCION"]].drop_duplicates()
f7.rename(columns = {"BARCODE":"COD","DESCRIPCION":"PRODUCTO"},inplace=True)
f7["PRODUCTO"] = f7["PRODUCTO"].str.upper()


mr['FARM7'] = mr['PRODUCTO'].apply(lambda x: get_matching_id(x, f7))




















#%%
import pandas as pd
from fuzzywuzzy import fuzz

# Crear el primer DataFrame
data1 = {
    'ID': [1, 2, 3, 4, 5,6],
    'NombreMedicamento1': ['Aspirina 20', 'Paracetamol', 'Ibuprofeno', 'Amoxicilina', 'Omeprazol',"Movidol Caps L.P. 200 Mg X 30"]
}

df1 = pd.DataFrame(data1)

# Crear el segundo DataFrame
data2 = {
    'ID': [6, 7, 8, 9, 10,11],
    'NombreMedicamento2': ['Aspirin 2.0', 'Paracetamol', 'Ibuprofen', 'Amoxycillin', 'Omeprazole',"PROGENDO 200 MG X 30 CAPS"]
}

df2 = pd.DataFrame(data2)

# Realizar la coincidencia de columnas utilizando el algoritmo de Levenshtein
def match_medicamentos(name1, name2):
    return fuzz.ratio(name1, name2)

# Función para obtener el ID correspondiente en df2
def get_matching_id(name1, df2):
    similarities = df2['NombreMedicamento2'].apply(lambda name2: match_medicamentos(name1, name2))
    best_match_index = similarities.idxmax()
    return df2.loc[best_match_index, 'ID']

df1['MejorCoincidencia'] = df1['NombreMedicamento1'].apply(lambda x: get_matching_id(x, df2))

# Imprimir el resultado
print(df1)



