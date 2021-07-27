# -*- coding: utf-8 -*-
# %% [markdown]
"""
# Colombia Data
"""

# %%
from IPython import get_ipython

# %%
import pandas as pd

# from sodapy import Socrata


# def download_dataset(domain, dataset_id):
#     # for this exercise, we're not using an app token,
#     # but you *should* sign-up and register for an app_token if you want to use the Socrata API
#     client = Socrata(domain, app_token=None)
#     offset = None
#     data = []
#     batch_size = 1000

#     while True:
#         records = client.get(dataset_id, offset=offset, limit=batch_size)
#         data.extend(records)
#         if len(records) < batch_size:
#             break
#         offset = offset + batch_size if (offset) else batch_size

#     return pd.DataFrame.from_dict(data)


# def download_covid_dataset():
#     return (
#         col_df
#         if "col_df" in globals()
#         else download_dataset("datos.gov.co", "gt2j-8ykr")
#     )


# # Descargar datos de COVID-19 de Colombia
# col_df = download_covid_dataset()


# %%
col_df = pd.read_csv(
    "https://www.datos.gov.co/api/views/gt2j-8ykr/rows.csv?accessType=DOWNLOAD"
)


# %%
col_df


# %%
print(col_df.shape)
col_df.tail(10)


# %%
col_df.info()


# %%
print(col_df.columns)


# %%
print(col_df["Nombre departamento"].unique())


# %%
# [i.title() for i in col_df["Nombre departamento"].unique()]


# %%
dptos_col = {
    "BOGOTA": "Bogotá D.C.",
    "VALLE": "Valle del Cauca",
    "ANTIOQUIA": "Antioquia",
    "CARTAGENA": "Bolívar",
    "HUILA": "Huila",
    "META": "Meta",
    "RISARALDA": "Risaralda",
    "NORTE SANTANDER": "Norte de Santander",
    "CALDAS": "Caldas",
    "SANTANDER": "Santander",
    "CUNDINAMARCA": "Cundinamarca",
    "TOLIMA": "Tolima",
    "BARRANQUILLA": "Atlántico",
    "QUINDIO": "Quindío",
    "CAUCA": "Cauca",
    "STA MARTA D.E.": "Magdalena",
    "CESAR": "Cesar",
    "SAN ANDRES": "San Andrés y Providencia",
    "CASANARE": "Casanare",
    "NARIÑO": "Nariño",
    "ATLANTICO": "Atlántico",
    "BOYACA": "Boyacá",
    "CORDOBA": "Córdoba",
    "BOLIVAR": "Bolívar",
    "SUCRE": "Sucre",
    "MAGDALENA": "Magdalena",
    "GUAJIRA": "La Guajira",
    "CHOCO": "Chocó",
    "AMAZONAS": "Amazonas",
    "CAQUETA": "Caquetá",
    "PUTUMAYO": "Putumayo",
    "ARAUCA": "Arauca",
    "VAUPES": "Vaupés",
    "GUAINIA": "Guainía",
    "VICHADA": "Vichada",
    "GUAVIARE": "Guaviare",
}


# %%
# Aparentemente hay un caso donde no hay departamento registrado, y por ende,
# tenemos que borrar ese registro
# https://stackoverflow.com/questions/26535563/querying-for-nan-and-other-names-in-pandas
col_df = col_df[col_df["Nombre departamento"].notna()]


# %%
col_df["departamentos"] = col_df["Nombre departamento"].map(dptos_col)


# %%
# col_df["departamentos"] = col_df.apply(
#     lambda x: clean(x["Nombre departamento"]), axis=1
# )


# %%
dpto = list(col_df["departamentos"].unique())
print(dpto)


# %%
col_df.loc[(col_df.departamentos == "Bolívar"), "fecha reporte web"].value_counts(
    dropna=False
).sort_index().cumsum(skipna=False)


# %%
# def fecha_dañada(fecha):
#     """Función temporal para arreglar error de fec"""
#     fecha = fecha.replace("19/15/2020", "2020-05-19T00:00:00.000")
#     return fecha


# col_df["fecha_diagnostico"] = col_df.apply(
#     lambda x: fecha_dañada(x["fecha_diagnostico"]), axis=1
# )


# %%
fechas = list(col_df["fecha reporte web"].unique())
print(fechas)


# %%
col_df


# %%
col_df.info()


# %%
# col_df["fecha_diagnostico"] = col_df["fecha reporte web"].apply(pd.to_datetime)


# %%
# col_df["fecha reporte web"].sample(5)


# %%
# col_df["fecha reporte web"].sample(5).apply(pd.to_datetime)


# %%
col_df["fecha_diagnostico"] = pd.to_datetime(
    col_df["fecha reporte web"], dayfirst=True, infer_datetime_format=True
)


# %%
fechas = list(col_df["fecha_diagnostico"].unique())


# %%
colombia_a = pd.DataFrame()

for d in dpto:
    #     conteo = col_df.loc[(col_df.departamentos == d), "fecha reporte web"].value_counts()
    conteo = col_df.loc[(col_df.departamentos == d), "fecha_diagnostico"].value_counts()
    conteo = conteo.sort_index(ascending=True)
    conteo = conteo.cumsum()

    colombia_a = colombia_a.append(conteo)


# %%
colombia_a = colombia_a.T


# %%
colombia_a.columns = dpto


# From fillna method descriptions:
#
# ```python
# method : {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}, default None Method to use for filling holes in reindexed Series pad / ffill:
#
# # propagate last valid observation forward to next valid backfill / bfill: use NEXT valid observation to fill gap
#
# df_new.Inventory = df_new.Inventory.fillna(method="ffill")
# ```

# %%
colombia_a = colombia_a.sort_index(axis=0)
colombia = colombia_a.fillna(method="ffill")


# %%
colombia.info()


# %%
colombia["fecha"] = colombia.index


# %%
colombia.set_index("fecha", inplace=True)


# %%
colombia


# %%
col_df.departamentos.unique()


# %%
divipola = {
    "Amazonas": 91,
    "Antioquia": 5,
    "Arauca": 81,
    "San Andrés y Providencia": 88,
    "Atlántico": 8,
    "Bogotá D.C.": 11,
    "Bolívar": 13,
    "Boyacá": 15,
    "Caldas": 17,
    "Caquetá": 18,
    "Casanare": 85,
    "Cauca": 19,
    "Cesar": 20,
    "Chocó": 27,
    "Cundinamarca": 25,
    "Córdoba": 23,
    "Guainía": 94,
    "Guaviare": 95,
    "Huila": 41,
    "La Guajira": 44,
    "Magdalena": 47,
    "Meta": 50,
    "Nariño": 52,
    "Norte de Santander": 54,
    "Putumayo": 86,
    "Quindío": 63,
    "Risaralda": 66,
    "Santander": 68,
    "Sucre": 70,
    "Tolima": 73,
    "Valle del Cauca": 76,
    "Vaupés": 97,
    "Vichada": 99,
}


# %%
x = colombia
y = x.reset_index()
y
z = pd.melt(y, id_vars=["fecha"], value_vars=dpto)
z.columns = ["fecha", "departamento", "casos"]
z = z.sort_values(by=["fecha", "casos"])
z = z.dropna()
z["dp"] = z["departamento"].map(divipola)
cronologia = z


# %%
cronologia


# %%
cronologia.to_csv("datos/cronologia.csv", index=False)


# %%
col_df["fecha_diagnostico"].value_counts(dropna=False).head(10)


# %%
col_df = col_df[["fecha_diagnostico"]].reset_index(drop=True)
col_df.tail(10)


# %%
import datetime

fixed_dates_df = col_df.copy()
fixed_dates_df["fecha_diagnostico"] = fixed_dates_df["fecha_diagnostico"].apply(
    pd.to_datetime
)
fixed_dates_df = fixed_dates_df.set_index(fixed_dates_df["fecha_diagnostico"])
grouped = fixed_dates_df.resample("D").count()
data_df = pd.DataFrame({"count": grouped.values.flatten()}, index=grouped.index)
data_df.tail(10)


# %%
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
plt.style.use("ggplot")

data_df.plot(color="purple")


# %%
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(data_df)
fig = result.plot()
fig.tight_layout()


# %%
data_df.info()


# %%
from fbprophet import Prophet

model = Prophet(daily_seasonality=True)
train_df = data_df.rename(columns={"count": "y"})
train_df["ds"] = train_df.index
model.fit(train_df)


# %%
pd.plotting.register_matplotlib_converters()
future = model.make_future_dataframe(12, freq="D", include_history=True)
forecast = model.predict(future)
model.plot(forecast)
