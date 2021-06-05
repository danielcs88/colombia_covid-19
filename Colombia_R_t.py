# %% [markdown]
"""
# Análisis de 🇨🇴 Colombia a nivel departamental | $R_t$

por Daniel Cárdenas

Si $R_t > 1$, el número de casos aumentará, como al comienzo de una epidemia.
Cuando $R_t = 1$, la enfermedad es endémica, y cuando $R_t <1$ habrá una
disminución en el número de casos.

Entonces, los epidemiólogos usan $R_t$ para hacer recomendaciones de políticas.
Es por eso que este número es tan importante.

## Notas

Por cuestiones de simplicidad, el único distrito especial es en el caso de
Bogotá. En la página nacional, hay varias ciudades que son su propio distrito
(Barranquilla, Cartagena, Santa Marta y Buenaventura) fueron incluidas en su
departamento respectivo (Atlántico, Bolívar, Magdalena y Valle del Cauca).

Mi modelo es una adaptación del model de [Kevin
Systrom](https://github.com/k-sys/covid-19)

## Fuente de Datos

Mi fuente de datos es del Ministerio de Salud de Colombia y su plataforma [Casos
positivos de COVID-19 en
Colombia](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia/gt2j-8ykr/data).
"""

# %%
from IPython import get_ipython

# %%
# from IPython.display import clear_output
# ! ipython Colombia_Data.ipynb
# clear_output(wait=False)


# %%
import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import clear_output
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.dates import date2num, num2date
from matplotlib.patches import Patch
from scipy import stats as sps
from scipy.interpolate import interp1d
from IPython.display import display

get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic("matplotlib", "inline")


# %%
from cycler import cycler

custom = cycler(
    "color",
    [
        "#B3220F",
        "#F16E53",
        "#FFC475",
        "#006F98",
        "#1ABBEF",
        "#7FD2FD",
        "#153D53",
        "#0F9197",
    ],
)


plt.rc("axes", prop_cycle=custom)
plt.rcParams["figure.dpi"] = 140


# %%
def highest_density_interval(pmf, p=0.9):
    # If we pass a DataFrame, just call this recursively on the columns
    if isinstance(pmf, pd.DataFrame):
        return pd.DataFrame(
            [highest_density_interval(pmf[col], p=p) for col in pmf], index=pmf.columns
        )

    cumsum = np.cumsum(pmf.values)
    best = None
    for i, value in enumerate(cumsum):
        for j, high_value in enumerate(cumsum[i + 1 :]):
            if (high_value - value > p) and (not best or j < best[1] - best[0]):
                best = (i, i + j + 1)
                break

    low = pmf.index[best[0]]
    high = pmf.index[best[1]]
    return pd.Series([low, high], index=[f"Low_{p*100:.0f}", f"High_{p*100:.0f}"])


# %%
cronologia = "datos/cronologia.csv"

colombia = pd.read_csv(cronologia)

# Print Counties
latest_date = colombia[-1:]
latest_date = latest_date.fecha
latest_date = " ".join(str(elem) for elem in latest_date)
print(latest_date)


# %%
departamentos = sorted(set(colombia.departamento.unique()))
len(departamentos)


# %% [markdown]
"""
### Filtros

* Departamento seleccionado
* Eliminar departamentos listados como "Desconocidos"
* Eliminar filas con menos de 10 casos `filtro_departamento`
* Eliminar departamentos con menos filas que `filtro_departamento_fila` después
  de suavizar
"""

# %%
filtro_departamento = 10
filtro_departamento_fila = 10


# %%
colombia = colombia[colombia.casos >= filtro_departamento_fila].copy()
colombia.shape


# %%
colombia.tail()
print(len(colombia))


# %%
colombia = colombia[["fecha", "departamento", "casos"]].copy()
colombia["fecha"] = pd.to_datetime(colombia["fecha"])
colombia = colombia.set_index(["departamento", "fecha"]).squeeze().sort_index()


# %%
colombia


# %%
colombia_g = (
    colombia.groupby(["departamento"])
    .count()
    .reset_index()
    .rename({"casos": "filas"}, axis=1)
)
colombia_g


# %%
lista_dpto = colombia_g[colombia_g.filas >= filtro_departamento_fila][
    "departamento"
].tolist()
print(lista_dpto)


# %%
w = widgets.Dropdown(
    options=lista_dpto,
    description="Escoja un departamento:",
    value="Bogotá D.C.",
    disabled=False,
)
display(w)


# %%
seleccion = w.value
d = w.index


# %%
dpto = lista_dpto[d]


def prepare_cases(casos, cutoff=1):
    new_cases = casos.diff()

    smoothed = (
        new_cases.rolling(7, win_type="gaussian", min_periods=1, center=True)
        .mean(std=3)
        .round()
    )

    idx_start = np.searchsorted(smoothed, cutoff)

    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]

    return original, smoothed


casos = colombia.xs(dpto).rename(f"Casos en {dpto}")

original, smoothed = prepare_cases(casos)

original.plot(
    title=f"{dpto} | Casos nuevos {latest_date}",
    c="k",
    linestyle=":",
    alpha=0.5,
    label="Actual",
    legend=True,
    figsize=(500 / 72, 300 / 72),
)

ax = smoothed.plot(label="Promedio semanal (7 días) ", legend=True)

ax.get_figure().set_facecolor("w")
plt.savefig("gráficos/bta_casos.svg", dpi=300)


# %%
# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316

GAMMA = 1 / 7

# We create an array for every possible value of Rt
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX * 100 + 1)


def get_posteriors(sr, sigma=0.15):

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data=sps.poisson.pmf(sr[1:].values, lam), index=r_t_range, columns=sr.index[1:]
    )

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range, scale=sigma).pdf(r_t_range[:, None])

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)

    # (4) Calculate the initial prior
    prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range, columns=sr.index, data={sr.index[0]: prior0}
    )

    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        # (5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]

        # (5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior

        # (5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)

        # Execute full Bayes' Rule
        posteriors[current_day] = numerator / denominator

        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator + 1)

    return posteriors, log_likelihood


# Note that we're fixing sigma to a value just for the example
posteriors, log_likelihood = get_posteriors(smoothed, sigma=0.25)


# %%
ax = posteriors.plot(
    title=f"{dpto} \n Posteriores diarios de $R_t$ \n {latest_date}",
    legend=False,
    lw=1,
    c="k",
    alpha=0.3,
    xlim=(0.4, 6),
)

ax.set_xlabel("$R_t$")


# %%
# Note that this takes a while to execute - it's not the most efficient algorithm
hdis = highest_density_interval(posteriors, p=0.9)

most_likely = posteriors.idxmax().rename("ML")

# Look into why you shift -1
result = pd.concat([most_likely, hdis], axis=1)

result.tail()


# %%
def plot_rt(result, ax, county_name):

    ax.set_title(f"{dpto}")

    # Colors
    ABOVE = [1, 0, 0]
    MIDDLE = [1, 1, 1]
    BELOW = [0, 0, 0]
    cmap = ListedColormap(
        np.r_[np.linspace(BELOW, MIDDLE, 25), np.linspace(MIDDLE, ABOVE, 25)]
    )
    color_mapped = lambda y: np.clip(y, 0.5, 1.5) - 0.5

    index = result["ML"].index.get_level_values("fecha")
    values = result["ML"].values

    # Plot dots and line
    ax.plot(index, values, c="k", zorder=1, alpha=0.25)
    ax.scatter(
        index,
        values,
        s=40,
        lw=0.5,
        c=cmap(color_mapped(values)),
        edgecolors="k",
        zorder=2,
    )

    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(
        date2num(index),
        result["Low_90"].values,
        bounds_error=False,
        fill_value="extrapolate",
    )

    highfn = interp1d(
        date2num(index),
        result["High_90"].values,
        bounds_error=False,
        fill_value="extrapolate",
    )

    extended = pd.date_range(
        start=pd.Timestamp("2020-03-01"), end=index[-1] + pd.Timedelta(days=1)
    )

    ax.fill_between(
        extended,
        lowfn(date2num(extended)),
        highfn(date2num(extended)),
        color="k",
        alpha=0.1,
        lw=0,
        zorder=3,
    )

    ax.axhline(1.0, c="k", lw=1, label="$R_t=1.0$", alpha=0.25)

    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(0)
    ax.grid(which="major", axis="y", c="k", alpha=0.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlim(
        pd.Timestamp("2020-03-01"),
        result.index.get_level_values("fecha")[-1] + pd.Timedelta(days=1),
    )
    fig.set_facecolor("w")


fig, ax = plt.subplots(figsize=(600 / 72, 400 / 72))

plot_rt(result, ax, dpto)
ax.set_title(f"{dpto} | $R_t$ \n {latest_date}")
# ax.xaxis.set_major_locator(mdates.WeekdayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.savefig("gráficos/bta_rt.svg")


# %%
sigmas = np.linspace(1 / 20, 1, 20)

targets = colombia.index.get_level_values("departamento").isin(lista_dpto)
dpto_proceso = colombia.loc[targets]

results = {}
failed_colombia = []
skipped_colombia = []

for dpto, casos in dpto_proceso.groupby(level="departamento"):

    print(dpto)
    new, smoothed = prepare_cases(casos, cutoff=1)

    if len(smoothed) < 5:
        skipped_colombia.append(dpto)
        continue

    result = {}

    # Holds all posteriors with every given value of sigma
    result["posteriors"] = []

    # Holds the log likelihood across all k for each value of sigma
    result["log_likelihoods"] = []

    try:
        for sigma in sigmas:
            posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)
            result["posteriors"].append(posteriors)
            result["log_likelihoods"].append(log_likelihood)
        # Store all results keyed off of state name
        results[dpto] = result
    #         clear_output(wait=True)
    except:
        failed_colombia.append(dpto)
        print(f"Posteriors failed for {dpto}")

print(f"Posteriors failed for {len(failed_colombia)} departamentos: {failed_colombia}")
print(f"Skipped {len(skipped_colombia)} departamentos: {skipped_colombia}")
print(f"Continuing with {len(results)} counties / {len(lista_dpto)}")
print("Done.")


# %%
# Each index of this array holds the total of the log likelihoods for
# the corresponding index of the sigmas array.
total_log_likelihoods = np.zeros_like(sigmas)

# Loop through each state's results and add the log likelihoods to the running total.
for dpto, result in results.items():
    total_log_likelihoods += result["log_likelihoods"]

# Select the index with the largest log likelihood total
max_likelihood_index = total_log_likelihoods.argmax()
# print(max_likelihood_index)

# Select the value that has the highest log likelihood
sigma = sigmas[max_likelihood_index]

# Plot it
fig, ax = plt.subplots()
ax.set_title(f"Valor de probabilidad máxima para $\sigma$ = {sigma:.2f}")
ax.plot(sigmas, total_log_likelihoods)
ax.axvline(sigma, color="k", linestyle=":")

# %% [markdown]
"""
### Compilar resultados finales

Dado que hemos seleccionado el óptimo $\sigma$, tomemos la parte posterior
precalculada correspondiente a ese valor de $\sigma$ para cada departamento.
Calculemos también los intervalos de densidad más alta del 90% y 50% (esto lleva
un poco de tiempo) y también el valor más probable.
"""

# %%
final_results = None
hdi_error_list = []

for dpto, result in results.items():
    print(dpto)
    try:
        posteriors = result["posteriors"][max_likelihood_index]
        hdis_90 = highest_density_interval(posteriors, p=0.9)
        hdis_50 = highest_density_interval(posteriors, p=0.5)
        most_likely = posteriors.idxmax().rename("ML")
        result = pd.concat([most_likely, hdis_90, hdis_50], axis=1)
        if final_results is None:
            final_results = result
        else:
            final_results = pd.concat([final_results, result])
        clear_output(wait=True)
    except:
        print(f"HDI failed for {dpto}")
        hdi_error_list.append(dpto)
        pass

print(f"HDI error list: {hdi_error_list}")
print("Done.")

# %% [markdown]
"""
### Trazar todos los departamentos que cumplen con los criterios
"""

# %%
ncols = 3
nrows = int(np.ceil(len(final_results.groupby("departamento")) / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 3))

for i, (dpto, result) in enumerate(final_results.groupby("departamento")):
    plot_rt(result, axes.flat[i], dpto)

fig.tight_layout()
fig.set_facecolor("w")
plt.savefig("gráficos/col_dptos_rt.svg")

# %% [markdown]
"""
### Export Data to CSV
"""

# %%
# Uncomment the following line if you'd like to export the data
final_results.to_csv(f"datos/rt_colombia.csv")


# %% [markdown]
"""
### Clasificaciones finales
"""

# %%
FULL_COLOR = [0.7, 0.7, 0.7]
NONE_COLOR = [179 / 255, 35 / 255, 14 / 255]
PARTIAL_COLOR = [0.5, 0.5, 0.5]
ERROR_BAR_COLOR = [0.3, 0.3, 0.3]


# %%
final_results


# %%
FILTERED_REGIONS = []
filtered = final_results.index.get_level_values(0).isin(FILTERED_REGIONS)
mr = final_results.loc[~filtered].groupby(level=0)[["ML", "High_90", "Low_90"]].last()


def plot_standings(mr, figsize=None, title="Most Likely Recent $R_t$ by County"):
    if not figsize:
        figsize = ((15.9 / 50) * len(mr) + 0.1, 4.6)

    fig, ax = plt.subplots()

    ax.set_title(title)
    err = mr[["Low_90", "High_90"]].sub(mr["ML"], axis=0).abs()
    bars = ax.bar(
        mr.index,
        mr["ML"],
        width=0.825,
        color=FULL_COLOR,
        ecolor=ERROR_BAR_COLOR,
        capsize=2,
        error_kw={"alpha": 0.5, "lw": 1},
        yerr=err.values.T,
    )

    labels = mr.index.to_series()
    ax.set_xticklabels(labels, rotation=90, fontsize=11)
    ax.margins(0)
    ax.set_ylim(0, 2.0)
    ax.axhline(1.0, linestyle=":", color="k", lw=1)

    #     fig.tight_layout()
    fig.set_facecolor("w")
    return fig, ax


mr.sort_values("ML", inplace=True)
plot_standings(mr, title=f"Valores más probables de $R_t$ | {latest_date}")
# plt.figure(figsize=(3,8))
plt.savefig("gráficos/colombia_rt.svg", bbox_inches="tight")


# %%
mr.sort_values("High_90", inplace=True)
plot_standings(mr, title=f"Valores más (altos) probables de $R_t$ | {latest_date}")
plt.savefig("gráficos/colombia_rt_alto.svg", bbox_inches="tight")


# %%
show = mr[mr.High_90.le(1)].sort_values("ML")
fig, ax = plot_standings(
    show, title=f"Departamentos que tienen la \n pandemia bajo control | {latest_date}"
)
plt.savefig("gráficos/colombia_rt_controlado.svg", bbox_inches="tight")


# %%
show = mr[mr.Low_90.ge(1.0)].sort_values("Low_90")
fig, ax = plot_standings(
    show,
    title=f"Departamentos que no tienen \n la pandemia bajo control | {latest_date}",
)
plt.savefig("gráficos/colombia_rt_descontrolada.svg", bbox_inches="tight")


# %%
cronologia = "datos/cronologia.csv"

colombia = pd.read_csv(cronologia)

# Print Counties
latest_date = colombia[-1:]
latest_date = latest_date.fecha
latest_date = " ".join(str(elem) for elem in latest_date)
print(latest_date)

col_latest = colombia
col_latest.drop(col_latest[col_latest["fecha"] != latest_date].index, inplace=True)

col_latest

pop_colombia = pd.read_csv("datos/población_colombia.csv")
pop_colombia = pop_colombia.set_index(["departamento"]).sort_index()

rt_colombia = pd.read_csv("datos/rt_colombia.csv")
latest_rt_colombia = rt_colombia[rt_colombia.fecha == latest_date]
departamentos_rt = list(latest_rt_colombia["departamento"])


pop_colombia = pop_colombia[pop_colombia.index.isin(departamentos_rt)]

col_latest = col_latest[col_latest.departamento.isin(departamentos_rt)]

col_latest = col_latest.sort_values(by=["departamento"])

col_latest["capital"] = list(pop_colombia["capital"])
col_latest["población"] = list(pop_colombia["población"])
col_latest["tasa_casos_por_población"] = round(
    col_latest["casos"] / col_latest["población"], 4
)
col_latest = col_latest.reset_index(drop=True)
col_latest.to_csv("datos/col_latest.csv", index=False)
col_latest.sort_values(by=["casos"], ascending=False)


# %%
dp_ref = col_latest

latest_rt_colombia["capital"] = list(dp_ref["capital"])
latest_rt_colombia["dp"] = list(dp_ref["dp"])
latest_rt_colombia["población"] = list(col_latest["población"])
latest_rt_colombia["casos"] = list(col_latest["casos"])
latest_rt_colombia["tasa_casos_por_población"] = list(
    col_latest["tasa_casos_por_población"]
)
latest_rt_colombia.reset_index(drop=True)


# %%
latest_rt_colombia.to_csv("datos/latest_rt_colombia.csv", index=False)


# %%
latest_rt_colombia


# %%
import json
from urllib.request import urlopen

import plotly.express as px
import plotly.io as pio

response = "datos/geo-colombia.json"

with open(response) as response:
    dptos = json.load(response)


df = pd.read_csv("datos/latest_rt_colombia.csv", dtype={"dp": int})


fig = px.choropleth_mapbox(
    df,
    geojson=dptos,
    locations="dp",
    featureidkey="properties.DPTO",
    color="ML",
    color_continuous_scale=[
        (0, "green"),
        (0.5, "rgb(135, 226, 135)"),
        (0.5, "rgb(226, 136, 136)"),
        (1, "red"),
    ],
    hover_name="departamento",
    hover_data=["fecha", "capital", "población", "casos", "tasa_casos_por_población"],
    range_color=(0, 2),
    mapbox_style="carto-positron",
    zoom=5.146781362543418,
    center={"lat": 4.425972776564322, "lon": -73.72494645042087},
    opacity=0.8,
    labels={
        "dp": "Código DIVIPOLA",
        "ML": "Valor más probable de Rₜ",
        "fecha": "Fecha",
        "población": "Población",
        "casos": "Casos",
        "tasa_casos_por_población": "Tasa: Casos por Población",
        "capital": "Capital",
    },
)

fig.layout.font.family = "Arial"

fig.update_layout(
    width=1000,
    height=1000,
    title=f"Colombia | Mapa Rₜ por Departamento [{latest_date}]",
    annotations=[
        dict(
            xanchor="right",
            x=1,
            yanchor="top",
            y=-0.05,
            showarrow=False,
            text="Fuente: Ministerio de Salud y Protección Social: Instituto Nacional de Salud, DANE",
        )
    ],
)


fig.show()


# %%
# pio.write_json(fig, "choro.json")


# %%
with open("../danielcs88.github.io/html/rt_colombia.html", "w") as f:
    f.write(fig.to_html(include_plotlyjs="cdn"))


# %%
get_ipython().system(" cd ../danielcs88.github.io/ && git pull")


# %%
get_ipython().system(
    ' cd ../danielcs88.github.io/ && git add --all && git commit -m "Update" && git push'
)
