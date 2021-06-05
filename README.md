<meta name="viewport" content="width=device-width, initial-scale=1.0">

# Análisis de la pandemia COVID-19 en 🇨🇴 Colombia a nivel departamental | R<sub>t</sub>

por Daniel Cárdenas

Si R<sub>t</sub> > 1, el número de casos aumentará, como al comienzo de una
epidemia. Cuando R<sub>t</sub> = 1, la enfermedad es endémica, y cuando
R<sub>t</sub> < 1 habrá una disminución en el número de casos.

Entonces, los epidemiólogos usan R<sub>t</sub> para hacer recomendaciones de
políticas. Es por eso que este número es tan importante.

## Notas

Por cuestiones de simplicidad, el único distrito especial es en el caso de
Bogotá. En la página nacional, hay varias ciudades que son su propio distrito
(Barranquilla, Cartagena, Santa Marta y Buenaventura) fueron incluidas en su departamento
respectivo (Atlántico, Bolívar, Magdalena y Valle del Cauca).

Mi modelo es una adaptación del model de
[Kevin Systrom](https://github.com/k-sys/covid-19)

## Fuente de Datos

Mi fuente de datos es del Ministerio de Salud de Colombia y su plataforma
[Casos positivos de COVID-19 en Colombia](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia/gt2j-8ykr/data)

## Requerimientos

- [Anaconda](https://www.anaconda.com/products/individual#Downloads)
- [`sodapy`](https://github.com/xmunoz/sodapy)

  sodapy es el cliente de Python para usar la API de Socrata Open Data. Se puede
  instalar usando `PyPI`.

  ```sh
  pip install sodapy
  ```

- [Plotly](https://plotly.com/python/getting-started/#installation)
