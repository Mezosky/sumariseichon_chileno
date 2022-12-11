# Sumariseichon_chileno 🐣

Objetivo: Finetunear modelos encoder decoder para hacer sumarizacion automatica
con texto chileno.

##  Instrucciones básicas

Crear el ambiente e instalar los requerimientos:

```bash
$ conda env create -f environment.yml
```

Para activarlo:

```bash
$ conda activate sum
```

## Entrenar el modelo

Ejecutar:

```bash
$ cd src
$ python run.py
```

Mientras, en otra terminal ejecutar


```bash
$ mlflow ui
```

## QA

La primera vez que se clone el proyecto, instalar pre-commit.

```bash
pre-commit install
```

Luego, para ejecutar el control de calidad, ejecutar

```bash
pre-commit run --all-files
```

## Referencias

1. [Fine Tuning a T5 transformer for any Summarization Task - Medium](https://towardsdatascience.com/fine-tuning-a-t5-transformer-for-any-summarization-task-82334c64c81)
2. ...
