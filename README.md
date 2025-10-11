## Proyecto de TFG centrado en la generación de visuales reactivas para música mediante modelos de Inteligencia Artificial. El sistema utiliza tres modelos distintos: uno para predicción de género musical, otro para detección de emociones en la música y un tercero para análisis de la estructura de la canción.

La arquitectura principal combina redes convolucionales (CNN) para extracción de características del audio con redes recurrentes (RNN/LSTM) para capturar la secuencia temporal de la música. Esto permite que las visuales respondan dinámicamente no solo al BPM, sino también a cambios de género, emociones y secciones de la canción.

El proyecto demuestra cómo aplicar técnicas de deep learning y procesamiento de audio para crear experiencias visuales únicas y adaptativas, orientadas a conciertos y shows en vivo.
## What to do at first

At first, create the virtual environment:

    python -m venv .venv

Then, activate the virtual environment:

    source .venv/bin/activate

Each time a new package is included use:

    pip install package


## Sanity Checks

They are executed in each push, but if you want to check code and typing style before pushing please follow these steps:

    pytest .
    black --check .
    ruff check
    mypy src tests
    flake8 src tests
    pylint src tests


## Others

To see the documentation run:

    mkdocs serve
