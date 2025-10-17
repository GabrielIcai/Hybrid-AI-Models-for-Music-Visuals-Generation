Bachelor's Thesis (TFG) project focused on the generation of reactive visuals for music using Artificial Intelligence models. The system uses three distinct models: one for musical genre prediction, another for detecting emotions in music, and a third for analyzing the song's structure.
The main architecture combines convolutional networks (CNNs) for audio feature extraction with recurrent networks (RNNs/LSTMs) to capture the temporal sequence of the music. This allows the visuals to respond dynamically not only to the BPM but also to changes in genre, emotion, and song sections.
The project demonstrates how to apply deep learning and audio processing techniques to create unique and adaptive visual experiences, oriented towards concerts and live shows.
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
