# OGH Summer School Hackathon 25: Estimating Canopy Height

This is my entry for OpenGeoHub's summer school 2025 hackathon. The task is to
predict canopy height from Google's Alpha Earth Foundations (AEF) embeddings.

Canopy height targets to predict are measured using the GEDI sensor and use the `rh98` 
value as a proxy for canopy heihgt.

Details for this hackathon can be found on [kaggle](https://www.kaggle.com/competitions/canopy-height-mapping-using-google-aef-embeddings/overview)

Thanks to the awesome people at [OpenGeoHub](https://opengeohub.org/) in the Netherlands for organizing this 
summer school and the hackathon!

## Approach

To estimate canopy height, I use a simple MLP using an input layer, one hidden layer,
one ReLU ad then a single number output.

## Installation

To run the script follow these steps:

1. Clone this repository: `git clone git@github.com:jonas-hurst/ogh-ss25-hackathon-canopy.git`
2. Install the environment using [uv](https://docs.astral.sh/uv/getting-started/installation/):
   `uv sync` (Could take some time as pytorch is big...)
3. Then activate the environment:
   - Linux: `source .venv/bin/activate`
   - For Windows (cmd): `.venv/bin/activate.bat`
   - For Windows (Powershell): `.venv/bin/activate.ps1`
   - Fow macOS: idk ¯\\\_(ツ)\_/¯
4. run the jupyter server: `jupyter lab`

The browser should now open. Select the 'hackathon.ipynb' notebook
