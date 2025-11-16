# IE Teachers Knowledge Graph

Student-friendly pipeline to parse IE University teacher bios, extract entities, normalise them, and build a lightweight knowledge graph. The notebook now fuses structured bullet parsers with dual Hugging Face NER models and a weighted scorer so every relation keeps provenance, confidence, and alias-aware canonical labels.

[![Open In Colab](https://colab.research.googleusercontent.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/<USER>/ie-teachers-kg/blob/main/notebooks/main.ipynb)

## How to run in Google Colab
1. Open the badge above or go to Colab and choose *File → Open Notebook → GitHub*.
2. Clone the repo inside Colab:
   ```bash
   !git clone https://github.com/<USER>/ie-teachers-kg.git
   %cd ie-teachers-kg
   ```
3. Open `notebooks/main.ipynb` and run `Runtime → Restart and run all`. All dependencies are installed in the first cell.

## Dependencies
`requirements.txt` lists the exact versions used (pandas, BeautifulSoup, transformers, spaCy, NetworkX, etc.). The notebook installs them automatically and is designed to be deterministic (random seeds set at the top). Please run in a fresh Colab session for the best experience.

## Outputs
Running the notebook produces the following artifacts inside `outputs/`:
- `nodes.csv`, `edges.csv`, `graph.gexf` with the NetworkX knowledge graph.
- QA snippets, plots, and a final ZIP bundle saved as `ie-teachers-kg_submit_YYYYMMDD.zip` containing the notebook, data subset, requirements, and outputs.

Each notebook section contains markdown guidance plus quick QA prints so students can inspect intermediate results.
