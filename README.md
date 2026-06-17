# EvoStruct-Kla: Evolutionary-structural ensemble learning for protein lysine lactylation site prediction
Lysine lactylation (Kla) is a recently identified post-translational modification (PTM) that plays important roles in a variety of physiological and pathological processes, including cancer cell proliferation, cardiac dysfunction, and immune-related diseases. Traditional experimental approaches for identifying Kla sites are typically labor-intensive and time-consuming, while existing computational methods mainly rely on sequence information or handcrafted prior features and therefore have difficulty effectively characterizing the three-dimensional structural features. In this study, we propose a novel prediction framework that integrates topology-aware structural neighborhood modeling with evolution-informed sequence context modeling, thereby fully exploiting the complementarity between structural and contextual information. Specifically, our framework includes a structural graph branch and an evolutionary context branch. The structural graph branch employs graph attention network (GAT) on contact graphs constructed from AlphaFold-predicted structures to model structural neighborhood relationships, the evolutionary context branch introduces ESM2-derived evolutionary pairwise bias into the attention mechanism to model contextual dependencies. The representations produced by the two branches are then fused through an ensemble learning framework.
![model](https://github.com/JaunYue/EvoStruct-Kla/blob/main/workflow.png)
# Full dataset for this study
Due to the GitHub upload size limit, we have only uploaded the demo dataset. The full dataset can be obtained via the link https://zenodo.org/records/20049452?preview=1. 
# Environment
- python 3.9.22
- torch 2.6.0
- numpy 1.23.5
- scikit-learn 1.6.1
- torch-geometric 2.6.1
- CUDA 12.4
# Reproduction
1. Clone this repository and enter the project directory:
```bash
git clone https://github.com/JaunYue/EvoStruct-Kla.git
cd EvoStruct-Kla
```
2. Create and activate the experimental environment:
```bash
conda create -n evostruct-kla python=3.9.22 -y
conda activate evostruct-kla
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```
4. Run EvoStruct-Kla from the repository root directory:
```bash
python Code/main.py
```
