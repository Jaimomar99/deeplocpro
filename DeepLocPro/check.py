import torch

path = '/home/projects/cu_10017/people/marodu/PredictionOfMembraneProteinTypesUsingLanguageModels/'

torch.hub.set_dir(path + "models/torchhub_models")

model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")

print(model)
