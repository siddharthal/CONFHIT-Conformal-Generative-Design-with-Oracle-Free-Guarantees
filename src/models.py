import torch
import numpy as np
from torch.utils.data import DataLoader
import random
import pandas as pd
from chemprop import models, nn
from chemprop.data.datapoints import MoleculeDatapoint
from chemprop.data.datasets import MoleculeDataset
from chemprop.data.collate import collate_batch

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def sample_for_testing(df, n_samples=1000, seed=42):
    """Randomly sample rows from a dataframe for testing/debugging."""
    if len(df) <= n_samples:
        return df
    random.seed(seed)
    return df.sample(n=n_samples, random_state=seed)

def define_binary_model():
    """Define the binary classification model architecture."""
    mpnn = models.MPNN(
        message_passing=nn.message_passing.BondMessagePassing(),
        agg=nn.agg.MeanAggregation(),
        predictor=nn.predictors.BinaryClassificationFFN(),
        batch_norm=True,
        metrics=[nn.metrics.BinaryAccuracy(), nn.metrics.BinaryAUROC()]
    )
    return mpnn

def load_model(checkpoint_path):
    """Load a binary model from a checkpoint."""
    model = define_binary_model()
    model = model.__class__.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def extract_features(model, smiles_pairs, batch_size=32):
    """Extract features and scores from a binary model for SMILES pairs."""
    features = []
    def hook_fn(_, __, output):
        features.append(output.detach().cpu().numpy())
    model.predictor.ffn[1].register_forward_hook(hook_fn)

    datapoints = [MoleculeDatapoint.from_smi(pair[1]) for pair in smiles_pairs]
    dataset = MoleculeDataset(datapoints)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
    all_features = []
    all_scores = []
    model.cpu()
    with torch.no_grad():
        for batch in loader:
            output = model(batch.bmg, batch.V_d, batch.X_d)
            scores = output.cpu().numpy()
            all_scores.append(scores)
            if features:
                all_features.append(features[-1])
                features = []
    all_features = np.vstack(all_features)
    all_scores = np.concatenate(all_scores, axis=0)
    all_scores = 1 - all_scores
    return all_features, all_scores.flatten()
