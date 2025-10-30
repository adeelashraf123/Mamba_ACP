import os
import time
import numpy as np
import pandas as pd
import torch
import csv
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, \
    confusion_matrix, classification_report, roc_curve, auc
# Update this import at the top of your file
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, \
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score  # Add roc_auc_score here
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import is_tensorboard_available
from transformers import TrainerCallback
import shutil
from sklearn.model_selection import KFold
import random
import torch.nn as nn
from transformers import AutoModel
torch.cuda.empty_cache()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import MambaConfig, MambaModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy import stats
import matplotlib as mpl
from scipy.special import softmax


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Model checkpoint and dataset paths
model_checkpoint = '/data1/adeel/Transformer/esm2_t30_150M_UR50D/'  # Larger model
poz_path = '/data1/adeel/pos_train.csv'
non_path = '/data1/adeel/non_train.csv'
val_poz_path = '/data1/adeel/pos_test.csv'
val_non_path = '/data1/adeel/non_test.csv'
aaindex_path = '/data1/adeel/aaindex_feature.csv'


# Load AAINDEX features
def load_aaindex_features(feature_file=aaindex_path):
    """Load AAIndex features from file into a dictionary"""
    aaindex_features = {}
    with open(feature_file, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            aa = row[0]
            features = [float(value) if value != '' else 0.0 for value in row[1:]]
            aaindex_features[aa] = features
    return aaindex_features


# BLOSUM62 matrix
BLOSUM62_MATRIX = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
    '*': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '_': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}


class FeaturePCA:
    def __init__(self, variance_threshold=0.95):
        self.pca = PCA(n_components=variance_threshold)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.blosum_matrix = None
        self.aaindex_features = None
        self.aaindex_dim = None

    def fit(self, blosum_matrix, aaindex_features):
        # Store references to the feature matrices
        self.blosum_matrix = blosum_matrix
        self.aaindex_features = aaindex_features
        self.aaindex_dim = len(next(iter(aaindex_features.values())))

        # Create combined features for all amino acids
        all_aas = set(blosum_matrix.keys()).union(set(aaindex_features.keys()))
        combined_features = []

        for aa in all_aas:
            blosum = np.array(blosum_matrix.get(aa, [0] * 20))
            aaindex = np.array(aaindex_features.get(aa, [0] * self.aaindex_dim))
            combined = np.concatenate([blosum, aaindex])
            combined_features.append(combined)

        # Standardize and fit PCA
        X = np.array(combined_features)
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.is_fitted = True
        return self

    def transform(self, sequences):
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted yet. Call fit() first.")

        # Extract original features for each amino acid in sequences
        seq_features = []
        for seq in sequences:
            for aa in seq:
                blosum = np.array(self.blosum_matrix.get(aa, [0] * 20))
                aaindex = np.array(self.aaindex_features.get(aa, [0] * self.aaindex_dim))
                combined = np.concatenate([blosum, aaindex])
                seq_features.append(combined)

        # Transform
        X = np.array(seq_features)
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)


# Feature extractor class
class FeatureExtractor:
    def __init__(self, blosum_matrix, aaindex_features, pca_variance=0.95):
        """Initialize with feature matrices and PCA variance threshold"""
        self.blosum_matrix = blosum_matrix
        self.aaindex_features = aaindex_features
        self.aaindex_dim = len(next(iter(aaindex_features.values())))
        self.pca_variance = pca_variance

        # Initialize PCA components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_variance)
        self._fit_pca()

    def _fit_pca(self):
        """Internal method to fit PCA on all amino acids"""
        all_aas = set(self.blosum_matrix.keys()).union(set(self.aaindex_features.keys()))
        combined_features = []

        for aa in all_aas:
            blosum = np.array(self.blosum_matrix.get(aa, [0] * 20))
            aaindex = np.array(self.aaindex_features.get(aa, [0] * self.aaindex_dim))
            combined = np.concatenate([blosum, aaindex])
            combined_features.append(combined)

        X = np.array(combined_features)
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)

    def extract_features(self, sequences):
        """Extract and reduce features for a batch of sequences"""
        features = []
        for seq in sequences:
            # Get features for each amino acid
            aa_features = []
            for aa in seq:
                blosum = np.array(self.blosum_matrix.get(aa, [0] * 20))
                aaindex = np.array(self.aaindex_features.get(aa, [0] * self.aaindex_dim))
                combined = np.concatenate([blosum, aaindex])
                aa_features.append(combined)

            # Transform and average
            if aa_features:
                X = np.array(aa_features)
                X_scaled = self.scaler.transform(X)
                pca_features = self.pca.transform(X_scaled)
                features.append(np.mean(pca_features, axis=0))
            else:
                features.append(np.zeros(self.pca.n_components_))

        return np.array(features)


# Load data
def load_data(file_path):
    return pd.read_csv(file_path)


# Data Augmentation Functions (same as your original code)
def mutate_sequence(sequence, mutation_rate=0.05):
    sequence = list(sequence)
    for i in range(len(sequence)):
        if random.random() < mutation_rate:
            sequence[i] = random.choice("ACDEFGHIKLMNPQRSTVWY")  # Random amino acid substitution
    return ''.join(sequence)


def insert_sequence(sequence, insertion_rate=0.05):
    sequence = list(sequence)
    for i in range(len(sequence)):
        if random.random() < insertion_rate:
            sequence.insert(i, random.choice("ACDEFGHIKLMNPQRSTVWY"))  # Insert random amino acid
    return ''.join(sequence)


def delete_sequence(sequence, deletion_rate=0.05):
    sequence = list(sequence)
    sequence = [aa for aa in sequence if random.random() > deletion_rate]
    return ''.join(sequence)


def shuffle_sequence(sequence):
    sequence = list(sequence)
    random.shuffle(sequence)
    return ''.join(sequence)


def augment_data(data, labels, augmentation_factor=2):
    augmented_data = []
    augmented_labels = []
    for seq, label in zip(data, labels):
        augmented_data.append(seq)
        augmented_labels.append(label)
        for _ in range(augmentation_factor):
            augmented_seq = mutate_sequence(seq)
            augmented_seq = insert_sequence(augmented_seq)
            augmented_seq = delete_sequence(augmented_seq)
            augmented_seq = shuffle_sequence(augmented_seq)
            augmented_data.append(augmented_seq)
            augmented_labels.append(label)
    return augmented_data, augmented_labels


def tokenize_data(tokenizer, data):
    return tokenizer(
        data,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512  # Adjust based on your sequence lengths
    )



# Enhanced Hybrid Model with BLOSUM62 and AAINDEX

class EnhancedHybridModel(nn.Module):
    def __init__(self, transformer_checkpoint, blosum_matrix=None, aaindex_features=None, pca_variance=0.95):
        super().__init__()

        # Feature extractor initialization (unchanged)
        self.blosum_matrix = blosum_matrix or BLOSUM62_MATRIX
        self.aaindex_features = aaindex_features or load_aaindex_features()
        self.feature_extractor = FeatureExtractor(
            blosum_matrix=self.blosum_matrix,
            aaindex_features=self.aaindex_features,
            pca_variance=pca_variance
        )

        # Get feature dimensions
        test_features = self.feature_extractor.extract_features(["A"])
        self.handcrafted_dim = test_features.shape[1]

        # Initialize transformer
        self.transformer = AutoModel.from_pretrained(transformer_checkpoint)
        transformer_dim = self.transformer.config.hidden_size

        # Mamba configuration - using MambaModel instead of MambaForCausalLM
        self.mamba_config = MambaConfig(
            hidden_size=transformer_dim + self.handcrafted_dim,
            num_hidden_layers=4,
            intermediate_size=256,
        )

        # Initialize base Mamba model without LM head
        self.mamba = MambaModel(self.mamba_config)

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim + self.handcrafted_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary classification
        )

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, sequences=None, labels=None):
        # Transformer features
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        transformer_features = outputs.last_hidden_state[:, 0]

        # Handcrafted features
        if sequences is None:
            handcrafted = torch.zeros(len(input_ids), self.handcrafted_dim,
                                      device=input_ids.device)
        else:
            features = []
            for seq in sequences:
                seq_features = self.feature_extractor.extract_features([seq])[0]
                features.append(seq_features)
            handcrafted = torch.tensor(np.array(features),
                                       device=input_ids.device,
                                       dtype=torch.float32)

        # Combine features
        combined = torch.cat([transformer_features, handcrafted], dim=1)

        # Reshape for Mamba: [batch_size, seq_len=1, hidden_size]
        combined = combined.unsqueeze(1)

        # Mamba forward pass - get last hidden state
        mamba_output = self.mamba(
            inputs_embeds=combined
        ).last_hidden_state

        # Remove sequence dimension and apply classifier
        mamba_output = mamba_output.squeeze(1)
        logits = self.classifier(mamba_output)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return {'loss': loss, 'logits': logits} if loss is not None else logits


# Enhanced Dataset Class
class EnhancedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, sequences=None, labels=None):
        """Initialize dataset with tokenized encodings, raw sequences, and labels"""
        self.encodings = encodings
        self.sequences = sequences if sequences is not None else [""] * len(encodings["input_ids"])
        self.labels = labels

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "sequences": self.sequences[idx]  # Keep as raw string
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """Return the total number of samples"""
        return len(self.encodings["input_ids"])


class HybridDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Separate sequences from other features
        sequences = [f.pop("sequences") for f in features]

        # Let HF collator handle the standard fields
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        # Add sequences back as a list
        batch["sequences"] = sequences
        return batch


# Update the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to probabilities using softmax
    probabilities = softmax(logits, axis=1)[:, 1]

    preds = np.argmax(logits, axis=1)

    # Calculate standard metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)

    # Calculate AUC
    auc_score = roc_auc_score(labels, probabilities)

    return {
        "eval_accuracy": accuracy,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_mcc": mcc,
        "eval_sensitivity": sensitivity,
        "eval_specificity": specificity,
        "eval_fpr": fpr,
        "eval_auc": auc_score,
    }


# Custom Trainer to handle handcrafted features
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # Add **kwargs
        sequences = inputs.pop("sequences", None)
        labels = inputs.pop("labels", None)

        # Convert sequences to list if it's a single string
        if isinstance(sequences, str):
            sequences = [sequences]

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            sequences=sequences,
            labels=labels
        )

        loss = outputs["loss"] if labels is not None else None
        return (loss, outputs) if return_outputs else loss


# Train the model
def train_model(model, tokenizer, train_dataset, val_dataset, fold_num=None):
    # Create unique output directory for each fold
    output_dir = f"/data1/adeel/Transformer/Output/fold_{fold_num}" if fold_num else "/data1/adeel/Transformer/Output/"

    data_collator = HybridDataCollator(tokenizer)
    batch_size = 16

    args = TrainingArguments(
        output_dir=output_dir,  # Use fold-specific directory
        eval_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=5,
        learning_rate=7e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.001,
        warmup_ratio=0.1,
        logging_dir="./logs",
        logging_steps=2,
        gradient_accumulation_steps=2,
        save_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        lr_scheduler_type="cosine",
        greater_is_better=True,
        max_grad_norm=1.0,
        fp16=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

    return trainer


# Predict using the model - updated to return probabilities
def predict_model(trainer, dataset):
    predictions = trainer.predict(dataset)
    # Get the softmax probabilities for the positive class (class 1)
    probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions.predictions), dim=1)[:, 1].numpy()
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    return true_labels, predicted_labels, probabilities


# Add this function to your code
def plot_roc_curves(fold_results, overall_true_labels, overall_probabilities, save_path=None):
    """
    Plot ROC curves for each fold, average ROC, and overall ROC

    Args:
        fold_results: List of tuples (fold_num, true_labels, probabilities)
        overall_true_labels: Combined true labels from all folds
        overall_probabilities: Combined probabilities from all folds
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))

    # Define a color palette with distinct colors
    fold_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
    average_color = '#2a52be'  # Distinct blue for average
    overall_color = '#e377c2'  # Distinct pink/magenta for overall

    # Store interpolated TPRs for average curve calculation
    all_fpr = np.linspace(0, 1, 100)
    interp_tprs = []
    fold_aucs = []

    # Plot individual folds with distinct colors
    for i, (fold_num, true_labels, probabilities) in enumerate(fold_results):
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        fold_auc = auc(fpr, tpr)
        fold_aucs.append(fold_auc)

        # Interpolate to common FPR points for average curve
        interp_tpr = np.interp(all_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

        # Plot individual fold curve
        plt.plot(fpr, tpr, lw=1.5, alpha=0.7, color=fold_colors[i],
                 label=f'Fold {fold_num} (AUC = {fold_auc:.4f})')

    # Calculate and plot average ROC curve
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color=average_color, linestyle=':', lw=3,
             label=f'Average (AUC = {mean_auc:.4f})', alpha=0.9)

    # Calculate and plot overall ROC curve
    overall_fpr, overall_tpr, _ = roc_curve(overall_true_labels, overall_probabilities)
    overall_auc = auc(overall_fpr, overall_tpr)
    plt.plot(overall_fpr, overall_tpr, color=overall_color, linestyle='-', lw=3,
             label=f'Overall (AUC = {overall_auc:.4f})', alpha=0.9)

    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)

    # Format plot
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('(B) ROC Curves on Set 2', fontsize=14, fontweight='bold')

    # Position legend to avoid overlapping curves
    plt.legend(loc='lower right', fontsize=10)

    # Add grid with subtle styling
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Plot radar map - updated (same as your original code)
def plot_radar_chart_with_mean_values(metrics_dict, avg_metrics, save_path=None):
    """Radar chart with integrated metric names in value annotations"""

    def convert_to_pct(val):
        return val * 100 if val <= 1.0 else val

    categories = ['precision', 'recall', 'f1', 'accuracy', 'specificity', 'mcc']
    category_labels = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity', 'MCC']

    labels = list(metrics_dict.keys())
    values = []
    for label in labels:
        fold_values = [convert_to_pct(metrics_dict[label][cat]) for cat in categories]
        values.append(fold_values)

    mean_values = [convert_to_pct(avg_metrics[f'eval_{cat}']) for cat in categories]
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    mean_wrapped = mean_values + [mean_values[0]]

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    # Color scheme
    fold_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Distinct colors
    mean_color = '#e377c2'  # Distinct magenta for mean

    # Plot individual folds
    for idx, vals in enumerate(values[:-1]):  # Exclude 'Average' entry
        vals_wrapped = vals + [vals[0]]
        ax.plot(angles, vals_wrapped, linewidth=2.0, alpha=0.7,
                color=fold_colors[idx % len(fold_colors)],
                label=f'Fold {idx + 1}')

    # Highlight mean
    ax.plot(angles, mean_wrapped, linewidth=4, color=mean_color,
            label='Mean', marker='o', markersize=10, zorder=10)
    ax.fill(angles, mean_wrapped, color=mean_color, alpha=0.15)

    # Radial axis setup
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_yticklabels([])

    # INTEGRATED ANNOTATIONS WITH METRIC NAMES
    for angle, val, cat_label in zip(angles[:-1], mean_values, category_labels):
        # Position text with metric name and value
        ax.text(angle, 105, f"{cat_label}\n{val:.2f}%",
                ha='center', va='center', fontsize=11, fontweight='bold',
                color=mean_color,
                bbox=dict(facecolor='white', alpha=0.95, boxstyle='round,pad=0.3',
                          edgecolor=mean_color, linewidth=1))

    # Remove category labels completely
    ax.set_xticklabels([])

    # Title and legend
    ax.set_title('(B) Performance Metrics on Set 2', pad=30, fontsize=16, fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=True, fontsize=11)

    # Grid styling
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.grid(axis='x', linestyle='-', alpha=0.8)
    ax.spines['polar'].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

# K-Fold Cross-Validation
# K-Fold Cross-Validation
def fold():
    # Load data
    poz = load_data(poz_path)
    non = load_data(non_path)
    val_poz = load_data(val_poz_path)
    val_non = load_data(val_non_path)

    # Concatenate all data
    all_data = pd.concat([poz, non, val_poz, val_non])
    all_labels = all_data["label"]

    # Shuffle data
    all_data = shuffle(all_data, random_state=42)

    # Load AAINDEX features
    aaindex_features = load_aaindex_features()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = EnhancedHybridModel(
        transformer_checkpoint=model_checkpoint,
        blosum_matrix=BLOSUM62_MATRIX,
        aaindex_features=aaindex_features
    )

    # Separate positive and negative samples
    pos_data = all_data[all_data["label"] == 1]
    neg_data = all_data[all_data["label"] == 0]

    # Number of folds
    n_folds = 5
    metrics_list = []
    all_true_labels = []
    all_probabilities = []
    fold_results = []  # Will store (fold_num, true_labels, probabilities)

    for fold in range(1, n_folds + 1):
        print(f"Processing fold {fold}")

        # Sample exactly 186 positive and 170 negative samples for validation
        val_pos_samples = pos_data.sample(n=186, random_state=fold)
        val_neg_samples = neg_data.sample(n=170, random_state=fold)

        # Combine to create the validation set
        val_data_fold = pd.concat([val_pos_samples, val_neg_samples])

        # Remove validation samples from the training data
        train_pos_samples = pos_data.drop(val_pos_samples.index)
        train_neg_samples = neg_data.drop(val_neg_samples.index)

        # Combine to create the training set
        train_data_fold = pd.concat([train_pos_samples, train_neg_samples])

        # Verify validation set size and distribution
        print(f"Validation set size for fold {fold}: {len(val_data_fold)} "
              f"(Positive: {len(val_data_fold[val_data_fold['label'] == 1])}, "
              f"Negative: {len(val_data_fold[val_data_fold['label'] == 0])})")

        # Augment training data
        train_data_augmented, train_labels_augmented = augment_data(
            train_data_fold["data"].tolist(),
            train_data_fold["label"].tolist()
        )

        # Tokenize train and validation data
        X_train_tokenized = tokenize_data(tokenizer, train_data_augmented)
        X_val_tokenized = tokenize_data(tokenizer, val_data_fold["data"].tolist())

        # Create datasets with sequences for handcrafted features
        train_dataset = EnhancedDataset(
            X_train_tokenized,
            sequences=train_data_augmented,
            labels=train_labels_augmented
        )
        val_dataset = EnhancedDataset(
            X_val_tokenized,
            sequences=val_data_fold["data"].tolist(),
            labels=val_data_fold["label"].tolist()
        )

        # Train model
        trainer = train_model(model, tokenizer, train_dataset, val_dataset, fold_num=fold)

        # Evaluate model
        eval_results = trainer.evaluate(val_dataset)
        metrics_list.append(eval_results)

        # Get predictions and probabilities for ROC curve
        true_labels, predicted_labels, probabilities = predict_model(trainer, val_dataset)
        fold_results.append((fold, true_labels, probabilities))
        all_true_labels.extend(true_labels)
        all_probabilities.extend(probabilities)

    # Print average metrics across folds
    avg_metrics = {metric: np.mean([m[metric] for m in metrics_list]) for metric in metrics_list[0]}
    print("\nAverage Metrics Across Folds:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Calculate overall AUC
    overall_auc = roc_auc_score(all_true_labels, all_probabilities)
    fold_aucs = [m['eval_auc'] for m in metrics_list]
    avg_auc = np.mean(fold_aucs)

    # Print AUC metrics
    print("\nAUC Metrics:")
    print(f"{'Fold':<10}{'AUC':<10}")
    for i, auc_val in enumerate(fold_aucs):
        print(f"Fold {i+1}:  {auc_val:.4f}")
    print(f"\nAverage Fold AUC: {avg_auc:.4f}")
    print(f"Overall AUC: {overall_auc:.4f}")

    # Prepare metrics for radar chart with MCC instead of Sensitivity
    metrics_dict = {
        f'Fold {i + 1}': {
            'precision': m['eval_precision'],
            'recall': m['eval_recall'],
            'f1': m['eval_f1'],
            'accuracy': m['eval_accuracy'],
            'specificity': m['eval_specificity'],
            'mcc': m['eval_mcc']  # Changed from sensitivity to MCC
        } for i, m in enumerate(metrics_list)
    }

    # Add average metrics
    metrics_dict['Average'] = {
        'precision': avg_metrics['eval_precision'],
        'recall': avg_metrics['eval_recall'],
        'f1': avg_metrics['eval_f1'],
        'accuracy': avg_metrics['eval_accuracy'],
        'specificity': avg_metrics['eval_specificity'],
        'mcc': avg_metrics['eval_mcc']  # Changed from sensitivity to MCC
    }

    plot_radar_chart_with_mean_values(metrics_dict, avg_metrics, "radar_with_folds.png")
    # Plot ROC curves
    plot_roc_curves(fold_results, all_true_labels, all_probabilities, "roc_curves.png")

if __name__ == "__main__":
    fold()
