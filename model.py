import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from imblearn.over_sampling import SMOTE
import subprocess  # For docking and MD simulation calls
import sys
sys.path.append(r"C:\Users\Afsa\AppData\Roaming\Python\Python312\site-packages")  # Adjust path if needed
import py3Dmol  # For visualization (requires PyMOL installation)

# EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False

# Dataset Class
class LigandProteinDataset(Dataset):
    def __init__(self, ligand_features, protein_features, labels=None):
        self.ligand_features = ligand_features
        self.protein_features = protein_features
        self.labels = labels

    def __len__(self):
        return len(self.ligand_features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return (self.ligand_features[idx], self.protein_features[idx], self.labels[idx])
        else:
            return (self.ligand_features[idx], self.protein_features[idx])

# Model Architecture
class EnhancedBindingPredictor(nn.Module):
    def __init__(self, ligand_dim, protein_dim, hidden_dim=256, num_heads=8, dropout=0.5):
        super().__init__()
        self.ligand_proj = nn.Sequential(
            nn.Linear(ligand_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2  # Reduced complexity
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, ligand, protein):
        ligand_embed = self.ligand_proj(ligand).unsqueeze(1)
        protein_embed = self.protein_proj(protein).unsqueeze(1)
        combined = torch.cat((ligand_embed, protein_embed), dim=1)
        transformer_out = self.transformer(combined)
        return self.classifier(transformer_out[:, 0, :]).squeeze()

# Improved Proxy Labels
def generate_proxy_labels(ligand_features):
    mol_wt = ligand_features[:, 0]
    h_donors = ligand_features[:, 1]
    h_acceptors = ligand_features[:, 2]
    logp = ligand_features[:, 4]
    acidic = ligand_features[:, 5]
    basic = ligand_features[:, 6]
    # Pharmacophore-inspired scoring
    binding_score = (0.3 * h_donors + 0.3 * h_acceptors + 0.2 * logp - 
                     0.1 * acidic + 0.1 * basic) / (mol_wt + 1e-6)  # Avoid division by zero
    median_score = np.median(binding_score)
    labels = (binding_score > median_score).astype(int)
    return labels

# Training and Evaluation Pipeline
class BindingTrainer:
    def __init__(self, ligand_dim, protein_dim):
        self.model = EnhancedBindingPredictor(ligand_dim, protein_dim)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', patience=5, factor=0.5)
        self.early_stopping = EarlyStopping(patience=15, min_delta=0.001)
        self.best_auc = 0

    def train_epoch(self, loader, criterion):
        self.model.train()
        total_loss = 0
        for ligand, protein, label in loader:
            self.optimizer.zero_grad()
            outputs = self.model(ligand, protein)
            loss = criterion(outputs, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        outputs, labels = [], []
        for ligand, protein, label in loader:
            preds = torch.sigmoid(self.model(ligand, protein))
            outputs.extend(preds.cpu().numpy())
            labels.extend(label.cpu().numpy())
        return np.array(outputs), np.array(labels)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

# Advanced Protein Feature Extraction
def extract_advanced_cavity_features(pdb_path, residue_numbers):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    target_residues = set(residue_numbers)
    features = []
    cavity_residues = []

    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.get_id()[1]
                if res_id in target_residues:
                    cavity_residues.append(residue)

    if not cavity_residues:
        raise ValueError(f"No residues found matching the provided numbers: {residue_numbers}")

    for res in cavity_residues:
        atoms = list(res.get_atoms())
        if atoms:
            centroid = sum(atom.coord for atom in atoms) / len(atoms)
            distances = [np.linalg.norm(a.coord - centroid) for a in atoms]
            b_factors = [atom.get_bfactor() for atom in atoms]
            features.extend(centroid)  # 3D coordinates
            features.append(np.mean(distances))  # Avg distance
            features.append(np.mean(b_factors))  # Avg flexibility
            features.append(np.std(b_factors))   # Flexibility variation
            features.append(len(atoms))          # Atom count

    if cavity_residues:
        mass_centers = [res.center_of_mass() for res in cavity_residues]
        features.extend(np.mean(mass_centers, axis=0).tolist())  # Mean center
        features.extend(np.std(mass_centers, axis=0).tolist())   # Std deviation
        features.append(len(cavity_residues))                    # Residue count

    if not features:
        raise ValueError("No features extracted from the cavity residues.")

    features = np.array(features, dtype=np.float32).reshape(1, -1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return torch.tensor(scaled_features[0], dtype=torch.float32)

# Load Ligand Features
def load_ligand_features(file_path):
    df = pd.read_excel(file_path)
    ligand_names = df['Ligand Name'].tolist()
    features = df[['mol_wt', 'num_h_donors', 'num_h_acceptors', 
                   'num_rotatable_bonds', 'logp', 'acidic_groups', 
                   'basic_groups', 'aliphatic_hydrophobic', 
                   'aromatic_hydrophobic']].copy()
    
    # Derived features
    features['hbd_hba_ratio'] = (features['num_h_donors'] + 1) / (features['num_h_acceptors'] + 1)
    features['hydrophobic_ratio'] = features['aliphatic_hydrophobic'] / (features['aromatic_hydrophobic'] + 1)
    features['polarity_index'] = (features['num_h_donors'] + features['num_h_acceptors']) / features['mol_wt']
    features['flexibility_index'] = features['num_rotatable_bonds'] / features['mol_wt']
    
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return torch.tensor(scaled_features, dtype=torch.float32), ligand_names

# Molecular Docking (requires AutoDock Vina)
def run_docking(pdb_file, ligand_sdf, output_dir, center, box_size=(20, 20, 20)):
    pdbqt_protein = pdb_file.replace('.pdb', '.pdbqt')
    ligand_pdbqt = ligand_sdf.replace('.sdf', '.pdbqt')
    output_pdbqt = os.path.join(output_dir, ligand_sdf.split('/')[-1].replace('.sdf', '_docked.pdbqt'))

    # Convert PDB to PDBQT (requires Open Babel)
    subprocess.run(['obabel', pdb_file, '-O', pdbqt_protein, '-xr'])
    subprocess.run(['obabel', ligand_sdf, '-O', ligand_pdbqt])

    # Run AutoDock Vina
    cmd = [
        'vina', '--receptor', pdbqt_protein, '--ligand', ligand_pdbqt,
        '--center_x', str(center[0]), '--center_y', str(center[1]), '--center_z', str(center[2]),
        '--size_x', str(box_size[0]), '--size_y', str(box_size[1]), '--size_z', str(box_size[2]),
        '--out', output_pdbqt, '--exhaustiveness', '8'
    ]
    subprocess.run(cmd)
    
    # Parse binding affinity from output
    with open(output_pdbqt, 'r') as f:
        for line in f:
            if 'REMARK VINA RESULT' in line:
                affinity = float(line.split()[3])
                return affinity  # kcal/mol
    return None

# MD Simulation Setup (requires GROMACS)
def run_md_simulation(pdb_file, output_dir, sim_time=100):  # sim_time in ns
    os.makedirs(output_dir, exist_ok=True)
    pdb_clean = os.path.join(output_dir, 'clean.pdb')
    gro_file = os.path.join(output_dir, 'protein.gro')
    top_file = os.path.join(output_dir, 'topol.top')
    traj_file = os.path.join(output_dir, 'traj.xtc')

    # Clean PDB and generate topology
    subprocess.run(['gmx', 'pdb2gmx', '-f', pdb_file, '-o', gro_file, '-p', top_file, '-water', 'spc', '-ff', 'amber99sb'])
    subprocess.run(['gmx', 'editconf', '-f', gro_file, '-o', gro_file, '-c', '-d', '1.0', '-bt', 'cubic'])
    subprocess.run(['gmx', 'solvate', '-cp', gro_file, '-cs', 'spc216.gro', '-o', gro_file, '-p', top_file])
    subprocess.run(['gmx', 'grompp', '-f', 'ions.mdp', '-c', gro_file, '-p', top_file, '-o', 'ions.tpr'])
    subprocess.run(['gmx', 'genion', '-s', 'ions.tpr', '-o', gro_file, '-p', top_file, '-neutral'])

    # Minimization and equilibration
    subprocess.run(['gmx', 'grompp', '-f', 'minim.mdp', '-c', gro_file, '-p', top_file, '-o', 'minim.tpr'])
    subprocess.run(['gmx', 'mdrun', '-v', '-deffnm', 'minim'])
    subprocess.run(['gmx', 'grompp', '-f', 'nvt.mdp', '-c', 'minim.gro', '-p', top_file, '-o', 'nvt.tpr'])
    subprocess.run(['gmx', 'mdrun', '-v', '-deffnm', 'nvt'])
    subprocess.run(['gmx', 'grompp', '-f', 'npt.mdp', '-c', 'nvt.gro', '-p', top_file, '-o', 'npt.tpr'])
    subprocess.run(['gmx', 'mdrun', '-v', '-deffnm', 'npt'])

    # Production run
    subprocess.run(['gmx', 'grompp', '-f', 'md.mdp', '-c', 'npt.gro', '-p', top_file, '-o', 'md.tpr'])
    subprocess.run(['gmx', 'mdrun', '-v', '-deffnm', 'md', '-s', 'md.tpr', '-c', 'md.gro', '-x', traj_file])

    # Extract frames
    subprocess.run(['gmx', 'trjconv', '-s', 'md.tpr', '-f', traj_file, '-o', os.path.join(output_dir, 'frame.pdb'), '-dump', '0', '-pbc', 'mol', '-ur', 'compact'])
    return traj_file

# Visualization with Py3Dmol
def visualize_binding(pdb_file, ligand_pdbqt, residue_numbers):
    view = py3Dmol.view(width=800, height=600)
    with open(pdb_file, 'r') as f:
        view.addModel(f.read(), 'pdb')
    with open(ligand_pdbqt, 'r') as f:
        view.addModel(f.read(), 'pdb')

    view.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum'}})
    view.setStyle({'model': 1}, {'stick': {'colorscheme': 'greenCarbon'}})
    res_sel = [{'resi': res} for res in residue_numbers]
    view.addStyle({'and': res_sel}, {'stick': {'color': 'yellow'}})
    view.zoomTo()
    view.show()
    return view  # Display in Jupyter notebook

# Main Training and Prediction Function
def train_and_predict(ligand_file, pdb_file, cavity_sequence, output_file, docking_dir='docking_results', md_dir='md_results'):
    # Load data
    ligand_features, ligand_names = load_ligand_features(ligand_file)
    protein_feature = extract_advanced_cavity_features(pdb_file, cavity_sequence)
    protein_features = torch.stack([protein_feature] * len(ligand_features))

    # Generate improved proxy labels
    labels = generate_proxy_labels(ligand_features.numpy())
    labels = torch.tensor(labels, dtype=torch.float32)

    # Apply SMOTE for class balance
    smote = SMOTE(random_state=42)
    ligand_features_np, labels_np = smote.fit_resample(ligand_features.numpy(), labels.numpy())
    ligand_features = torch.tensor(ligand_features_np, dtype=torch.float32)
    protein_features = torch.stack([protein_feature] * len(ligand_features))
    labels = torch.tensor(labels_np, dtype=torch.float32)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels.numpy())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights[1] / class_weights[0]))

    # Data splitting
    train_ligands, val_ligands, train_proteins, val_proteins, train_labels, val_labels = \
        train_test_split(ligand_features, protein_features, labels, test_size=0.15, stratify=labels, random_state=42)

    # Create dataloaders
    train_dataset = LigandProteinDataset(train_ligands, train_proteins, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = LigandProteinDataset(val_ligands, val_proteins, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
 
    # Initialize trainer
    trainer = BindingTrainer(ligand_features.shape[1], protein_feature.shape[0])
    best_val_auc = 0

    # Training loop
    for epoch in range(10):  # Increased epochs for better training
        train_loss = trainer.train_epoch(train_loader, criterion)
        val_outputs, val_labels = trainer.evaluate(val_loader)
        val_auc = roc_auc_score(val_labels, val_outputs)
        trainer.scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            trainer.save_model('best_model.pth')
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} (Saved)")
        else:
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

        if trainer.early_stopping(val_auc):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Generate evaluation plots
    plt.figure(figsize=(15, 5))
    fpr, tpr, _ = roc_curve(val_labels, val_outputs)
    roc_auc = auc(fpr, tpr)
    plt.subplot(131)
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}', color='blue', lw=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(val_labels, val_outputs)
    pr_auc = auc(recall, precision)
    plt.subplot(132)
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}', color='red', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    pred_classes = (np.array(val_outputs) > 0.5).astype(int)
    cm = confusion_matrix(val_labels, pred_classes)
    plt.subplot(133)
    plt.imshow(cm, cmap='Blues')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=12)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('performance_metrics.svg', dpi=300)
    plt.show()

    # Generate predictions for all ligands
    trainer.load_model('best_model.pth')
    full_dataset = LigandProteinDataset(ligand_features, protein_features)
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)
    predictions = []
    with torch.no_grad():
        for ligand, protein in full_loader:
            preds = torch.sigmoid(trainer.model(ligand, protein)).cpu().numpy()
            predictions.extend(preds)

    # Save predictions
    results_df = pd.DataFrame({
        'Ligand Name': ligand_names[:len(predictions)],
        'Binding Probability': predictions
    })
    results_df.to_excel(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Docking validation for top ligands
    os.makedirs(docking_dir, exist_ok=True)
    top_ligands = results_df.nlargest(5, 'Binding Probability')
    docking_scores = []
    center = [0, 0, 0]  # Replace with actual cavity center coordinates from PDB
    for idx, row in top_ligands.iterrows():
        ligand_sdf = os.path.join(r"C:\Users\Afsa\OneDrive\Desktop\ciods\Task2\Ligand", row['Ligand Name'])  # Adjust path to your SDF files
        score = run_docking(pdb_file, ligand_sdf, docking_dir, center)
        docking_scores.append(score)
        print(f"Docking score for {row['Ligand Name']}: {score} kcal/mol")

    # MD simulation and visualization
    os.makedirs(md_dir, exist_ok=True)
    traj_file = run_md_simulation(pdb_file, md_dir)
    for idx, row in top_ligands.iterrows():
        ligand_pdbqt = os.path.join(docking_dir, row['Ligand Name'].replace('.sdf', '_docked.pdbqt'))
        output_pse = os.path.join(md_dir, f"{row['Ligand Name'].replace('.sdf', '')}_binding.pse")
        visualize_binding(pdb_file, ligand_pdbqt, cavity_sequence)
        print(f"Visualization saved to {output_pse}")

if __name__ == "__main__":
    train_and_predict(
        ligand_file=r"C:\Users\Afsa\OneDrive\Desktop\ciods\Task2\ligand_pharmacophore_features.xlsx",
        pdb_file=r"C:\Users\Afsa\OneDrive\Desktop\ciods\Task2\monkeypox_DNA_polymerase.pdb",
        cavity_sequence=[44, 150, 233, 236, 156],
        output_file="enhanced_predictions_for_monkeypox.xlsx"
    )