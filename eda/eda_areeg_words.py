"""
==========================================
Analyse Exploratoire des Données (EDA)
Dataset: ArEEG_Words - EEG Recording Dataset
==========================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration pour les graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10


class ArEEGDatasetEDA:
    """Classe pour l'analyse exploratoire du dataset ArEEG_Words"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.word_folders = []
        self.dataset_info = {}
        self.all_data = []
        
        # Canaux EEG disponibles
        self.eeg_channels = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 
                             'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 
                             'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']
        
        # Canaux de qualité
        self.cq_channels = ['CQ.AF3', 'CQ.F7', 'CQ.F3', 'CQ.FC5', 'CQ.T7', 
                            'CQ.P7', 'CQ.O1', 'CQ.O2', 'CQ.P8', 'CQ.T8', 
                            'CQ.FC6', 'CQ.F4', 'CQ.F8', 'CQ.AF4']
        
        # Données de mouvement
        self.mot_channels = ['MOT.Q0', 'MOT.Q1', 'MOT.Q2', 'MOT.Q3', 
                             'MOT.AccX', 'MOT.AccY', 'MOT.AccZ', 
                             'MOT.MagX', 'MOT.MagY', 'MOT.MagZ']
        
        print("=" * 80)
        print("ANALYSE EXPLORATOIRE DES DONNÉES - DATASET ArEEG_Words")
        print("=" * 80)
        print()
    
    def load_dataset_structure(self):
        """Charge la structure du dataset"""
        print("📂 CHARGEMENT DE LA STRUCTURE DU DATASET")
        print("-" * 80)
        
        # Liste tous les dossiers de mots
        self.word_folders = [f for f in self.base_path.iterdir() if f.is_dir()]
        self.word_folders.sort()
        
        print(f"✓ Nombre de mots (classes) trouvés: {len(self.word_folders)}")
        print(f"\n📋 Liste des mots:")
        
        for i, folder in enumerate(self.word_folders, 1):
            csv_files = list(folder.glob("*.csv"))
            self.dataset_info[folder.name] = {
                'path': folder,
                'n_files': len(csv_files),
                'files': csv_files
            }
            print(f"  {i:2d}. {folder.name:20s} - {len(csv_files)} fichiers")
        
        print()
    
    def analyze_dataset_statistics(self):
        """Analyse les statistiques globales du dataset"""
        print("\n" + "=" * 80)
        print("📊 STATISTIQUES GLOBALES DU DATASET")
        print("=" * 80)
        
        total_files = sum([info['n_files'] for info in self.dataset_info.values()])
        
        print(f"\n📈 Statistiques générales:")
        print(f"  • Nombre total de classes (mots): {len(self.word_folders)}")
        print(f"  • Nombre total de fichiers: {total_files}")
        print(f"  • Moyenne de fichiers par classe: {total_files / len(self.word_folders):.1f}")
        
        # Distribution des fichiers par classe
        n_files_per_class = [info['n_files'] for info in self.dataset_info.values()]
        print(f"  • Min fichiers par classe: {min(n_files_per_class)}")
        print(f"  • Max fichiers par classe: {max(n_files_per_class)}")
        print(f"  • Médiane fichiers par classe: {np.median(n_files_per_class):.1f}")
        
        # Visualisation de la distribution
        plt.figure(figsize=(16, 6))
        
        plt.subplot(1, 2, 1)
        words = [folder.name for folder in self.word_folders]
        counts = [self.dataset_info[word]['n_files'] for word in words]
        bars = plt.bar(range(len(words)), counts, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.xlabel('Mot (Classe)', fontsize=12, fontweight='bold')
        plt.ylabel('Nombre de fichiers', fontsize=12, fontweight='bold')
        plt.title('Distribution des fichiers par classe', fontsize=14, fontweight='bold')
        plt.xticks(range(len(words)), words, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(n_files_per_class, vert=True)
        plt.ylabel('Nombre de fichiers', fontsize=12, fontweight='bold')
        plt.title('Distribution statistique des fichiers', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.base_path / 'eda_distribution_fichiers.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Graphique sauvegardé: eda_distribution_fichiers.png")
        plt.close()
    
    def analyze_sample_file(self):
        """Analyse détaillée d'un fichier échantillon"""
        print("\n" + "=" * 80)
        print("🔍 ANALYSE D'UN FICHIER ÉCHANTILLON")
        print("=" * 80)
        
        # Prendre le premier fichier du premier mot
        first_word = self.word_folders[0].name
        sample_file = self.dataset_info[first_word]['files'][0]
        
        print(f"\n📄 Fichier analysé: {sample_file.name}")
        
        # Charger le fichier
        df = pd.read_csv(sample_file, skiprows=1)
        
        print(f"\n📐 Dimensions du fichier:")
        print(f"  • Nombre de lignes (échantillons): {len(df)}")
        print(f"  • Nombre de colonnes: {len(df.columns)}")
        
        print(f"\n📋 Colonnes disponibles ({len(df.columns)} colonnes):")
        
        # Regrouper par type
        col_groups = {
            'Timestamp': [c for c in df.columns if 'Timestamp' in c or 'Counter' in c],
            'EEG Channels': [c for c in df.columns if c.startswith('EEG.') and not any(x in c for x in ['Counter', 'Interpolated', 'RawCq', 'Battery', 'Marker'])],
            'EEG Metadata': [c for c in df.columns if c.startswith('EEG.') and any(x in c for x in ['Counter', 'Interpolated', 'RawCq', 'Battery', 'Marker'])],
            'Contact Quality': [c for c in df.columns if c.startswith('CQ.')],
            'Equipment Quality': [c for c in df.columns if c.startswith('EQ.')],
            'Motion Data': [c for c in df.columns if c.startswith('MOT.')],
            'Markers': [c for c in df.columns if 'Marker' in c and not c.startswith('EEG.')]
        }
        
        for group_name, cols in col_groups.items():
            if cols:
                print(f"\n  {group_name} ({len(cols)} colonnes):")
                for col in cols[:5]:  # Afficher les 5 premières
                    print(f"    - {col}")
                if len(cols) > 5:
                    print(f"    ... et {len(cols) - 5} autres")
        
        print(f"\n📊 Statistiques temporelles:")
        duration = df['Timestamp'].max() - df['Timestamp'].min()
        print(f"  • Durée de l'enregistrement: {duration:.2f} secondes")
        
        if 'EEG.Counter' in df.columns:
            # Calculer la fréquence d'échantillonnage
            time_diffs = df['Timestamp'].diff().dropna()
            avg_sampling_interval = time_diffs.mean()
            sampling_rate = 1 / avg_sampling_interval if avg_sampling_interval > 0 else 0
            print(f"  • Fréquence d'échantillonnage moyenne: {sampling_rate:.2f} Hz")
        
        # Statistiques sur les canaux EEG
        print(f"\n🧠 Statistiques des canaux EEG:")
        eeg_cols = [c for c in self.eeg_channels if c in df.columns]
        
        if eeg_cols:
            eeg_data = df[eeg_cols]
            print(f"  • Nombre de canaux EEG: {len(eeg_cols)}")
            print(f"  • Plage de valeurs globale: [{eeg_data.min().min():.2f}, {eeg_data.max().max():.2f}]")
            print(f"  • Moyenne globale: {eeg_data.mean().mean():.2f}")
            print(f"  • Écart-type global: {eeg_data.std().mean():.2f}")
        
        # Qualité du signal
        print(f"\n✨ Qualité du signal (Contact Quality):")
        cq_cols = [c for c in self.cq_channels if c in df.columns]
        
        if cq_cols:
            cq_data = df[cq_cols]
            print(f"  • Nombre de canaux CQ: {len(cq_cols)}")
            print(f"  • Qualité moyenne par canal (0-4, 4=meilleur):")
            for col in cq_cols:
                if col in df.columns:
                    avg_quality = df[col].mean()
                    print(f"    - {col.replace('CQ.', '')}: {avg_quality:.2f}")
        
        return df
    
    def load_and_aggregate_data(self, max_files_per_class=5):
        """Charge et agrège les données de plusieurs fichiers"""
        print("\n" + "=" * 80)
        print(f"🔄 CHARGEMENT ET AGRÉGATION DES DONNÉES (max {max_files_per_class} fichiers/classe)")
        print("=" * 80)
        
        self.all_data = []
        
        for word in self.word_folders:
            word_name = word.name
            files = self.dataset_info[word_name]['files'][:max_files_per_class]
            
            print(f"\n  Chargement: {word_name} ({len(files)} fichiers)...", end=" ")
            
            word_data = []
            for file_path in files:
                try:
                    df = pd.read_csv(file_path, skiprows=1)
                    df['word'] = word_name
                    df['filename'] = file_path.name
                    df['participant'] = file_path.name.split()[0]  # Extract participant number
                    word_data.append(df)
                except Exception as e:
                    print(f"\n    ⚠ Erreur fichier {file_path.name}: {e}")
            
            if word_data:
                self.all_data.extend(word_data)
                print(f"✓ {len(word_data)} fichiers chargés")
        
        print(f"\n✓ Total: {len(self.all_data)} fichiers chargés")
        return self.all_data
    
    def analyze_eeg_signals(self):
        """Analyse approfondie des signaux EEG"""
        print("\n" + "=" * 80)
        print("🧠 ANALYSE APPROFONDIE DES SIGNAUX EEG")
        print("=" * 80)
        
        if not self.all_data:
            print("⚠ Aucune donnée chargée. Veuillez d'abord charger les données.")
            return
        
        # Combiner toutes les données
        all_df = pd.concat(self.all_data, ignore_index=True)
        
        eeg_cols = [c for c in self.eeg_channels if c in all_df.columns]
        
        if not eeg_cols:
            print("⚠ Aucun canal EEG trouvé.")
            return
        
        print(f"\n📊 Statistiques globales des signaux EEG:")
        
        # Statistiques par canal
        stats_data = []
        for col in eeg_cols:
            channel_name = col.replace('EEG.', '')
            stats_data.append({
                'Canal': channel_name,
                'Moyenne': all_df[col].mean(),
                'Écart-type': all_df[col].std(),
                'Min': all_df[col].min(),
                'Max': all_df[col].max(),
                'Médiane': all_df[col].median()
            })
        
        stats_df = pd.DataFrame(stats_data)
        print("\n" + stats_df.to_string(index=False))
        
        # Visualisation 1: Distribution des amplitudes par canal
        plt.figure(figsize=(18, 10))
        
        # Boxplot
        plt.subplot(2, 2, 1)
        eeg_data = all_df[eeg_cols]
        eeg_data.columns = [c.replace('EEG.', '') for c in eeg_cols]
        eeg_data.boxplot(rot=45)
        plt.ylabel('Amplitude (µV)', fontsize=12, fontweight='bold')
        plt.title('Distribution des amplitudes par canal EEG', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Violin plot pour les premiers 7 canaux
        plt.subplot(2, 2, 2)
        data_to_plot = [all_df[col].dropna() for col in eeg_cols[:7]]
        labels = [c.replace('EEG.', '') for c in eeg_cols[:7]]
        parts = plt.violinplot(data_to_plot, positions=range(len(labels)), showmeans=True, showmedians=True)
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.ylabel('Amplitude (µV)', fontsize=12, fontweight='bold')
        plt.title('Distribution des amplitudes (Violin Plot) - 7 premiers canaux', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Heatmap de corrélation
        plt.subplot(2, 2, 3)
        correlation_matrix = all_df[eeg_cols].corr()
        correlation_matrix.columns = [c.replace('EEG.', '') for c in correlation_matrix.columns]
        correlation_matrix.index = [c.replace('EEG.', '') for c in correlation_matrix.index]
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Matrice de corrélation entre les canaux EEG', fontsize=14, fontweight='bold')
        
        # Moyenne par canal
        plt.subplot(2, 2, 4)
        means = [all_df[col].mean() for col in eeg_cols]
        stds = [all_df[col].std() for col in eeg_cols]
        x_pos = range(len(eeg_cols))
        labels = [c.replace('EEG.', '') for c in eeg_cols]
        plt.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5, color='steelblue', edgecolor='navy')
        plt.xticks(x_pos, labels, rotation=45, ha='right')
        plt.ylabel('Amplitude moyenne (µV)', fontsize=12, fontweight='bold')
        plt.title('Amplitude moyenne par canal EEG (avec écart-type)', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.base_path / 'eda_analyse_signaux_eeg.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Graphique sauvegardé: eda_analyse_signaux_eeg.png")
        plt.close()
        
        # Sauvegarder les statistiques
        stats_df.to_csv(self.base_path / 'eda_statistiques_eeg.csv', index=False)
        print(f"✓ Statistiques sauvegardées: eda_statistiques_eeg.csv")
    
    def analyze_signal_quality(self):
        """Analyse de la qualité du signal"""
        print("\n" + "=" * 80)
        print("✨ ANALYSE DE LA QUALITÉ DU SIGNAL")
        print("=" * 80)
        
        if not self.all_data:
            print("⚠ Aucune donnée chargée.")
            return
        
        all_df = pd.concat(self.all_data, ignore_index=True)
        
        cq_cols = [c for c in self.cq_channels if c in all_df.columns]
        
        if not cq_cols:
            print("⚠ Aucune donnée de qualité trouvée.")
            return
        
        print(f"\n📊 Qualité moyenne par canal (échelle 0-4, 4=meilleur):")
        
        quality_data = []
        for col in cq_cols:
            channel_name = col.replace('CQ.', '')
            avg_quality = all_df[col].mean()
            quality_data.append({
                'Canal': channel_name,
                'Qualité moyenne': avg_quality,
                '% Bonne qualité (≥3)': (all_df[col] >= 3).mean() * 100
            })
        
        quality_df = pd.DataFrame(quality_data)
        print("\n" + quality_df.to_string(index=False))
        
        # Visualisation
        plt.figure(figsize=(16, 6))
        
        plt.subplot(1, 2, 1)
        channels = [d['Canal'] for d in quality_data]
        qualities = [d['Qualité moyenne'] for d in quality_data]
        colors = ['green' if q >= 3 else 'orange' if q >= 2 else 'red' for q in qualities]
        bars = plt.bar(channels, qualities, color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel('Canal EEG', fontsize=12, fontweight='bold')
        plt.ylabel('Qualité moyenne (0-4)', fontsize=12, fontweight='bold')
        plt.title('Qualité moyenne du signal par canal', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=3, color='green', linestyle='--', linewidth=2, label='Seuil bonne qualité')
        plt.axhline(y=2, color='orange', linestyle='--', linewidth=2, label='Seuil qualité moyenne')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(1, 2, 2)
        good_quality_pct = [d['% Bonne qualité (≥3)'] for d in quality_data]
        bars = plt.barh(channels, good_quality_pct, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.xlabel('% Échantillons avec bonne qualité', fontsize=12, fontweight='bold')
        plt.ylabel('Canal EEG', fontsize=12, fontweight='bold')
        plt.title('Pourcentage d\'échantillons avec bonne qualité (≥3)', fontsize=14, fontweight='bold')
        plt.xlim(0, 100)
        plt.grid(axis='x', alpha=0.3)
        
        # Ajouter les valeurs
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2., 
                    f'{width:.1f}%',
                    ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.base_path / 'eda_qualite_signal.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Graphique sauvegardé: eda_qualite_signal.png")
        plt.close()
        
        quality_df.to_csv(self.base_path / 'eda_qualite_signal.csv', index=False)
        print(f"✓ Statistiques sauvegardées: eda_qualite_signal.csv")
    
    def analyze_by_word(self):
        """Analyse comparative par mot"""
        print("\n" + "=" * 80)
        print("📝 ANALYSE COMPARATIVE PAR MOT (CLASSE)")
        print("=" * 80)
        
        if not self.all_data:
            print("⚠ Aucune donnée chargée.")
            return
        
        # Calculer des features pour chaque fichier
        word_features = []
        
        for df in self.all_data:
            word = df['word'].iloc[0]
            participant = df['participant'].iloc[0]
            
            eeg_cols = [c for c in self.eeg_channels if c in df.columns]
            
            if eeg_cols:
                features = {
                    'word': word,
                    'participant': participant,
                    'duration': df['Timestamp'].max() - df['Timestamp'].min(),
                    'n_samples': len(df),
                    'mean_amplitude': df[eeg_cols].mean().mean(),
                    'std_amplitude': df[eeg_cols].std().mean(),
                    'max_amplitude': df[eeg_cols].max().max(),
                    'min_amplitude': df[eeg_cols].min().min()
                }
                
                # Ajouter la qualité moyenne si disponible
                cq_cols = [c for c in self.cq_channels if c in df.columns]
                if cq_cols:
                    features['avg_quality'] = df[cq_cols].mean().mean()
                
                word_features.append(features)
        
        features_df = pd.DataFrame(word_features)
        
        print(f"\n📊 Statistiques par mot:")
        summary = features_df.groupby('word').agg({
            'n_samples': ['mean', 'std', 'min', 'max'],
            'duration': ['mean', 'std'],
            'mean_amplitude': ['mean', 'std'],
            'std_amplitude': ['mean']
        }).round(2)
        
        print("\n" + summary.to_string())
        
        # Visualisation
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1. Durée moyenne par mot
        ax1 = axes[0, 0]
        word_duration = features_df.groupby('word')['duration'].mean().sort_values()
        word_duration.plot(kind='barh', ax=ax1, color='steelblue', edgecolor='navy', alpha=0.7)
        ax1.set_xlabel('Durée moyenne (secondes)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mot', fontsize=12, fontweight='bold')
        ax1.set_title('Durée moyenne d\'enregistrement par mot', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Nombre moyen d'échantillons par mot
        ax2 = axes[0, 1]
        word_samples = features_df.groupby('word')['n_samples'].mean().sort_values()
        word_samples.plot(kind='barh', ax=ax2, color='coral', edgecolor='darkred', alpha=0.7)
        ax2.set_xlabel('Nombre moyen d\'échantillons', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mot', fontsize=12, fontweight='bold')
        ax2.set_title('Nombre moyen d\'échantillons par mot', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Amplitude moyenne par mot
        ax3 = axes[1, 0]
        word_amplitude = features_df.groupby('word')['mean_amplitude'].mean().sort_values()
        word_amplitude.plot(kind='barh', ax=ax3, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        ax3.set_xlabel('Amplitude moyenne (µV)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Mot', fontsize=12, fontweight='bold')
        ax3.set_title('Amplitude EEG moyenne par mot', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Variabilité par mot
        ax4 = axes[1, 1]
        word_std = features_df.groupby('word')['std_amplitude'].mean().sort_values()
        word_std.plot(kind='barh', ax=ax4, color='plum', edgecolor='purple', alpha=0.7)
        ax4.set_xlabel('Écart-type moyen (µV)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Mot', fontsize=12, fontweight='bold')
        ax4.set_title('Variabilité du signal EEG par mot', fontsize=14, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.base_path / 'eda_analyse_par_mot.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Graphique sauvegardé: eda_analyse_par_mot.png")
        plt.close()
        
        # Sauvegarder les features
        features_df.to_csv(self.base_path / 'eda_features_par_mot.csv', index=False)
        summary.to_csv(self.base_path / 'eda_statistiques_par_mot.csv')
        print(f"✓ Features sauvegardés: eda_features_par_mot.csv")
        print(f"✓ Statistiques sauvegardées: eda_statistiques_par_mot.csv")
    
    def analyze_motion_data(self):
        """Analyse des données de mouvement"""
        print("\n" + "=" * 80)
        print("🏃 ANALYSE DES DONNÉES DE MOUVEMENT")
        print("=" * 80)
        
        if not self.all_data:
            print("⚠ Aucune donnée chargée.")
            return
        
        all_df = pd.concat(self.all_data, ignore_index=True)
        
        mot_cols = [c for c in self.mot_channels if c in all_df.columns]
        
        if not mot_cols:
            print("⚠ Aucune donnée de mouvement trouvée.")
            return
        
        # Retirer les colonnes avec trop de NaN
        mot_data = all_df[mot_cols].dropna(how='all', axis=1)
        mot_cols_clean = mot_data.columns.tolist()
        
        if not mot_cols_clean:
            print("⚠ Toutes les colonnes de mouvement contiennent uniquement des NaN.")
            return
        
        print(f"\n📊 Statistiques des capteurs de mouvement:")
        
        motion_stats = []
        for col in mot_cols_clean:
            data = all_df[col].dropna()
            if len(data) > 0:
                motion_stats.append({
                    'Capteur': col.replace('MOT.', ''),
                    'Moyenne': data.mean(),
                    'Écart-type': data.std(),
                    'Min': data.min(),
                    'Max': data.max(),
                    '% Données valides': (len(data) / len(all_df)) * 100
                })
        
        motion_df = pd.DataFrame(motion_stats)
        print("\n" + motion_df.to_string(index=False))
        
        # Visualisation
        if len(mot_cols_clean) >= 4:
            plt.figure(figsize=(16, 10))
            
            # Accéléromètre
            acc_cols = [c for c in mot_cols_clean if 'Acc' in c]
            if acc_cols:
                plt.subplot(2, 2, 1)
                for col in acc_cols:
                    data = all_df[col].dropna()
                    if len(data) > 0:
                        plt.hist(data, bins=50, alpha=0.6, label=col.replace('MOT.', ''))
                plt.xlabel('Valeur', fontsize=12, fontweight='bold')
                plt.ylabel('Fréquence', fontsize=12, fontweight='bold')
                plt.title('Distribution des données d\'accélération', fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Magnétomètre
            mag_cols = [c for c in mot_cols_clean if 'Mag' in c]
            if mag_cols:
                plt.subplot(2, 2, 2)
                for col in mag_cols:
                    data = all_df[col].dropna()
                    if len(data) > 0:
                        plt.hist(data, bins=50, alpha=0.6, label=col.replace('MOT.', ''))
                plt.xlabel('Valeur', fontsize=12, fontweight='bold')
                plt.ylabel('Fréquence', fontsize=12, fontweight='bold')
                plt.title('Distribution des données magnétiques', fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Quaternions
            q_cols = [c for c in mot_cols_clean if 'Q' in c and 'Overall' not in c]
            if q_cols:
                plt.subplot(2, 2, 3)
                q_data = all_df[q_cols].dropna()
                if len(q_data) > 0:
                    q_data.boxplot()
                    plt.ylabel('Valeur', fontsize=12, fontweight='bold')
                    plt.title('Distribution des quaternions (orientation)', fontsize=14, fontweight='bold')
                    plt.xticks([i+1 for i in range(len(q_cols))], 
                              [c.replace('MOT.', '') for c in q_cols])
                    plt.grid(True, alpha=0.3)
            
            # Résumé
            plt.subplot(2, 2, 4)
            sensors = [s['Capteur'] for s in motion_stats]
            pct_valid = [s['% Données valides'] for s in motion_stats]
            colors = ['green' if p > 80 else 'orange' if p > 50 else 'red' for p in pct_valid]
            plt.barh(sensors, pct_valid, color=colors, alpha=0.7, edgecolor='black')
            plt.xlabel('% Données valides', fontsize=12, fontweight='bold')
            plt.ylabel('Capteur', fontsize=12, fontweight='bold')
            plt.title('Complétude des données de mouvement', fontsize=14, fontweight='bold')
            plt.xlim(0, 100)
            plt.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.base_path / 'eda_donnees_mouvement.png', dpi=300, bbox_inches='tight')
            print(f"\n✓ Graphique sauvegardé: eda_donnees_mouvement.png")
            plt.close()
        
        motion_df.to_csv(self.base_path / 'eda_statistiques_mouvement.csv', index=False)
        print(f"✓ Statistiques sauvegardées: eda_statistiques_mouvement.csv")
    
    def generate_summary_report(self):
        """Génère un rapport récapitulatif"""
        print("\n" + "=" * 80)
        print("📄 GÉNÉRATION DU RAPPORT RÉCAPITULATIF")
        print("=" * 80)
        
        report_path = self.base_path / 'EDA_RAPPORT_COMPLET.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(" " * 20 + "RAPPORT D'ANALYSE EXPLORATOIRE - DATASET ArEEG_Words\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("1. INFORMATIONS GÉNÉRALES\n")
            f.write("-" * 100 + "\n")
            f.write(f"   • Dataset: ArEEG_Words - Enregistrements EEG pour mots arabes\n")
            f.write(f"   • Nombre de classes (mots): {len(self.word_folders)}\n")
            
            total_files = sum([info['n_files'] for info in self.dataset_info.values()])
            f.write(f"   • Nombre total de fichiers: {total_files}\n")
            f.write(f"   • Équipement: EPOCX EEG Headset\n")
            f.write(f"   • Canaux EEG: 14 canaux\n\n")
            
            f.write("2. LISTE DES MOTS (CLASSES)\n")
            f.write("-" * 100 + "\n")
            for i, word in enumerate(self.word_folders, 1):
                n_files = self.dataset_info[word.name]['n_files']
                f.write(f"   {i:2d}. {word.name:25s} - {n_files:3d} fichiers\n")
            f.write("\n")
            
            f.write("3. CANAUX EEG\n")
            f.write("-" * 100 + "\n")
            f.write("   Canaux disponibles (14 canaux):\n")
            for i, channel in enumerate(self.eeg_channels, 1):
                f.write(f"   {i:2d}. {channel}\n")
            f.write("\n")
            
            f.write("4. DONNÉES COLLECTÉES\n")
            f.write("-" * 100 + "\n")
            f.write("   • Signaux EEG (14 canaux)\n")
            f.write("   • Qualité du contact (Contact Quality) pour chaque canal\n")
            f.write("   • Qualité de l'équipement (Equipment Quality)\n")
            f.write("   • Données de mouvement (accéléromètre, magnétomètre, quaternions)\n")
            f.write("   • Timestamps et métadonnées\n\n")
            
            f.write("5. FICHIERS GÉNÉRÉS PAR L'ANALYSE\n")
            f.write("-" * 100 + "\n")
            f.write("   Graphiques:\n")
            f.write("   • eda_distribution_fichiers.png - Distribution des fichiers par classe\n")
            f.write("   • eda_analyse_signaux_eeg.png - Analyse détaillée des signaux EEG\n")
            f.write("   • eda_qualite_signal.png - Analyse de la qualité du signal\n")
            f.write("   • eda_analyse_par_mot.png - Comparaison des mots\n")
            f.write("   • eda_donnees_mouvement.png - Analyse des données de mouvement\n\n")
            f.write("   Données CSV:\n")
            f.write("   • eda_statistiques_eeg.csv - Statistiques des canaux EEG\n")
            f.write("   • eda_qualite_signal.csv - Métriques de qualité du signal\n")
            f.write("   • eda_features_par_mot.csv - Features extraites par fichier\n")
            f.write("   • eda_statistiques_par_mot.csv - Statistiques agrégées par mot\n")
            f.write("   • eda_statistiques_mouvement.csv - Statistiques des capteurs de mouvement\n\n")
            
            f.write("6. RECOMMANDATIONS POUR L'ANALYSE ULTÉRIEURE\n")
            f.write("-" * 100 + "\n")
            f.write("   • Prétraitement: Filtrer les signaux EEG (bande de fréquence appropriée)\n")
            f.write("   • Feature Engineering: Extraire des features temporelles et fréquentielles\n")
            f.write("   • Segmentation: Diviser les enregistrements en epochs cohérents\n")
            f.write("   • Classification: Tester différents algorithmes (CNN, LSTM, SVM, etc.)\n")
            f.write("   • Validation: Utiliser une validation croisée appropriée\n")
            f.write("   • Qualité: Filtrer ou pondérer par la qualité du signal\n\n")
            
            f.write("=" * 100 + "\n")
            f.write(" " * 35 + "FIN DU RAPPORT\n")
            f.write("=" * 100 + "\n")
        
        print(f"\n✓ Rapport complet sauvegardé: EDA_RAPPORT_COMPLET.txt")
    
    def run_complete_eda(self, max_files_per_class=5):
        """Exécute l'analyse complète"""
        print("\n" + "🚀" * 40)
        print(" " * 30 + "DÉMARRAGE DE L'EDA COMPLÈTE")
        print("🚀" * 40 + "\n")
        
        # 1. Charger la structure
        self.load_dataset_structure()
        
        # 2. Statistiques globales
        self.analyze_dataset_statistics()
        
        # 3. Analyse d'un fichier échantillon
        self.analyze_sample_file()
        
        # 4. Charger les données
        self.load_and_aggregate_data(max_files_per_class=max_files_per_class)
        
        # 5. Analyse des signaux EEG
        self.analyze_eeg_signals()
        
        # 6. Analyse de la qualité
        self.analyze_signal_quality()
        
        # 7. Analyse par mot
        self.analyze_by_word()
        
        # 8. Analyse du mouvement
        self.analyze_motion_data()
        
        # 9. Rapport final
        self.generate_summary_report()
        
        print("\n" + "=" * 80)
        print("✅ ANALYSE EXPLORATOIRE TERMINÉE AVEC SUCCÈS!")
        print("=" * 80)
        print(f"\n📁 Tous les résultats sont sauvegardés dans: {self.base_path}")
        print("\n📊 Fichiers générés:")
        print("   • 5 graphiques PNG (haute résolution)")
        print("   • 5 fichiers CSV de statistiques")
        print("   • 1 rapport texte complet")
        print("\n" + "🎉" * 40 + "\n")


def main():
    """Fonction principale"""
    # Chemin vers le dossier du dataset
    dataset_path = r"c:\Users\HATIM\Desktop\agor ai\Nouveau dossier (2)\ArEEG_Words\ArEEG_Words\كلمات\كلمات"
    
    # Créer l'instance et lancer l'analyse
    eda = ArEEGDatasetEDA(dataset_path)
    eda.run_complete_eda(max_files_per_class=5)  # Analyser 5 fichiers par classe maximum


if __name__ == "__main__":
    main()
