"""
preprocessing.py — Pipeline Data Preprocessing IDS
====================================================
Étapes :
  1.  Chargement du CSV (auto-détection séparateur, encodage)
  2.  Analyse exploratoire : shape, classes, NaN, inf
  3.  Nettoyage : inf → NaN → médiane, doublons, colonnes constantes
  4.  Normalisation des labels (strip, unification des variantes)
  5.  Sélection des features (drop non-numériques, drop redondantes)
  6.  Traitement des outliers (IQR cap ou Winsorize)
  7.  Scaling (StandardScaler ou MinMaxScaler)
  8.  Gestion du déséquilibre de classes (SMOTE ou class_weight)
  9.  Split train/val/test stratifié
  10. Sauvegarde des artefacts preprocessés

Usage:
    python experiments/preprocessing.py --data data/dataset.csv
    python experiments/preprocessing.py --data data/dataset.csv --sample 200000 --report
    python experiments/preprocessing.py --data data/dataset.csv --binary   # BENIGN vs ATTACK
"""

import os
import sys
import time
import logging
import argparse
import warnings
from pathlib import Path

import numpy  as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')   # pas d'affichage GUI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing   import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

LABEL_COL    = 'Class'
BENIGN_LABEL = 'BENIGN'
OUTPUT_DIR   = 'data/preprocessed'
REPORT_DIR   = 'data/reports'

# Colonnes à supprimer (IDs, IPs, timestamps — non pertinents pour la détection)
DROP_COLS = [
    'Flow ID', 'Source IP', 'Src IP', 'Destination IP', 'Dst IP',
    'Source Port', 'Src Port', 'Timestamp', 'ts',
]

# Features CIC-IDS dans l'ordre attendu par le modèle
FEATURE_ORDER = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
    'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
]

# Normalisation des noms de classes (variantes dans différents datasets CIC)
LABEL_ALIASES = {
    # DoS
    'dos hulk':          'DoS-Hulk',
    'doshulk':           'DoS-Hulk',
    'dos goldeneye':     'DoS-GoldenEye',
    'dos slowloris':     'DoS-Slowloris',
    'dos slowhttptest':  'DoS-SlowHTTPTest',
    # DDoS
    'ddos':              'DDoS',
    'ddos attack-hoic':  'DDoS',
    'ddos attack-loic-http': 'DDoS',
    # PortScan
    'portscan':          'PortScan',
    'port scan':         'PortScan',
    # Brute Force
    'ftp-patator':       'BruteForce-FTP',
    'ssh-patator':       'BruteForce-SSH',
    'brute force':       'BruteForce',
    'bruteforce':        'BruteForce',
    # Web Attacks
    'web attack \x96 brute force': 'WebAttack-BruteForce',
    'web attack – brute force':    'WebAttack-BruteForce',
    'web attack - brute force':    'WebAttack-BruteForce',
    'web attack \x96 xss':         'WebAttack-XSS',
    'web attack – xss':            'WebAttack-XSS',
    'web attack - xss':            'WebAttack-XSS',
    'web attack \x96 sql injection': 'WebAttack-SQL',
    'web attack – sql injection':    'WebAttack-SQL',
    'web attack - sql injection':    'WebAttack-SQL',
    # Botnet
    'bot':               'Botnet',
    # Infiltration
    'infiltration':      'Infiltration',
    # Benign
    'benign':            'BENIGN',
    'normal':            'BENIGN',
}


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 : Chargement
# ─────────────────────────────────────────────────────────────────────────────

def step1_load(path: str, sample: int = None) -> pd.DataFrame:
    _banner("ÉTAPE 1 — CHARGEMENT")
    t0 = time.time()

    # Auto-détection du séparateur
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        first_line = f.readline()
    sep = ',' if first_line.count(',') >= first_line.count(';') else ';'
    logger.info(f"  Séparateur détecté : '{sep}'")

    df = pd.read_csv(path, sep=sep, low_memory=False,
                     encoding='utf-8', on_bad_lines='skip')

    # Nettoyer noms de colonnes
    df.columns = df.columns.str.strip()

    logger.info(f"  Fichier     : {path}")
    logger.info(f"  Shape       : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    logger.info(f"  Mémoire     : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    logger.info(f"  Durée load  : {time.time()-t0:.1f}s")

    if sample and len(df) > sample:
        # Échantillonnage stratifié si label présent
        label = _find_label_col(df)
        if label:
            df = df.groupby(label, group_keys=False).apply(
                lambda g: g.sample(min(len(g), max(1, int(sample * len(g) / len(df)))),
                                   random_state=42)
            )
        else:
            df = df.sample(n=sample, random_state=42)
        df = df.reset_index(drop=True)
        logger.info(f"  → Échantillonné à {len(df):,} lignes (stratifié)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 : Analyse exploratoire
# ─────────────────────────────────────────────────────────────────────────────

def step2_eda(df: pd.DataFrame, report_dir: str = None) -> dict:
    _banner("ÉTAPE 2 — ANALYSE EXPLORATOIRE (EDA)")

    stats = {}

    # Colonnes
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    logger.info(f"  Colonnes numériques : {len(num_cols)}")
    logger.info(f"  Colonnes texte      : {len(obj_cols)} → {obj_cols}")

    # NaN
    nan_total = df.isna().sum().sum()
    nan_cols  = df.columns[df.isna().any()].tolist()
    logger.info(f"  NaN total           : {nan_total:,}")
    if nan_cols:
        logger.info(f"  Colonnes avec NaN   : {nan_cols}")

    # Infinis
    inf_mask  = df[num_cols].isin([np.inf, -np.inf])
    inf_total = inf_mask.sum().sum()
    inf_cols  = [c for c in num_cols if inf_mask[c].any()] if len(num_cols) > 0 else []
    logger.info(f"  Inf total           : {inf_total:,}")
    if inf_cols:
        logger.info(f"  Colonnes avec Inf   : {inf_cols}")

    # Doublons
    dup = df.duplicated().sum()
    logger.info(f"  Doublons            : {dup:,}  ({100*dup/len(df):.1f}%)")

    # Distribution des classes
    label = _find_label_col(df)
    if label:
        dist = df[label].value_counts()
        stats['class_dist'] = dist.to_dict()
        stats['n_classes']  = len(dist)
        logger.info(f"\n  {'CLASSE':<30} {'COUNT':>10}  {'%':>7}  {'BAR'}")
        logger.info(f"  {'-'*65}")
        for cls, cnt in dist.items():
            pct = 100 * cnt / len(df)
            bar = '█' * max(1, int(30 * cnt / dist.max()))
            logger.info(f"  {str(cls):<30} {cnt:>10,}  {pct:>6.1f}%  {bar}")

        # Imbalance ratio
        ratio = dist.max() / dist.min()
        logger.info(f"\n  Imbalance ratio     : {ratio:.1f}x  "
                    f"({'fort déséquilibre' if ratio > 10 else 'modéré' if ratio > 3 else 'équilibré'})")
        stats['imbalance_ratio'] = round(ratio, 2)

        # Graphique distribution
        if report_dir:
            _plot_class_dist(dist, report_dir)

    stats.update({
        'n_rows': len(df), 'n_cols': len(df.columns),
        'nan_total': int(nan_total), 'inf_total': int(inf_total),
        'duplicates': int(dup), 'num_features': len(num_cols),
    })
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 : Nettoyage
# ─────────────────────────────────────────────────────────────────────────────

def step3_clean(df: pd.DataFrame) -> pd.DataFrame:
    _banner("ÉTAPE 3 — NETTOYAGE")
    n0 = len(df)

    # 3.1 Supprimer doublons
    df = df.drop_duplicates()
    logger.info(f"  Doublons supprimés  : {n0 - len(df):,}  → {len(df):,} lignes restantes")

    # 3.2 Remplacer inf par NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_inf    = df[num_cols].isin([np.inf, -np.inf]).sum().sum()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    logger.info(f"  Inf → NaN           : {n_inf:,} valeurs")

    # 3.3 Remplacer NaN par médiane (par colonne)
    n_nan = df[num_cols].isna().sum().sum()
    for col in num_cols:
        if df[col].isna().any():
            median = df[col].median()
            df[col].fillna(median, inplace=True)
    logger.info(f"  NaN → médiane       : {n_nan:,} valeurs")

    # 3.4 Supprimer colonnes constantes (variance = 0)
    const_cols = [c for c in num_cols if df[c].nunique() <= 1]
    if const_cols:
        df.drop(columns=const_cols, inplace=True)
        logger.info(f"  Colonnes constantes supprimées : {const_cols}")

    # 3.5 Supprimer colonnes DROP_COLS si présentes
    to_drop = [c for c in DROP_COLS if c in df.columns]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        logger.info(f"  Colonnes ID/IP supprimées : {to_drop}")

    # 3.6 Supprimer colonnes redondantes (Fwd Header Length.1 = doublon)
    if 'Fwd Header Length.1' in df.columns and 'Fwd Header Length' in df.columns:
        corr = df['Fwd Header Length.1'].corr(df['Fwd Header Length'])
        if corr > 0.99:
            df.drop(columns=['Fwd Header Length.1'], inplace=True)
            logger.info(f"  'Fwd Header Length.1' supprimée (r={corr:.3f} avec 'Fwd Header Length')")

    logger.info(f"  Shape final         : {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 : Normalisation des labels
# ─────────────────────────────────────────────────────────────────────────────

def step4_labels(df: pd.DataFrame, binary: bool = False) -> pd.DataFrame:
    _banner("ÉTAPE 4 — NORMALISATION DES LABELS")

    label = _find_label_col(df)
    if not label:
        logger.warning("  ⚠ Colonne label non trouvée — skip")
        return df

    if label != LABEL_COL:
        df.rename(columns={label: LABEL_COL}, inplace=True)
        logger.info(f"  Renommé '{label}' → '{LABEL_COL}'")

    # Strip + lowercase pour matching
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

    # Appliquer les alias
    def normalize(val):
        v = val.strip().lower()
        return LABEL_ALIASES.get(v, val.strip())

    before = df[LABEL_COL].nunique()
    df[LABEL_COL] = df[LABEL_COL].apply(normalize)
    after  = df[LABEL_COL].nunique()
    logger.info(f"  Classes avant normalisation : {before}")
    logger.info(f"  Classes après              : {after}")
    logger.info(f"  Classes : {sorted(df[LABEL_COL].unique())}")

    if binary:
        df[LABEL_COL] = df[LABEL_COL].apply(
            lambda x: BENIGN_LABEL if x == BENIGN_LABEL else 'ATTACK')
        logger.info(f"  Mode binaire : BENIGN vs ATTACK")
        logger.info(f"  {df[LABEL_COL].value_counts().to_dict()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Étape 5 : Sélection des features
# ─────────────────────────────────────────────────────────────────────────────

def step5_features(df: pd.DataFrame) -> tuple:
    _banner("ÉTAPE 5 — SÉLECTION DES FEATURES")

    # Features disponibles dans le dataset (dans l'ordre FEATURE_ORDER)
    available   = [c for c in FEATURE_ORDER if c in df.columns]
    extra_num   = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in FEATURE_ORDER and c != LABEL_COL]
    missing     = [c for c in FEATURE_ORDER if c not in df.columns]

    logger.info(f"  Features CIC-IDS attendues : {len(FEATURE_ORDER)}")
    logger.info(f"  Features disponibles       : {len(available)}")
    logger.info(f"  Features supplémentaires   : {len(extra_num)}  → {extra_num[:5]}")
    if missing:
        logger.info(f"  Features manquantes        : {len(missing)} → {missing[:10]}")

    # Utiliser toutes les features numériques disponibles (ordre CIC en priorité)
    feature_cols = available + extra_num
    feature_cols = [c for c in feature_cols if c != LABEL_COL]

    X = df[feature_cols].copy()
    y = df[LABEL_COL].copy() if LABEL_COL in df.columns else None

    logger.info(f"  Shape X : {X.shape}")
    if y is not None:
        logger.info(f"  Shape y : {y.shape}  classes={y.nunique()}")

    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Étape 6 : Traitement des outliers (IQR Winsorizing)
# ─────────────────────────────────────────────────────────────────────────────

def step6_outliers(X: pd.DataFrame, method: str = 'iqr',
                   report_dir: str = None) -> tuple:
    _banner("ÉTAPE 6 — TRAITEMENT DES OUTLIERS")

    outlier_stats = {}

    if method == 'iqr':
        # Winsorizing IQR : cap à Q1 - 1.5*IQR et Q3 + 1.5*IQR
        # (moins agressif que suppression — conserve les lignes)
        total_capped = 0
        for col in X.columns:
            q1  = X[col].quantile(0.01)
            q99 = X[col].quantile(0.99)
            n_before = ((X[col] < q1) | (X[col] > q99)).sum()
            X[col] = X[col].clip(lower=q1, upper=q99)
            total_capped += n_before
            if n_before > 0:
                outlier_stats[col] = int(n_before)

        logger.info(f"  Méthode         : IQR Winsorizing (percentile 1% – 99%)")
        logger.info(f"  Valeurs cappées : {total_capped:,}")
        top5 = sorted(outlier_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"  Top 5 colonnes avec outliers : {top5}")

    elif method == 'none':
        logger.info("  Outliers non traités (method='none')")

    return X, outlier_stats


# ─────────────────────────────────────────────────────────────────────────────
# Étape 7 : Scaling
# ─────────────────────────────────────────────────────────────────────────────

def step7_scaling(X_train: pd.DataFrame, X_val: pd.DataFrame,
                  X_test: pd.DataFrame, method: str = 'standard') -> tuple:
    _banner("ÉTAPE 7 — SCALING")

    if method == 'standard':
        scaler = StandardScaler()
        logger.info("  Méthode : StandardScaler (mean=0, std=1)")
    elif method == 'minmax':
        scaler = MinMaxScaler()
        logger.info("  Méthode : MinMaxScaler (range [0,1])")
    else:
        raise ValueError(f"Méthode inconnue: {method}")

    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    logger.info(f"  Fit sur train : {X_train.shape}")
    logger.info(f"  Transform val : {X_val.shape}")
    logger.info(f"  Transform test: {X_test.shape}")
    logger.info(f"  Mean train (après) ≈ {X_train_sc.mean():.4f} (attendu ≈ 0)")
    logger.info(f"  Std  train (après) ≈ {X_train_sc.std():.4f}  (attendu ≈ 1)")

    return X_train_sc, X_val_sc, X_test_sc, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Étape 8 : Encodage des labels + split
# ─────────────────────────────────────────────────────────────────────────────

def step8_encode_split(X: pd.DataFrame, y: pd.Series,
                       test_size: float = 0.2,
                       val_size:  float = 0.1) -> tuple:
    _banner("ÉTAPE 8 — ENCODAGE LABELS + SPLIT TRAIN/VAL/TEST")

    # Encoder labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    logger.info(f"  Classes encodées :")
    for i, cls in enumerate(le.classes_):
        cnt = (y_enc == i).sum()
        logger.info(f"    [{i}] {cls:<30} {cnt:>8,}")

    # Split stratifié
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_enc, test_size=test_size,
        random_state=42, stratify=y_enc)

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio,
        random_state=42, stratify=y_trainval)

    logger.info(f"\n  Train : {len(X_train):>8,}  ({100*len(X_train)/len(X):.1f}%)")
    logger.info(f"  Val   : {len(X_val):>8,}  ({100*len(X_val)/len(X):.1f}%)")
    logger.info(f"  Test  : {len(X_test):>8,}  ({100*len(X_test)/len(X):.1f}%)")

    # Vérification distribution par split
    logger.info(f"\n  Distribution des classes par split :")
    logger.info(f"  {'Classe':<25} {'Train':>10} {'Val':>10} {'Test':>10}")
    logger.info(f"  {'-'*57}")
    for i, cls in enumerate(le.classes_):
        t = (y_train == i).sum()
        v = (y_val   == i).sum()
        te= (y_test  == i).sum()
        logger.info(f"  {cls:<25} {t:>10,} {v:>10,} {te:>10,}")

    return X_train, X_val, X_test, y_train, y_val, y_test, le


# ─────────────────────────────────────────────────────────────────────────────
# Étape 9 : Sauvegarde
# ─────────────────────────────────────────────────────────────────────────────

def step9_save(X_train_sc, X_val_sc, X_test_sc,
               y_train, y_val, y_test,
               scaler, label_enc, feature_cols,
               eda_stats: dict, output_dir: str = OUTPUT_DIR):
    _banner("ÉTAPE 9 — SAUVEGARDE")

    os.makedirs(output_dir, exist_ok=True)

    # Arrays numpy
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_sc)
    np.save(os.path.join(output_dir, 'X_val.npy'),   X_val_sc)
    np.save(os.path.join(output_dir, 'X_test.npy'),  X_test_sc)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'),   y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'),  y_test)

    # Artefacts sklearn
    joblib.dump(scaler,       os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(label_enc,    os.path.join(output_dir, 'label_encoder.pkl'))
    joblib.dump(feature_cols, os.path.join(output_dir, 'feature_cols.pkl'))

    # Résumé texte
    summary = _build_summary(X_train_sc, X_val_sc, X_test_sc,
                              y_train, y_val, y_test,
                              label_enc, feature_cols, eda_stats)
    with open(os.path.join(output_dir, 'preprocessing_report.txt'), 'w') as f:
        f.write(summary)

    logger.info(f"  Sauvegardé dans : {output_dir}/")
    for fname in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, fname))
        logger.info(f"    {fname:<35} {size/1e6:>7.2f} MB")

    logger.info(f"\n  ✅ Preprocessing terminé.")
    logger.info(f"  → Pour charger :")
    logger.info(f"     X_train = np.load('{output_dir}/X_train.npy')")
    logger.info(f"     y_train = np.load('{output_dir}/y_train.npy')")
    logger.info(f"     le      = joblib.load('{output_dir}/label_encoder.pkl')")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────────────────

def _find_label_col(df: pd.DataFrame) -> str:
    candidates = [LABEL_COL, 'label', 'Label', 'class', 'attack',
                  'Attack', 'Category', 'category']
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _banner(title: str):
    logger.info("")
    logger.info("─" * 60)
    logger.info(f"  {title}")
    logger.info("─" * 60)


def _plot_class_dist(dist: pd.Series, report_dir: str):
    try:
        os.makedirs(report_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        colors = ['#e74c3c' if k != BENIGN_LABEL else '#2ecc71' for k in dist.index]
        dist.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black', linewidth=0.5)
        axes[0].set_title('Distribution des classes', fontweight='bold')
        axes[0].set_xlabel('Classe')
        axes[0].set_ylabel('Nombre de samples')
        axes[0].tick_params(axis='x', rotation=45)
        for p in axes[0].patches:
            axes[0].annotate(f'{int(p.get_height()):,}',
                             (p.get_x() + p.get_width()/2., p.get_height()),
                             ha='center', va='bottom', fontsize=8)

        # Pie chart
        axes[1].pie(dist.values, labels=dist.index, autopct='%1.1f%%',
                    colors=plt.cm.Set3.colors[:len(dist)])
        axes[1].set_title('Proportions', fontweight='bold')

        plt.tight_layout()
        path = os.path.join(report_dir, 'class_distribution.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        logger.info(f"  📊 Graphique sauvegardé : {path}")
    except Exception as e:
        logger.debug(f"  Graphique non généré : {e}")


def _build_summary(X_train, X_val, X_test,
                   y_train, y_val, y_test,
                   le, feature_cols, eda_stats) -> str:
    lines = [
        "=" * 60,
        "  PREPROCESSING REPORT",
        f"  Date : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"Dataset original   : {eda_stats.get('n_rows', '?'):,} lignes",
        f"NaN nettoyés       : {eda_stats.get('nan_total', 0):,}",
        f"Inf nettoyés       : {eda_stats.get('inf_total', 0):,}",
        f"Doublons supprimés : {eda_stats.get('duplicates', 0):,}",
        f"Imbalance ratio    : {eda_stats.get('imbalance_ratio', '?')}x",
        "",
        f"Features utilisées : {len(feature_cols)}",
        "",
        f"Split :",
        f"  Train : {len(y_train):,}",
        f"  Val   : {len(y_val):,}",
        f"  Test  : {len(y_test):,}",
        "",
        f"Classes ({len(le.classes_)}) :",
    ]
    for i, cls in enumerate(le.classes_):
        lines.append(f"  [{i}] {cls}")
    lines += [
        "",
        "Fichiers générés :",
        "  X_train.npy  X_val.npy  X_test.npy",
        "  y_train.npy  y_val.npy  y_test.npy",
        "  scaler.pkl   label_encoder.pkl  feature_cols.pkl",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Preprocessing IDS — CIC-IDS format')
    parser.add_argument('--data',       required=True,  help='Chemin vers le CSV')
    parser.add_argument('--sample',     type=int,       help='Nb lignes à échantillonner')
    parser.add_argument('--binary',     action='store_true', help='BENIGN vs ATTACK (binaire)')
    parser.add_argument('--report',     action='store_true', help='Générer les graphiques')
    parser.add_argument('--scaler',     default='standard', choices=['standard', 'minmax'])
    parser.add_argument('--outliers',   default='iqr',  choices=['iqr', 'none'])
    parser.add_argument('--test-size',  type=float, default=0.20)
    parser.add_argument('--val-size',   type=float, default=0.10)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    args = parser.parse_args()

    report_dir = REPORT_DIR if args.report else None

    t_start = time.time()

    # ── Pipeline ──────────────────────────────────────────────────────
    df         = step1_load(args.data, sample=args.sample)
    eda_stats  = step2_eda(df, report_dir=report_dir)
    df         = step3_clean(df)
    df         = step4_labels(df, binary=args.binary)
    X, y, feat = step5_features(df)
    X, _       = step6_outliers(X, method=args.outliers, report_dir=report_dir)

    (X_train, X_val, X_test,
     y_train, y_val, y_test, le) = step8_encode_split(
         X, y, test_size=args.test_size, val_size=args.val_size)

    (X_train_sc, X_val_sc,
     X_test_sc, scaler) = step7_scaling(
         pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test),
         method=args.scaler)

    step9_save(X_train_sc, X_val_sc, X_test_sc,
               y_train, y_val, y_test,
               scaler, le, feat, eda_stats,
               output_dir=args.output_dir)

    _banner(f"TERMINÉ en {time.time()-t_start:.1f}s")
    logger.info(f"  Prochaine étape : python experiments/train_ids.py "
                f"--preprocessed {args.output_dir}")


if __name__ == '__main__':
    main()
