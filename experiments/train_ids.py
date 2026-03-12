"""
train_ids.py — Training Pipeline IDS (Mahmud et al., 2024)
===========================================================
Modèles : Random Forest, Decision Tree, Gradient Boosting, AdaBoost
Métriques : Accuracy, Precision, Recall, F1, ROC AUC
Figures   : Confusion Matrix, ROC Curve, Comparaison modèles

Usage:
    python experiments/train_ids.py --preprocessed data/preprocessed
    python experiments/train_ids.py --preprocessed data/preprocessed --report
"""

import os, argparse, logging, warnings, time, json
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble    import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree        import DecisionTreeClassifier
from sklearn.metrics     import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix, roc_auc_score,
                                  roc_curve, classification_report)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

MODELS_DIR = 'ml_models'
REPORT_DIR = 'data/reports'


def _banner(title):
    logger.info(f"\n{'─'*60}\n  {title}\n{'─'*60}")

def _savefig(fig, report_dir, name):
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, name)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Figure sauvegardee : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 : Chargement des données preprocessées
# ─────────────────────────────────────────────────────────────────────────────

def step1_load(preprocessed_dir):
    _banner("ÉTAPE 1 — CHARGEMENT DES DONNÉES")

    X_train = np.load(os.path.join(preprocessed_dir, 'X_train.npy'))
    X_test  = np.load(os.path.join(preprocessed_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(preprocessed_dir, 'y_train.npy'))
    y_test  = np.load(os.path.join(preprocessed_dir, 'y_test.npy'))
    le      = joblib.load(os.path.join(preprocessed_dir, 'label_encoder.pkl'))
    feat    = joblib.load(os.path.join(preprocessed_dir, 'feature_cols.pkl'))

    logger.info(f"  X_train : {X_train.shape}")
    logger.info(f"  X_test  : {X_test.shape}")
    logger.info(f"  Classes : {list(le.classes_)}")
    logger.info(f"  Features: {len(feat)}")
    logger.info(f"\n  Distribution y_train :")
    for i, cls in enumerate(le.classes_):
        cnt = (y_train == i).sum()
        logger.info(f"    [{i}] {str(cls):<35} {cnt:>8,}")

    return X_train, X_test, y_train, y_test, le, feat


# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 : Définition des modèles
# ─────────────────────────────────────────────────────────────────────────────

def step2_define_models():
    _banner("ÉTAPE 2 — DÉFINITION DES MODÈLES")

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=None,
            class_weight='balanced',
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        ),
    }

    for name, model in models.items():
        logger.info(f"  {name:<25} → {model.__class__.__name__}")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 : Entraînement + Évaluation
# ─────────────────────────────────────────────────────────────────────────────

def step3_train_evaluate(models, X_train, X_test, y_train, y_test, le):
    _banner("ÉTAPE 3 — ENTRAÎNEMENT & ÉVALUATION")

    results = {}

    for name, model in models.items():
        logger.info(f"\n  ── {name} ──────────────────────────────")

        # Entraînement
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        logger.info(f"  Durée entraînement : {train_time:.1f}s")

        # Prédiction
        y_pred = model.predict(X_test)
        y_prob = None
        try:
            y_prob = model.predict_proba(X_test)
        except:
            pass

        # Métriques
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # ROC AUC (multiclass OVR)
        auc = None
        if y_prob is not None:
            try:
                if len(le.classes_) == 2:
                    auc = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            except:
                pass

        logger.info(f"  Accuracy  : {acc*100:.2f}%")
        logger.info(f"  Precision : {prec:.4f}")
        logger.info(f"  Recall    : {rec:.4f}")
        logger.info(f"  F1 Score  : {f1:.4f}")
        if auc: logger.info(f"  ROC AUC   : {auc:.4f}")

        # Classification report
        report = classification_report(y_test, y_pred,
                                        target_names=[str(c) for c in le.classes_],
                                        zero_division=0)
        logger.info(f"\n{report}")

        results[name] = {
            'model':      model,
            'y_pred':     y_pred,
            'y_prob':     y_prob,
            'accuracy':   acc,
            'precision':  prec,
            'recall':     rec,
            'f1':         f1,
            'auc':        auc,
            'train_time': train_time,
            'cm':         confusion_matrix(y_test, y_pred),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 : Sauvegarde des modèles
# ─────────────────────────────────────────────────────────────────────────────

def step4_save_models(results, models_dir):
    _banner("ÉTAPE 4 — SAUVEGARDE DES MODÈLES")
    os.makedirs(models_dir, exist_ok=True)

    for name, res in results.items():
        fname = name.lower().replace(' ', '_') + '.pkl'
        path  = os.path.join(models_dir, fname)
        joblib.dump(res['model'], path)
        size  = os.path.getsize(path) / 1e6
        logger.info(f"  {fname:<35} {size:.2f} MB")

    # Meilleur modèle (par accuracy)
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_path = os.path.join(models_dir, 'best_model.pkl')
    joblib.dump(results[best_name]['model'], best_path)
    logger.info(f"\n  Meilleur modèle : {best_name} ({results[best_name]['accuracy']*100:.2f}%)")
    logger.info(f"  Sauvegardé      : {best_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 5 : Figures
# ─────────────────────────────────────────────────────────────────────────────

def step5_figures(results, le, report_dir):
    _banner("ÉTAPE 5 — GÉNÉRATION DES FIGURES")

    class_names = [str(c) for c in le.classes_]
    n_classes   = len(class_names)
    model_names = list(results.keys())

    # ── Fig 1 : Confusion Matrix pour chaque modèle ──────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    axes = axes.flatten()
    for ax, (name, res) in zip(axes, results.items()):
        cm = res['cm']
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm, ax=ax, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor='grey',
                    annot_kws={'size': 9})
        ax.set_title(f'{name}\nAccuracy: {res["accuracy"]*100:.2f}%',
                     fontweight='bold', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.tick_params(axis='y', rotation=0,  labelsize=8)
    fig.suptitle('Fig 1 — Confusion Matrix — 4 Modèles', fontsize=15, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, report_dir, 'fig1_confusion_matrices.png')

    # ── Fig 2 : ROC Curves ───────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    axes = axes.flatten()
    colors_roc = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for ax, (name, res), color in zip(axes, results.items(), colors_roc):
        if res['y_prob'] is not None and n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, res['y_prob'][:, 1])
            auc_val = res['auc'] or 0
            ax.plot(fpr, tpr, color=color, lw=2.5,
                    label=f'ROC (AUC = {auc_val:.2f})')
        elif res['y_prob'] is not None:
            # One-vs-Rest pour multiclass
            y_bin = label_binarize(y_test, classes=list(range(n_classes)))
            for i, cls in enumerate(class_names):
                if i < res['y_prob'].shape[1]:
                    try:
                        fpr, tpr, _ = roc_curve(y_bin[:, i], res['y_prob'][:, i])
                        auc_i = roc_auc_score(y_bin[:, i], res['y_prob'][:, i])
                        ax.plot(fpr, tpr, lw=1.8, label=f'{cls} (AUC={auc_i:.2f})')
                    except: pass
        ax.plot([0,1],[0,1],'k--', lw=1, alpha=0.5)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        auc_str = f'{res["auc"]:.2f}' if res['auc'] else 'N/A'
        ax.set_title(f'{name}\nAUC = {auc_str}', fontweight='bold', fontsize=11)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)
    fig.suptitle('Fig 2 — ROC Curves — 4 Modèles', fontsize=15, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, report_dir, 'fig2_roc_curves.png')

    # ── Fig 3 : Comparaison des métriques (Table 1 du paper) ─
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    labels  = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(model_names))
    w = 0.2
    colors_m = ['#3498db', '#e74c3c', '#2ecc71', '#e67e22']

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors_m)):
        vals = [results[n][metric] for n in model_names]
        bars = ax.bar(x + i*w, vals, w, label=label, color=color,
                      edgecolor='black', linewidth=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, rotation=90)

    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Fig 3 — Comparaison des Métriques — 4 Modèles', fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.99, color='red', linestyle='--', alpha=0.4, label='99% threshold')
    plt.tight_layout()
    _savefig(fig, report_dir, 'fig3_metrics_comparison.png')

    # ── Fig 4 : Accuracy bar (comme Table 1 du paper) ────────
    accs  = [results[n]['accuracy']*100 for n in model_names]
    colors_acc = ['#e74c3c' if a == max(accs) else '#3498db' for a in accs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, accs, color=colors_acc, edgecolor='black', linewidth=0.5, width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylim(min(accs)-2, 101)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Fig 4 — Accuracy par Modèle', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    best = model_names[accs.index(max(accs))]
    ax.text(0.98, 0.02, f'Best: {best}\n{max(accs):.2f}%',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#ffeaa7', edgecolor='gray'))
    plt.tight_layout()
    _savefig(fig, report_dir, 'fig4_accuracy_comparison.png')

    # ── Fig 5 : Training Time ─────────────────────────────────
    times = [results[n]['train_time'] for n in model_names]
    colors_t = ['#9b59b6' if t == max(times) else '#1abc9c' for t in times]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, times, color=colors_t, edgecolor='black', linewidth=0.5, width=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{t:.1f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Fig 5 — Temps d\'Entraînement par Modèle', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _savefig(fig, report_dir, 'fig5_training_time.png')

    # ── Fig 6 : Tableau récap (Table 1 du paper) ─────────────
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis('off')
    headers = ['Algorithm', 'Precision', 'Recall', 'F1 Score', 'Accuracy (%)']
    rows = []
    for name in model_names:
        r = results[name]
        rows.append([name, f'{r["precision"]:.4f}', f'{r["recall"]:.4f}',
                     f'{r["f1"]:.4f}', f'{r["accuracy"]*100:.2f}'])

    # Trier par accuracy desc
    rows = sorted(rows, key=lambda x: float(x[4]), reverse=True)

    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Header style
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Meilleure ligne en vert
    best_acc = max(float(r[4]) for r in rows)
    for i, row in enumerate(rows):
        color = '#d5f5e3' if float(row[4]) == best_acc else '#fdfefe'
        for j in range(len(headers)):
            table[i+1, j].set_facecolor(color)

    ax.set_title('Fig 6 — Performance Metrics (Table 1 — Mahmud et al. 2024)',
                 fontweight='bold', fontsize=13, pad=20)
    plt.tight_layout()
    _savefig(fig, report_dir, 'fig6_metrics_table.png')

    # ── Fig 7 : Feature Importance (Random Forest) ───────────
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feat_imp  = rf_model.feature_importances_
        feat_names = joblib.load('data/preprocessed/feature_cols.pkl')
        top_n = 15
        indices = np.argsort(feat_imp)[-top_n:]

        fig, ax = plt.subplots(figsize=(12, 7))
        colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
        bars = ax.barh(range(top_n), feat_imp[indices], color=colors_fi,
                       edgecolor='black', linewidth=0.4)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feat_names[i] for i in indices], fontsize=9)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Fig 7 — Top {top_n} Feature Importances (Random Forest)',
                     fontweight='bold', fontsize=13)
        ax.grid(axis='x', alpha=0.3)
        for bar, val in zip(bars, feat_imp[indices]):
            ax.text(val+0.0005, bar.get_y()+bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)
        plt.tight_layout()
        _savefig(fig, report_dir, 'fig7_feature_importance.png')


# ─────────────────────────────────────────────────────────────────────────────
# Étape 6 : Résumé final
# ─────────────────────────────────────────────────────────────────────────────

def step6_summary(results):
    _banner("ÉTAPE 6 — RÉSUMÉ FINAL")

    logger.info(f"\n  {'MODÈLE':<25} {'ACCURACY':>10} {'PRECISION':>10} {'RECALL':>10} {'F1':>10}")
    logger.info(f"  {'─'*65}")
    for name, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        logger.info(f"  {name:<25} {res['accuracy']*100:>9.2f}% {res['precision']:>10.4f} "
                    f"{res['recall']:>10.4f} {res['f1']:>10.4f}")

    best = max(results, key=lambda k: results[k]['accuracy'])
    logger.info(f"\n  Meilleur modèle : {best}")
    logger.info(f"  Accuracy        : {results[best]['accuracy']*100:.2f}%")
    logger.info(f"  F1 Score        : {results[best]['f1']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed', default='data/preprocessed')
    parser.add_argument('--models-dir',   default=MODELS_DIR)
    parser.add_argument('--report-dir',   default=REPORT_DIR)
    parser.add_argument('--report',       action='store_true')
    args = parser.parse_args()

    global y_test
    t_start = time.time()

    X_train, X_test, y_train, y_test, le, feat = step1_load(args.preprocessed)
    models  = step2_define_models()
    results = step3_train_evaluate(models, X_train, X_test, y_train, y_test, le)
    step4_save_models(results, args.models_dir)

    if args.report:
        step5_figures(results, le, args.report_dir)
        logger.info(f"\n  Figures sauvegardees dans : {args.report_dir}/")
        for f in sorted(os.listdir(args.report_dir)):
            if f.startswith('fig') and f.endswith('.png'):
                logger.info(f"    {f}")

    step6_summary(results)
    _banner(f"TERMINE en {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()