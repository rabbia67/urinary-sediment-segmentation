"""
09_seg_results_generator.py
============================
Generate All LaTeX Tables and Publication Figures

10 LaTeX Tables:
1. Dataset characteristics
2. Class distribution
3. U-Net hyperparameters
4. SegFormer hyperparameters
5. Segmentation metrics comparison
6. Feature descriptions
7. Classification comparison (6 experiments)
8. SHAP feature rankings
9. ODE parameters
10. ODE simulation outcomes

Author: UTI Prediction Team
Date: 2024
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.latex_utils import df_to_latex, save_latex_table, format_metric

RESULTS_DIR = PROJECT_ROOT / "outputs" / "seg_results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "seg_figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "seg_tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

from utils.seg_data_utils import CLASS_NAMES


def generate_dataset_table():
    """Table 1: Dataset Characteristics."""
    with open(RESULTS_DIR / 'dataset_summary.json', 'r') as f:
        summary = json.load(f)

    splits = summary.get('splits', {})
    size = summary.get('image_size', {})
    data = [
        {'Characteristic': 'Dataset', 'Value': 'Urinary Sediment (Liang et al.)'},
        {'Characteristic': 'Total Images', 'Value': str(summary.get('total_images', ''))},
        {'Characteristic': 'Image Dimensions', 'Value': f"{size.get('width', '')} x {size.get('height', '')} pixels"},
        {'Characteristic': 'Format', 'Value': 'JPEG, masks from VOC boxes'},
        {'Characteristic': 'Train / Val / Test', 'Value': f"{splits.get('train', '')} / {splits.get('val', '')} / {splits.get('test', '')}"},
        {'Characteristic': 'Segmentation Classes', 'Value': '8 (1 background + 7 particle classes)'},
        {'Characteristic': 'Patch Size', 'Value': f"{summary.get('patch_size', 512)} x {summary.get('patch_size', 512)} pixels"},
        {'Characteristic': 'Severity Classes', 'Value': '3 (Mild / Moderate / Severe)'},
    ]
    df = pd.DataFrame(data)
    latex = df_to_latex(df, caption='Dataset Characteristics', label='tab:dataset')
    save_latex_table(latex, TABLES_DIR, 'table01_dataset')
    print("  Table 1: Dataset Characteristics")


def generate_class_distribution_table():
    """Table 2: Per-class distribution."""
    with open(RESULTS_DIR / 'class_statistics.json', 'r') as f:
        stats = json.load(f)

    n_images = sum(s['n_images_present'] for s in stats.values() if s.get('class_id') == 0)
    data = []
    for cls_id in range(1, 8):
        cls_name = CLASS_NAMES[cls_id]
        s = stats[cls_name]
        data.append({
            'Class': cls_name,
            'Total Objects': f"{s['total_objects']:,}",
            'Total Pixels': f"{s['total_pixels']:,}",
            'Images Present': f"{s['n_images_present']}/{n_images}",
            'Mean Objects/Image': f"{s['mean_objects_per_image']:.1f}",
        })

    df = pd.DataFrame(data)
    latex = df_to_latex(df, caption='Class Distribution in the Urinary Sediment Dataset',
                        label='tab:class_dist')
    save_latex_table(latex, TABLES_DIR, 'table02_class_distribution')
    print("  Table 2: Class Distribution")


def generate_unet_hyperparams_table():
    """Table 3: U-Net Hyperparameters."""
    with open(RESULTS_DIR / 'unet_training_results.json', 'r') as f:
        results = json.load(f)

    hp = results['hyperparameters']
    data = [
        {'Parameter': 'Encoder', 'Value': 'ResNet-34 (ImageNet pretrained)'},
        {'Parameter': 'Input Size', 'Value': '512 x 512 x 3'},
        {'Parameter': 'Output Classes', 'Value': '8'},
        {'Parameter': 'Learning Rate', 'Value': str(hp['learning_rate'])},
        {'Parameter': 'Encoder LR Multiplier', 'Value': str(hp['encoder_lr_multiplier'])},
        {'Parameter': 'Encoder Unfreeze Epoch', 'Value': str(hp['encoder_unfreeze_epoch'])},
        {'Parameter': 'Weight Decay', 'Value': str(hp['weight_decay'])},
        {'Parameter': 'Scheduler', 'Value': hp['scheduler']},
        {'Parameter': 'Loss Function', 'Value': hp['loss']},
        {'Parameter': 'Batch Size', 'Value': str(results['batch_size'])},
        {'Parameter': 'Total Parameters', 'Value': f"{results['total_parameters']:,}"},
    ]

    df = pd.DataFrame(data)
    latex = df_to_latex(df, caption='U-Net (ResNet-34) Hyperparameters',
                        label='tab:unet_hp', column_format='ll')
    save_latex_table(latex, TABLES_DIR, 'table03_unet_hyperparams')
    print("  Table 3: U-Net Hyperparameters")


def generate_segformer_hyperparams_table():
    """Table 4: SegFormer Hyperparameters."""
    with open(RESULTS_DIR / 'segformer_training_results.json', 'r') as f:
        results = json.load(f)

    hp = results['hyperparameters']
    data = [
        {'Parameter': 'Architecture', 'Value': 'SegFormer-B2'},
        {'Parameter': 'Pretrained', 'Value': 'ADE-20K'},
        {'Parameter': 'Input Size', 'Value': '512 x 512 x 3'},
        {'Parameter': 'Output Classes', 'Value': '8'},
        {'Parameter': 'Learning Rate', 'Value': str(hp['learning_rate'])},
        {'Parameter': 'Warmup Epochs', 'Value': str(hp['warmup_epochs'])},
        {'Parameter': 'Scheduler', 'Value': hp['scheduler']},
        {'Parameter': 'Loss Function', 'Value': hp['loss']},
        {'Parameter': 'Batch Size', 'Value': str(results['batch_size'])},
        {'Parameter': 'Total Parameters', 'Value': f"{results['total_parameters']:,}"},
    ]

    df = pd.DataFrame(data)
    latex = df_to_latex(df, caption='SegFormer-B2 Hyperparameters',
                        label='tab:segformer_hp', column_format='ll')
    save_latex_table(latex, TABLES_DIR, 'table04_segformer_hyperparams')
    print("  Table 4: SegFormer Hyperparameters")


def generate_segmentation_metrics_table():
    """Table 5: Segmentation metrics comparison."""
    metrics_df = pd.read_csv(RESULTS_DIR / 'segmentation_metrics.csv')

    with open(RESULTS_DIR / 'segmentation_evaluation.json', 'r') as f:
        eval_results = json.load(f)

    # Summary row
    summary_data = [
        {
            'Model': 'U-Net (ResNet-34)',
            'Mean Dice': f"{eval_results['unet']['mean_dice']:.4f} $\\pm$ {eval_results['unet']['std_dice']:.4f}",
            'Binary Dice': f"{eval_results['unet']['mean_binary_dice']:.4f}",
        },
        {
            'Model': 'SegFormer-B2',
            'Mean Dice': f"{eval_results['segformer']['mean_dice']:.4f} $\\pm$ {eval_results['segformer']['std_dice']:.4f}",
            'Binary Dice': f"{eval_results['segformer']['mean_binary_dice']:.4f}",
        },
    ]
    summary_df = pd.DataFrame(summary_data)

    latex = df_to_latex(summary_df, caption='Segmentation Performance Comparison',
                        label='tab:seg_metrics', column_format='lcc')
    save_latex_table(latex, TABLES_DIR, 'table05_segmentation_metrics')
    print("  Table 5: Segmentation Metrics")


def generate_feature_descriptions_table():
    """Table 6: Feature descriptions."""
    data = [
        {'Feature Group': 'Per-class (x7)', 'Feature': '{class}\\_count', 'Description': 'Connected component count'},
        {'Feature Group': '', 'Feature': '{class}\\_total\\_area', 'Description': 'Total pixel area'},
        {'Feature Group': '', 'Feature': '{class}\\_mean\\_area', 'Description': 'Mean object area'},
        {'Feature Group': '', 'Feature': '{class}\\_max\\_area', 'Description': 'Largest object area'},
        {'Feature Group': 'Cross-class', 'Feature': 'total\\_cell\\_count', 'Description': 'Sum of all cell counts'},
        {'Feature Group': '', 'Feature': 'cell\\_density', 'Description': 'Cells per pixel'},
        {'Feature Group': '', 'Feature': 'foreground\\_ratio', 'Description': 'Non-background pixel ratio'},
        {'Feature Group': '', 'Feature': 'mycete\\_cell\\_ratio', 'Description': 'Mycete area / red and white cell area'},
        {'Feature Group': '', 'Feature': 'epithelial\\_combined\\_area', 'Description': 'Epithelial cell plus nucleus area'},
        {'Feature Group': '', 'Feature': 'epithelial\\_nucleus\\_ratio', 'Description': 'Nucleus area / epithelial area'},
        {'Feature Group': '', 'Feature': 'infection\\_signature', 'Description': 'Mycete area / foreground'},
        {'Feature Group': '', 'Feature': 'inflammation\\_score', 'Description': 'Red and white cell area / foreground'},
    ]
    df = pd.DataFrame(data)
    latex = df_to_latex(df, caption='Extracted Tabular Features (36 Total)',
                        label='tab:features', column_format='lll')
    save_latex_table(latex, TABLES_DIR, 'table06_feature_descriptions')
    print("  Table 6: Feature Descriptions")


def generate_classification_comparison_table():
    """Table 7: Classification comparison (6 experiments)."""
    cls_df = pd.read_csv(RESULTS_DIR / 'classification_comparison.csv')
    latex = df_to_latex(cls_df,
                        caption='Severity Classification: 6 Experiments Comparison',
                        label='tab:classification')
    save_latex_table(latex, TABLES_DIR, 'table07_classification_comparison')
    print("  Table 7: Classification Comparison")


def generate_shap_ranking_table():
    """Table 8: SHAP feature importance rankings."""
    shap_df = pd.read_csv(RESULTS_DIR / 'shap_importance_ranking.csv')
    top10 = shap_df.head(10)
    latex = df_to_latex(top10, caption='Top 10 Features by SHAP Importance',
                        label='tab:shap')
    save_latex_table(latex, TABLES_DIR, 'table08_shap_rankings')
    print("  Table 8: SHAP Rankings")


def generate_ode_parameters_table():
    """Table 9: ODE Parameters."""
    param_df = pd.read_csv(RESULTS_DIR / 'ode_parameters.csv')
    latex = df_to_latex(param_df, caption='SAD ODE Model Parameters',
                        label='tab:ode_params', column_format='llcll')
    save_latex_table(latex, TABLES_DIR, 'table09_ode_parameters')
    print("  Table 9: ODE Parameters")


def generate_ode_outcomes_table():
    """Table 10: ODE Simulation Outcomes."""
    outcome_df = pd.read_csv(RESULTS_DIR / 'ode_outcomes.csv')
    latex = df_to_latex(outcome_df, caption='SAD ODE Simulation Outcomes by Severity',
                        label='tab:ode_outcomes')
    save_latex_table(latex, TABLES_DIR, 'table10_ode_outcomes')
    print("  Table 10: ODE Outcomes")


def generate_figure_summary():
    """Print summary of all generated figures."""
    figures = sorted(FIGURES_DIR.glob('fig*.png'))
    print(f"\n  Total figures generated: {len(figures)}")
    for fig in figures:
        print(f"    {fig.name}")


def main():
    print("=" * 60)
    print("RESULTS GENERATOR - LaTeX TABLES & FIGURES")
    print("=" * 60)

    print("\n[LaTeX Tables]")
    table_generators = [
        generate_dataset_table,
        generate_class_distribution_table,
        generate_unet_hyperparams_table,
        generate_segformer_hyperparams_table,
        generate_segmentation_metrics_table,
        generate_feature_descriptions_table,
        generate_classification_comparison_table,
        generate_shap_ranking_table,
        generate_ode_parameters_table,
        generate_ode_outcomes_table,
    ]

    for gen in table_generators:
        try:
            gen()
        except FileNotFoundError as e:
            print(f"  SKIPPED (missing data): {e}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n[Figure Summary]")
    generate_figure_summary()

    # Save complete results summary
    summary = {'tables_generated': [], 'figures_generated': []}

    for tex_file in sorted(TABLES_DIR.glob('*.tex')):
        summary['tables_generated'].append(tex_file.name)
    for fig_file in sorted(FIGURES_DIR.glob('fig*.png')):
        summary['figures_generated'].append(fig_file.name)

    with open(RESULTS_DIR / 'pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
