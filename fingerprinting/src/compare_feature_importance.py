import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_rank_changes(main_report_path: str, variant_report_path: str, output_plot_path: str):
    """
    Compare l'importance des caractéristiques entre deux classificateurs (par exemple,
    un classificateur général et un classificateur de variantes) et visualise les changements de rang.
    """
    print(f"Chargement du rapport principal depuis : {main_report_path}")
    try:
        df_main = pd.read_csv(main_report_path)
    except FileNotFoundError:
        print(f"Erreur : Fichier non trouvé '{main_report_path}'. Veuillez d'abord le générer.")
        return
    
    print(f"Chargement du rapport des variantes depuis : {variant_report_path}")
    try:
        df_variant = pd.read_csv(variant_report_path)
    except FileNotFoundError:
        print(f"Erreur : Fichier non trouvé '{variant_report_path}'. Veuillez d'abord le générer.")
        return
    
    # Ajoute une colonne de rang à chaque DataFrame (le rang est basé sur l'index, car les fichiers sont déjà triés)
    df_main['rank_main'] = df_main.index + 1
    df_variant['rank_variant'] = df_variant.index + 1

    df_merged = pd.merge(
        df_main[['feature', 'rank_main']],
        df_variant[['feature', 'rank_variant']],
        on='feature',
        how='outer'
    )

    # Remplace les NaN par le rang maximum (pour les caractéristiques présentes dans un seul rapport)
    max_rank = df_merged[['rank_main', 'rank_variant']].max().max()
    df_merged.fillna(max_rank, inplace=True)

    # Un changement positif signifie que la métrique est MIEUX classée (plus importante) dans le classificateur de variantes
    # Un changement négatif signifie qu'elle est MOINS bien classée
    df_merged['rank_change'] = df_merged['rank_main'] - df_merged['rank_variant']

    df_merged['abs_rank_change'] = df_merged['rank_change'].abs()
    df_sorted = df_merged.sort_values(by='abs_rank_change', ascending=False).reset_index(drop=True)

    print("\nTop 15 des métriques avec le plus grand changement de classement :")
    print(df_sorted[['feature', 'rank_main', 'rank_variant', 'rank_change']].head(15))

    plot_data = df_sorted.sort_values(by='rank_change', ascending=False)
    num_metrics = len(plot_data)

    figure_height = max(14, num_metrics * 0.4)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 20))

    colors = ['#2ca02c' if x > 0 else '#d62728' for x in plot_data['rank_change']]

    sns.barplot(x='rank_change', y='feature', data=plot_data, palette=colors, hue='feature', dodge=False)

    plt.xlabel('Changement de rang (rang classificateur principal - rang mini-classificateur)', fontsize=12)
    plt.ylabel('Métrique', fontsize=12)
    plt.title(f'Changements d\'importance des métriques', fontsize=16, pad=20)
    plt.legend([], [], frameon=False) # Cache la légende générée par `hue`
    plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--') # Ligne centrale à zéro

    plt.text(0.98, 1.01, '↓ Moins important pour les variantes', transform=plt.gca().transAxes, ha='right', color='#d62728', style='italic')
    plt.text(0.02, 1.01, '↑ Plus important pour les variantes', transform=plt.gca().transAxes, ha='left', color='#2ca02c', style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(output_plot_path)
    print(f"\nGraphique de comparaison sauvegardé dans : {output_plot_path}")
    plt.show()


if __name__ == "__main__":
    MAIN_REPORT = 'feature_importance_report2.csv'
    VARIANT_REPORT = 'variant_feature_importance.csv'
    OUTPUT_PLOT = 'metric_rank_change_comparison.png'

    visualize_rank_changes(MAIN_REPORT, VARIANT_REPORT, OUTPUT_PLOT)