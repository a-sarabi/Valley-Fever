import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import re


# =====================================================================================
# VISUALIZATION AND SUMMARY GENERATION FUNCTIONS
# =====================================================================================

def load_stability_analysis_results(results_directory):
    """
    Load stability analysis results from the specified directory

    Returns:
        dict: Dictionary containing loaded results for each horizon
    """

    print(f"🔍 Loading stability analysis results from: {results_directory}")

    if not os.path.exists(results_directory):
        print(f"❌ Error: Directory {results_directory} does not exist!")
        return None

    # Find all horizon directories
    horizon_dirs = []
    for item in os.listdir(results_directory):
        item_path = os.path.join(results_directory, item)
        if os.path.isdir(item_path) and item.startswith('horizon_'):
            try:
                horizon = int(item.split('_')[1])
                horizon_dirs.append((horizon, item_path))
            except (IndexError, ValueError):
                continue

    if not horizon_dirs:
        print(f"❌ Error: No horizon directories found in {results_directory}")
        return None

    horizon_dirs.sort()  # Sort by horizon
    print(f"📂 Found {len(horizon_dirs)} horizon directories: {[f'horizon_{h}' for h, _ in horizon_dirs]}")

    # Load results for each horizon
    loaded_results = {}

    for horizon, horizon_dir in horizon_dirs:
        print(f"\n📖 Loading results for {horizon}-week horizon...")

        # Required files
        required_files = {
            'metadata': os.path.join(horizon_dir, 'forecasting_metadata.json'),
            'importance_across_seeds': os.path.join(horizon_dir, 'feature_importance_across_seeds.csv'),
            'stability_results': os.path.join(horizon_dir, 'stability_analysis_results.csv')
        }

        # Check if all required files exist
        missing_files = []
        for file_type, file_path in required_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{file_type}: {file_path}")

        if missing_files:
            print(f"⚠️  Warning: Missing files for {horizon}-week horizon:")
            for missing in missing_files:
                print(f"     - {missing}")
            print("   Skipping this horizon...")
            continue

        try:
            # Load metadata
            with open(required_files['metadata'], 'r') as f:
                metadata = json.load(f)

            # Load importance across seeds
            importance_df = pd.read_csv(required_files['importance_across_seeds'])

            # Load stability results
            stability_results = pd.read_csv(required_files['stability_results'])

            # Store loaded data
            loaded_results[horizon] = {
                'metadata': metadata,
                'importance_df': importance_df,
                'stability_results': stability_results,
                'lagged_feature_names': metadata['lagged_feature_names']
            }

            print(f"   ✅ Successfully loaded data for {horizon}-week horizon")

        except Exception as e:
            print(f"   ❌ Error loading files for {horizon}-week horizon: {str(e)}")
            continue

    if not loaded_results:
        print(f"\n❌ No valid horizon results could be loaded!")
        return None

    print(f"\n✅ Successfully loaded results for {len(loaded_results)} horizons: {list(loaded_results.keys())}")
    return loaded_results


def create_multi_horizon_visualizations(horizon_results, save_directory, feature_selection_percent=0.1):
    """
    Create comprehensive visualizations comparing feature importance across horizons
    """

    print(f"\n🎨 Creating multi-horizon visualizations...")
    print(f"📂 Saving to: {save_directory}")

    horizons = sorted(horizon_results.keys())
    lagged_feature_names = list(horizon_results.values())[0]['lagged_feature_names']

    # ===== 1. HEATMAP: Top Features Across All Horizons =====
    print("📊 Creating multi-horizon feature importance heatmap...")

    # Collect top features for each horizon
    top_features_per_horizon = {}
    all_important_features = set()

    for horizon in horizons:
        if horizon not in horizon_results:
            continue

        stability_results = horizon_results[horizon]['stability_results']
        total_features = len(stability_results)
        top_n = max(1, int(total_features * feature_selection_percent))

        # Get top features by mean importance
        top_features = stability_results.nlargest(top_n, 'mean_importance')

        if 'feature_name' in top_features.columns:
            feature_names = top_features['feature_name'].tolist()
        else:
            feature_names = [lagged_feature_names[idx] for idx in top_features['feature_index']]

        top_features_per_horizon[horizon] = dict(zip(feature_names, top_features['mean_importance']))
        all_important_features.update(feature_names)

    # Create heatmap data
    heatmap_data = []
    feature_list = sorted(list(all_important_features))

    for feature in feature_list:
        row = []
        for horizon in horizons:
            if horizon in top_features_per_horizon and feature in top_features_per_horizon[horizon]:
                row.append(top_features_per_horizon[horizon][feature])
            else:
                row.append(0)  # Feature not important for this horizon
        heatmap_data.append(row)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(feature_list) * 0.3)))
    heatmap_df = pd.DataFrame(heatmap_data,
                              columns=[f'{h}-week' for h in horizons],
                              index=feature_list)

    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlBu_r',
                cbar_kws={'label': 'Feature Importance'}, ax=ax)
    plt.title('Feature Importance Across Forecast Horizons\n(0 = Not in Top Features for that Horizon)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Forecast Horizon')
    plt.ylabel('Features')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{save_directory}/multi_horizon_feature_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ===== 2. FEATURE RANKING COMPARISON =====
    print("📊 Creating feature ranking comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for i, horizon in enumerate(horizons):
        if horizon not in horizon_results or i >= len(axes):
            continue

        stability_results = horizon_results[horizon]['stability_results']
        top_features = stability_results.head(15)  # Top 15 for visibility

        if 'feature_name' in top_features.columns:
            feature_names = top_features['feature_name']
        else:
            feature_names = [lagged_feature_names[idx] for idx in top_features['feature_index']]

        # Create horizontal bar plot
        y_pos = np.arange(len(feature_names))
        axes[i].barh(y_pos, top_features['mean_importance'], alpha=0.8)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(feature_names, fontsize=9)
        axes[i].set_xlabel('Mean Importance')
        axes[i].set_title(f'{horizon}-Week Horizon\nTop 15 Features')
        axes[i].grid(True, alpha=0.3)
        axes[i].invert_yaxis()  # Top feature at the top

    plt.tight_layout()
    plt.savefig(f'{save_directory}/feature_ranking_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ===== 3. SHORT-TERM VS LONG-TERM ANALYSIS =====
    print("📊 Creating short-term vs long-term analysis...")

    # Define short-term (2, 4 weeks) vs long-term (8, 16 weeks)
    short_term_horizons = [h for h in horizons if h <= 4]
    long_term_horizons = [h for h in horizons if h >= 8]

    # Aggregate importance scores
    short_term_importance = defaultdict(list)
    long_term_importance = defaultdict(list)

    # Collect importance for short-term horizons
    for horizon in short_term_horizons:
        if horizon not in horizon_results:
            continue
        importance_df = horizon_results[horizon]['importance_df']
        for idx, feature_name in enumerate(lagged_feature_names):
            if idx < len(importance_df):
                # Calculate mean importance across seeds (excluding non-numeric columns)
                numeric_cols = importance_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    mean_importance = importance_df.iloc[idx][numeric_cols].mean()
                    short_term_importance[feature_name].append(mean_importance)

    # Collect importance for long-term horizons
    for horizon in long_term_horizons:
        if horizon not in horizon_results:
            continue
        importance_df = horizon_results[horizon]['importance_df']
        for idx, feature_name in enumerate(lagged_feature_names):
            if idx < len(importance_df):
                # Calculate mean importance across seeds (excluding non-numeric columns)
                numeric_cols = importance_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    mean_importance = importance_df.iloc[idx][numeric_cols].mean()
                    long_term_importance[feature_name].append(mean_importance)

    # Calculate average importance for each feature
    short_term_avg = {feat: np.mean(scores) for feat, scores in short_term_importance.items() if scores}
    long_term_avg = {feat: np.mean(scores) for feat, scores in long_term_importance.items() if scores}

    # Create comparison DataFrame
    all_features = set(short_term_avg.keys()) | set(long_term_avg.keys())
    comparison_data = []

    for feature in all_features:
        short_score = short_term_avg.get(feature, 0)
        long_score = long_term_avg.get(feature, 0)
        comparison_data.append({
            'feature': feature,
            'short_term': short_score,
            'long_term': long_score,
            'difference': long_score - short_score,
            'preference': 'Long-term' if long_score > short_score else 'Short-term'
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('difference', ascending=False)

    # Plot short-term vs long-term scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Scatter plot
    scatter = ax1.scatter(comparison_df['short_term'], comparison_df['long_term'],
                          c=comparison_df['difference'], cmap='RdBu_r', alpha=0.7, s=50)
    ax1.plot([0, max(comparison_df[['short_term', 'long_term']].max())],
             [0, max(comparison_df[['short_term', 'long_term']].max())], 'k--', alpha=0.5)
    ax1.set_xlabel('Short-term Importance (2-4 weeks)')
    ax1.set_ylabel('Long-term Importance (8-16 weeks)')
    ax1.set_title('Feature Importance: Short-term vs Long-term')
    plt.colorbar(scatter, ax=ax1, label='Long-term - Short-term')
    ax1.grid(True, alpha=0.3)

    # Top differences
    top_long_term = comparison_df.head(10)
    top_short_term = comparison_df.tail(10)

    y_pos = np.arange(10)
    width = 0.35

    ax2.barh(y_pos - width / 2, top_long_term['long_term'], width,
             label='Long-term Preferred', color='red', alpha=0.7)
    ax2.barh(y_pos + width / 2, -top_short_term['short_term'], width,
             label='Short-term Preferred', color='blue', alpha=0.7)

    # Combine labels
    combined_labels = list(top_long_term['feature'].str[:25])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(combined_labels, fontsize=9)
    ax2.set_xlabel('Importance (Long-term: positive, Short-term: negative)')
    ax2.set_title('Top Features by Time Preference')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_directory}/short_vs_long_term_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ===== 4. FEATURE CATEGORY ANALYSIS ACROSS HORIZONS =====
    print("📊 Creating feature category analysis...")

    # Define feature categories (improved grouping based on Valley Fever ecology)
    feature_category_map = {
        'MARICOPA': 'Surveillance',
        '20" Soil Temp': 'Soil',
        '4" Soil Temp': 'Soil',
        'Air Temp': 'Temperature',
        # Moisture-related variables (affect fungal growth and spore viability)
        'Dewpoint': 'Moisture/Humidity',
        'Dew-point': 'Moisture/Humidity',
        'Actual Vapor Pressure': 'Moisture/Humidity',
        'RH': 'Moisture/Humidity',
        'VPD': 'Moisture/Humidity',
        'Precipitation': 'Moisture/Humidity',
        # Wind and radiation (affect spore dispersal and environmental stress)
        'Wind Speed': 'Wind/Radiation',
        'Wind Vector': 'Wind/Radiation',
        'Wind Direction': 'Wind/Radiation',
        'Max Wind Speed': 'Wind/Radiation',
        'Solar Rad': 'Wind/Radiation',
        'Heat Units': 'Agricultural',
        'Reference ET': 'Agricultural',
        'AQI': 'Air Quality',
        'PM10': 'Air Quality'
    }

    def parse_feature_info(feature_name):
        # Extract base feature and lag
        match = re.match(r'(.+?) \((-?\d+)\)$', feature_name)
        if match:
            base_feature = match.group(1).strip()
            lag = int(match.group(2))

            # Determine category
            category = 'Other'
            for key, cat in feature_category_map.items():
                if key in base_feature:
                    category = cat
                    break

            # Determine lag bin
            lag_bin = 'Recent (≤3)' if abs(lag) <= 3 else 'Delayed (>3)'

            return category, lag_bin, abs(lag)
        return 'Other', 'Recent (≤3)', 0

    # Analyze category importance across horizons
    category_horizon_data = []

    for horizon in horizons:
        if horizon not in horizon_results:
            continue

        stability_results = horizon_results[horizon]['stability_results']

        for _, row in stability_results.iterrows():
            if 'feature_name' in stability_results.columns:
                feature_name = row['feature_name']
            else:
                feature_name = lagged_feature_names[row['feature_index']]

            category, lag_bin, lag_value = parse_feature_info(feature_name)

            category_horizon_data.append({
                'horizon': horizon,
                'feature': feature_name,
                'category': category,
                'lag_bin': lag_bin,
                'lag_value': lag_value,
                'importance': row['mean_importance'],
                'stability': row['coefficient_of_variation']
            })

    category_df = pd.DataFrame(category_horizon_data)

    # Create category analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Category importance by horizon
    category_means = category_df.groupby(['horizon', 'category'])['importance'].mean().reset_index()
    category_pivot = category_means.pivot(index='category', columns='horizon', values='importance')

    sns.heatmap(category_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0, 0])
    axes[0, 0].set_title('Average Feature Importance by Category and Horizon')
    axes[0, 0].set_xlabel('Forecast Horizon (weeks)')

    # Lag bin analysis
    lag_means = category_df.groupby(['horizon', 'lag_bin'])['importance'].mean().reset_index()
    lag_pivot = lag_means.pivot(index='lag_bin', columns='horizon', values='importance')

    sns.heatmap(lag_pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0, 1])
    axes[0, 1].set_title('Average Feature Importance by Lag Bin and Horizon')
    axes[0, 1].set_xlabel('Forecast Horizon (weeks)')

    # Category distribution across horizons
    sns.boxplot(data=category_df, x='horizon', y='importance', hue='category', ax=axes[1, 0])
    axes[1, 0].set_title('Feature Importance Distribution by Category')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Lag value vs importance across horizons
    for horizon in horizons:
        horizon_data = category_df[category_df['horizon'] == horizon]
        axes[1, 1].scatter(horizon_data['lag_value'], horizon_data['importance'],
                           label=f'{horizon}-week', alpha=0.6)

    axes[1, 1].set_xlabel('Lag Value (weeks)')
    axes[1, 1].set_ylabel('Feature Importance')
    axes[1, 1].set_title('Lag Value vs Importance Across Horizons')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_directory}/category_analysis_across_horizons.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ===== 5. SUMMARY REPORT =====
    print("📊 Creating summary report...")

    with open(f'{save_directory}/multi_horizon_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-HORIZON FEATURE IMPORTANCE ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for horizon in horizons:
            if horizon not in horizon_results:
                continue

            f.write(f"{horizon}-WEEK HORIZON ANALYSIS:\n")
            f.write("-" * 50 + "\n")

            stability_results = horizon_results[horizon]['stability_results']

            # Top 5 features
            top_5 = stability_results.head(5)
            f.write("Top 5 Most Important Features:\n")
            for idx, (_, row) in enumerate(top_5.iterrows(), 1):
                if 'feature_name' in stability_results.columns:
                    feature_name = row['feature_name']
                else:
                    feature_name = lagged_feature_names[row['feature_index']]

                f.write(f"  {idx}. {feature_name[:60]:<60} (Importance: {row['mean_importance']:.4f})\n")

            # Stability stats
            stable_features = len(stability_results[stability_results['coefficient_of_variation'] < 0.3])
            total_features = len(stability_results)

            f.write(f"\nStability Statistics:\n")
            f.write(f"  • Total features: {total_features}\n")
            f.write(
                f"  • Stable features (CV < 0.3): {stable_features} ({stable_features / total_features * 100:.1f}%)\n")
            f.write(f"  • Median stability (CV): {stability_results['coefficient_of_variation'].median():.3f}\n\n")

        # Cross-horizon insights
        f.write("CROSS-HORIZON INSIGHTS:\n")
        f.write("-" * 50 + "\n")

        # Most consistent features across horizons
        feature_consistency = defaultdict(list)
        for horizon in horizons:
            if horizon not in horizon_results:
                continue
            stability_results = horizon_results[horizon]['stability_results']
            top_10_percent = int(len(stability_results) * 0.1)
            top_features = stability_results.head(top_10_percent)

            for _, row in top_features.iterrows():
                if 'feature_name' in stability_results.columns:
                    feature_name = row['feature_name']
                else:
                    feature_name = lagged_feature_names[row['feature_index']]
                feature_consistency[feature_name].append(horizon)

        # Features appearing in multiple horizons
        multi_horizon_features = {feat: horizons_list for feat, horizons_list in feature_consistency.items()
                                  if len(horizons_list) > 1}

        f.write("Features Important Across Multiple Horizons:\n")
        for feature, feature_horizons in sorted(multi_horizon_features.items(),
                                                key=lambda x: len(x[1]), reverse=True)[:10]:
            f.write(f"  • {feature[:50]:<50} -> {feature_horizons}\n")

    print(f"✅ Multi-horizon visualizations completed!")
    print(f"📂 Results saved to: {save_directory}/")
    print(f"📊 Generated visualizations:")
    print(f"   - multi_horizon_feature_heatmap.png")
    print(f"   - feature_ranking_comparison.png")
    print(f"   - short_vs_long_term_analysis.png")
    print(f"   - category_analysis_across_horizons.png")
    print(f"📋 Summary report: multi_horizon_summary_report.txt")


# =====================================================================================
# MAIN EXECUTION CODE
# =====================================================================================

if __name__ == '__main__':

    print("🎨 Multi-Horizon Visualization Generator")
    print("=" * 60)
    print("This script loads stability analysis results and generates")
    print("comparative visualizations and summaries across horizons.")
    print("=" * 60)

    # ===== USER CONFIGURATION =====

    # Path to the analysis results directory (from the stability analysis script)
    # Example: "multi_horizon_results_2024-12-19_14-30-45_Clean Processed Data2"
    analysis_results_directory = input("📂 Enter the path to your stability analysis results directory: ").strip()

    # Feature selection percentage (should match what was used in analysis)
    feature_selection_percent = 0.1  # Top 10% features

    # ===== LOAD STABILITY ANALYSIS RESULTS =====
    loaded_results = load_stability_analysis_results(analysis_results_directory)

    if loaded_results is None:
        print("\n❌ Failed to load stability analysis results. Please check the directory path.")
        exit(1)

    # ===== GENERATE VISUALIZATIONS =====
    try:
        create_multi_horizon_visualizations(
            horizon_results=loaded_results,
            save_directory=analysis_results_directory,
            feature_selection_percent=feature_selection_percent
        )

        print(f"\n🎉 Visualization generation completed successfully!")
        print(f"📂 All visualizations saved to: {analysis_results_directory}")

    except Exception as e:
        print(f"\n❌ Error during visualization generation: {str(e)}")
        exit(1)

    # ===== FINAL SUMMARY =====
    print(f"\n" + "=" * 80)
    print("📋 VISUALIZATION GENERATION SUMMARY")
    print("=" * 80)

    available_horizons = list(loaded_results.keys())
    print(f"✅ Generated visualizations for {len(available_horizons)} horizons: {available_horizons}")

    print(f"\n📊 Generated Files:")
    print(f"   - multi_horizon_feature_heatmap.png")
    print(f"   - feature_ranking_comparison.png")
    print(f"   - short_vs_long_term_analysis.png")
    print(f"   - category_analysis_across_horizons.png")
    print(f"   - multi_horizon_summary_report.txt")

    print(f"\n🔬 Research Insights Available:")
    print(f"   • Feature importance patterns across forecast horizons")
    print(f"   • Short-term (≤4 weeks) vs long-term (≥8 weeks) feature preferences")
    print(f"   • Environmental category analysis (Soil, Temperature, Moisture, etc.)")
    print(f"   • Lag effect patterns for different prediction horizons")
    print(f"   • Cross-horizon feature consistency analysis")

    print("=" * 80)
    print("🎉 Multi-horizon visualization script completed!")
    print("=" * 80)