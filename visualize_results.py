"""
visualize_results.py
====================
Visualization script to create parity plots comparing model predictions
with experimental data, matching the MATLAB plotting style.

USAGE
-----
After running liquac_fit.py or d_resolution.py:
    python visualize_results.py

This will generate plots showing:
- DIPA mole fraction (organic phase) vs experimental
- Water mole fraction (aqueous phase) vs experimental
- Color-coded by feed salt concentration
- Inset plots for detailed views
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from liquac_inputs import LIQUACInputs


# ═══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

SALT    = "NaCl"
SOLVENT = "DIPA"
DATA_DIR = "/Users/lucascaldentey/Desktop/Yip Lab"

# Choose which results to plot:
# "liquac"      → results from liquac_fit.py (coarse D values)
# "dresolution" → results from d_resolution.py (refined D values)
RESULT_TYPE = "dresolution"

# ═══════════════════════════════════════════════════════════════════════════════


def load_results(data_dir, salt, solvent, result_type):
    """Load experimental and predicted data."""
    folder = Path(data_dir) / f"{salt}-{solvent}"
    
    # Load experimental data
    loader = LIQUACInputs(data_dir)
    (z_exp, xE_exp, xR_exp, T_exp, 
     rho_solvent, dielec_solvent, 
     Selec_exp, ip_Gmehling) = loader.experimental_data(salt, solvent)
    
    # Load model predictions
    if result_type == "liquac":
        results_file = folder / "results_liquac.csv"
    elif result_type == "dresolution":
        results_file = folder / "results_Dresolution.csv"
    else:
        raise ValueError(f"Unknown result_type: {result_type}")
    
    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file}\n"
            f"Run {'liquac_fit.py' if result_type == 'liquac' else 'd_resolution.py'} first."
        )
    
    results = pd.read_csv(results_file)
    
    # Extract predicted compositions
    xE_pred = results[['xE_solvent', 'xE_water', 'xE_cation', 'xE_anion']].values
    xR_pred = results[['xR_solvent', 'xR_water', 'xR_cation', 'xR_anion']].values
    
    # Calculate feed salt concentration (molarity approximation)
    # Feed composition in ternary form
    z_ternary = loader.to_ternary(z_exp)
    M_NaCl_feed = z_ternary[:, 2]  # This is the salt-free normalized salt fraction
    
    return xE_exp, xR_exp, xE_pred, xR_pred, M_NaCl_feed, z_exp


def create_parity_plots(xE_exp, xR_exp, xE_pred, xR_pred, M_NaCl_feed, 
                         salt, solvent, result_type):
    """
    Create parity plots matching the MATLAB style.
    
    Plot A: DIPA (solvent) organic phase
    Plot B: Water aqueous phase
    """
    
    # Color mapping based on feed salt concentration
    vmin, vmax = M_NaCl_feed.min(), M_NaCl_feed.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_org = cm.get_cmap('YlOrRd')  # Yellow-Orange-Red for organic phase (DIPA)
    cmap_aq  = cm.get_cmap('Blues')   # Blues for aqueous phase (water)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Panel A: DIPA mole fraction in organic phase
    # ─────────────────────────────────────────────────────────────────────────
    
    # Main plot
    scatter1 = ax1.scatter(xE_exp[:, 0], xE_pred[:, 0], 
                          c=M_NaCl_feed, cmap=cmap_org, 
                          s=80, alpha=0.8, edgecolors='k', linewidth=0.5,
                          norm=norm)
    
    # Diagonal reference line
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, zorder=1)
    
    # Labels and annotations
    ax1.set_xlabel(f'{solvent} Experimental Mole\nFraction, $x_{{{solvent}}}$ (−)', 
                   fontsize=12)
    ax1.set_ylabel(f'{solvent} Model\nMole Fraction, $x_{{{solvent}}}$ (−)', 
                   fontsize=12)
    ax1.set_title(f'A)  {solvent}', fontsize=14, fontweight='bold', loc='left')
    
    # Add phase label
    ax1.text(0.05, 0.95, 'Aqueous Phase', 
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax1.annotate('', xy=(0.15, 0.88), xytext=(0.05, 0.88),
                transform=ax1.transAxes,
                arrowprops=dict(arrowstyle='->', lw=1.5))
    
    ax1.text(0.72, 0.5, 'Organic Phase', 
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='center',
            rotation=45)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Inset for aqueous phase (low DIPA concentration)
    ax1_inset = inset_axes(ax1, width="35%", height="35%", 
                           bbox_to_anchor=(0.38, 0.02, 0.6, 0.4),
                           bbox_transform=ax1.transAxes)
    
    # Filter points for inset (aqueous phase with low DIPA)
    mask_aq = xE_exp[:, 0] < 0.04
    ax1_inset.scatter(xE_exp[mask_aq, 0], xE_pred[mask_aq, 0],
                     c=M_NaCl_feed[mask_aq], cmap=cmap_org,
                     s=40, alpha=0.8, edgecolors='k', linewidth=0.5,
                     marker='^', norm=norm)
    ax1_inset.plot([0, 0.03], [0, 0.03], 'k--', linewidth=0.8, alpha=0.5)
    ax1_inset.set_xlim(0, 0.03)
    ax1_inset.set_ylim(0, 0.03)
    ax1_inset.grid(True, alpha=0.3)
    ax1_inset.tick_params(labelsize=8)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Panel B: Water mole fraction in aqueous phase
    # ─────────────────────────────────────────────────────────────────────────
    
    # Main plot
    scatter2 = ax2.scatter(xR_exp[:, 1], xR_pred[:, 1],
                          c=M_NaCl_feed, cmap=cmap_aq,
                          s=80, alpha=0.8, edgecolors='k', linewidth=0.5,
                          norm=norm)
    
    # Diagonal reference line
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, zorder=1)
    
    # Labels and annotations
    ax2.set_xlabel(r'Water Experimental Mole' + '\n' + r'Fraction, $x_{\mathrm{H_2O}}$ (−)', 
                   fontsize=12)
    ax2.set_ylabel(r'Water Model' + '\n' + r'Mole Fraction, $x_{\mathrm{H_2O}}$ (−)', 
                   fontsize=12)
    ax2.set_title(r'B)  H$_2$O', fontsize=14, fontweight='bold', loc='left')
    
    # Add phase label
    ax2.text(0.95, 0.95, 'Aqueous Phase', 
            transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax2.annotate('', xy=(0.85, 0.88), xytext=(0.95, 0.88),
                transform=ax2.transAxes,
                arrowprops=dict(arrowstyle='->', lw=1.5))
    
    ax2.text(0.5, 0.28, 'Organic Phase', 
            transform=ax2.transAxes, fontsize=10,
            verticalalignment='center',
            rotation=45)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Inset for organic phase (low water concentration)
    # Position the inset in the upper left area
    ax2_inset = inset_axes(ax2, width="28%", height="28%",
                           bbox_to_anchor=(0.72, 0.72, 0.27, 0.27),
                           bbox_transform=ax2.transAxes)
    
    # Filter points for inset (organic phase with low water)
    mask_org = xR_exp[:, 1] < 0.15
    if mask_org.sum() > 0:
        ax2_inset.scatter(xR_exp[mask_org, 1], xR_pred[mask_org, 1],
                         c=M_NaCl_feed[mask_org], cmap=cmap_aq,
                         s=30, alpha=0.8, edgecolors='k', linewidth=0.5,
                         marker='o', norm=norm)
        max_val = max(xR_exp[mask_org, 1].max(), xR_pred[mask_org, 1].max()) * 1.1
        ax2_inset.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, alpha=0.5)
        ax2_inset.set_xlim(0, max_val)
        ax2_inset.set_ylim(0, max_val)
        ax2_inset.grid(True, alpha=0.3)
        ax2_inset.tick_params(labelsize=8)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Colorbars
    # ─────────────────────────────────────────────────────────────────────────
    
    # Colorbar for panel A (organic)
    cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.02, aspect=30)
    cbar1.set_label(f'0.1 □──□ 4.5 mol/L: $M^{{\\mathrm{{Feed}}}}_{{\\mathrm{{{salt}}}}}$',
                   fontsize=10, labelpad=10)
    cbar1.ax.tick_params(labelsize=9)
    
    # Colorbar for panel B (aqueous)
    cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.02, aspect=30)
    cbar2.set_label(f'0.1 □──□ 4.5 mol/L: $M^{{\\mathrm{{Feed}}}}_{{\\mathrm{{{salt}}}}}$',
                   fontsize=10, labelpad=10)
    cbar2.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    
    return fig


def calculate_statistics(xE_exp, xR_exp, xE_pred, xR_pred):
    """Calculate goodness-of-fit statistics."""
    
    # Overall RMS error (matching the objective function)
    rms_E = np.sqrt(np.mean((xE_exp - xE_pred)**2))
    rms_R = np.sqrt(np.mean((xR_exp - xR_pred)**2))
    rms_total = np.sqrt(np.mean((xE_exp - xE_pred)**2) + np.mean((xR_exp - xR_pred)**2))
    
    # Component-wise statistics
    stats = {}
    components = ['solvent', 'water', 'cation', 'anion']
    
    for i, comp in enumerate(components):
        # Organic phase
        rmse_E = np.sqrt(np.mean((xE_exp[:, i] - xE_pred[:, i])**2))
        mae_E  = np.mean(np.abs(xE_exp[:, i] - xE_pred[:, i]))
        
        # Aqueous phase
        rmse_R = np.sqrt(np.mean((xR_exp[:, i] - xR_pred[:, i])**2))
        mae_R  = np.mean(np.abs(xR_exp[:, i] - xR_pred[:, i]))
        
        stats[comp] = {
            'organic_rmse': rmse_E,
            'organic_mae': mae_E,
            'aqueous_rmse': rmse_R,
            'aqueous_mae': mae_R,
        }
    
    return stats, rms_total


def print_statistics(stats, rms_total):
    """Print formatted statistics table."""
    print(f"\n{'='*70}")
    print(f"{'MODEL PERFORMANCE STATISTICS':^70}")
    print(f"{'='*70}\n")
    
    print(f"Overall RMS: {rms_total:.6f}\n")
    
    print(f"{'Component':<12} {'Organic Phase':<30} {'Aqueous Phase':<30}")
    print(f"{'':12} {'RMSE':<15} {'MAE':<15} {'RMSE':<15} {'MAE':<15}")
    print(f"{'-'*70}")
    
    for comp, stat in stats.items():
        print(f"{comp.capitalize():<12} "
              f"{stat['organic_rmse']:<15.6f} {stat['organic_mae']:<15.6f} "
              f"{stat['aqueous_rmse']:<15.6f} {stat['aqueous_mae']:<15.6f}")
    
    print(f"{'='*70}\n")


def main():
    """Main visualization workflow."""
    
    print(f"\n{'='*70}")
    print(f"LIQUAC RESULTS VISUALIZATION")
    print(f"{'='*70}")
    print(f"Salt:        {SALT}")
    print(f"Solvent:     {SOLVENT}")
    print(f"Result type: {RESULT_TYPE}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading data...")
    xE_exp, xR_exp, xE_pred, xR_pred, M_NaCl_feed, z_exp = load_results(
        DATA_DIR, SALT, SOLVENT, RESULT_TYPE
    )
    print(f"Loaded {len(xE_exp)} data points.\n")
    
    # Calculate statistics
    print("Calculating statistics...")
    stats, rms_total = calculate_statistics(xE_exp, xR_exp, xE_pred, xR_pred)
    print_statistics(stats, rms_total)
    
    # Create plots
    print("Creating parity plots...")
    fig = create_parity_plots(
        xE_exp, xR_exp, xE_pred, xR_pred, M_NaCl_feed,
        SALT, SOLVENT, RESULT_TYPE
    )
    
    # Save figure
    output_dir = Path(DATA_DIR) / f"{SALT}-{SOLVENT}"
    output_file = output_dir / f"parity_plot_{RESULT_TYPE}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    print(f"\n{'='*70}")
    print("Visualization complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()