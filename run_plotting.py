import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.manifold import TSNE # Cần import lại TSNE trong môi trường pubfigures nếu muốn chạy plot


# Định nghĩa thông số chung cho IEEE (Times New Roman, vector PDF)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.format': 'pdf'
})

# --- CONFIG (Cập nhật đường dẫn thực tế của bạn) ---
RESULTS_FILE = "final_paper_results.csv"
OUTPUT_DIR = "figures_for_paper"

# Đường dẫn file Log Training
LOG_FILE_CLASSIC = "experiments/20251128_204521_kaggle_Classic_ResNet18/training_log.csv"
LOG_FILE_QUANTUM = "experiments/20251128_205147_kaggle_Quantum_ResNet18/training_log.csv"

# ======================================================================
# --- FUNCTIONS FOR PLOTTING FIGURES ---
# ======================================================================

def load_data():
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"File kết quả {RESULTS_FILE} không tồn tại.")
    return pd.read_csv(RESULTS_FILE)

def plot_line_comparison(df, x_col, y_col, hue_col, title, xlabel, ylabel, output_path, y_min=40, y_max=101, custom_annotations=None):

    df_plot = df.copy()
    df_plot['Model'] = df_plot['Model'].replace({
        'Classic_ResNet': 'Classic ResNet-18',
        'Quantum_ResNet': 'Q-ResNet (Proposed)'
    })

    fig, ax = plt.subplots(figsize=(3.45, 2.5))

    sns.lineplot(
        data=df_plot,
        x=x_col,
        y=y_col,
        hue=hue_col,
        style=hue_col,
        markers=True,
        dashes=False,
        ax=ax,
        linewidth=1.5
    )

    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Model', loc='best', frameon=True)

    # Custom Annotations (Chỉ dùng cho Figure 2)
    if custom_annotations:
        for x, y, color in custom_annotations:
            ax.scatter([x], [y], color=color, marker='o', s=50, zorder=10)

    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def plot_training_dynamics():

    # Load Logs và xử lý lỗi NaN trong cột Acc/AUC (xảy ra trong log thật)
    df_classic = pd.read_csv(LOG_FILE_CLASSIC).assign(Model='Classic ResNet-18')
    df_quantum = pd.read_csv(LOG_FILE_QUANTUM).assign(Model='Q-ResNet (Proposed)')

    # Fill NaN bằng 0 hoặc giá trị an toàn
    df_classic[['val_acc', 'val_auc']] = df_classic[['val_acc', 'val_auc']].fillna(100.0)
    df_quantum[['val_acc', 'val_auc']] = df_quantum[['val_acc', 'val_auc']].fillna(100.0)

    df_logs = pd.concat([df_classic, df_quantum])

    # --- FIGURE 6a: Training Loss ---
    fig1, ax1 = plt.subplots(figsize=(3.45, 2.5))

    sns.lineplot(data=df_logs, x='epoch', y='train_loss', hue='Model', style='Model', markers=True, ax=ax1, linewidth=1.5)
    ax1.set_title('Training Loss Convergence')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss (Cross Entropy)')
    ax1.set_yscale('log')
    ax1.legend(title='Model', loc='best', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    output_path_loss = os.path.join(OUTPUT_DIR, 'Figure_6a_Training_Loss.pdf')
    plt.savefig(output_path_loss, dpi=600, bbox_inches='tight')
    plt.close(fig1)
    print(f"[SUCCESS] Saved Figure 6a: Training Loss PDF")

    # --- FIGURE 6b: Validation Accuracy ---
    fig2, ax2 = plt.subplots(figsize=(3.45, 2.5))

    # Đảm bảo trục Y bắt đầu từ 99.0% (Không zoom quá sát để tránh lỗi hiển thị)
    sns.lineplot(data=df_logs, x='epoch', y='val_acc', hue='Model', style='Model', markers=True, ax=ax2, linewidth=1.5)
    ax2.set_title('Validation Accuracy Dynamics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_ylim(99.9, 100.01) # Zoom vừa phải, từ 99.9%
    ax2.legend(title='Model', loc='best', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)

    output_path_acc = os.path.join(OUTPUT_DIR, 'Figure_6b_Validation_Acc.pdf')
    plt.savefig(output_path_acc, dpi=600, bbox_inches='tight')
    plt.close(fig2)
    print(f"[SUCCESS] Saved Figure 6b: Validation Acc PDF")


# ======================================================================
# --- FUNCTION FOR TABLE GENERATION ---
# ======================================================================

def generate_robustness_table(df):

    noise_points = [0.0, 0.4, 0.6]
    df_noise = df[df['Attack'] == 'Gaussian_Noise']

    occ_points = [0.0, 80.0, 100.0]
    df_occ = df[df['Attack'] == 'Occlusion']

    records = []

    def get_acc(df_in, model_name, severity):
        return df_in[(df_in['Severity'] == severity) & (df_in['Model'] == model_name)]['Accuracy'].iloc[0]

    # Lấy dữ liệu Noise
    for sigma in noise_points:
        classic_acc = get_acc(df_noise, 'Classic_ResNet', sigma)
        quantum_acc = get_acc(df_noise, 'Quantum_ResNet', sigma)

        records.append({
            'Attack Type': 'Gaussian Noise',
            'Severity Level': f'Sigma = {sigma:.1f}',
            'Classic Acc (\%)': f'{classic_acc:.2f}',
            'Quantum Acc (\%)': f'{quantum_acc:.2f}',
            'Gap (Q - C)': f'{quantum_acc - classic_acc:+.2f}'
        })

    # Lấy dữ liệu Occlusion
    for bs in occ_points:
        classic_acc = get_acc(df_occ, 'Classic_ResNet', bs)
        quantum_acc = get_acc(df_occ, 'Quantum_ResNet', bs)

        records.append({
            'Attack Type': 'Occlusion',
            'Severity Level': f'Block = {int(bs)}x{int(bs)}',
            'Classic Acc (\%)': f'{classic_acc:.2f}',
            'Quantum Acc (\%)': f'{quantum_acc:.2f}',
            'Gap (Q - C)': f'{quantum_acc - classic_acc:+.2f}'
        })

    df_table = pd.DataFrame(records)

    output_csv = os.path.join(OUTPUT_DIR, 'Table_2_Robustness_Key_Metrics.csv')
    df_table.to_csv(output_csv, index=False)

    print(f"\n[SUCCESS] Saved Table 2 (Robustness Metrics) to {output_csv}")
    print("\n--- TABLE 2 PREVIEW (MARKDOWN) ---")
    print(df_table.to_markdown(index=False))
    print("------------------------------------")

# ======================================================================
# --- MAIN EXECUTION ---
# ======================================================================

def plot_tsne_comparison():
    
    # Load kết quả t-SNE đã được tính toán
    TSNE_FILE = os.path.join(OUTPUT_DIR, "tsne_features.csv")
    if not os.path.exists(TSNE_FILE):
        print(f"LỖI: Chưa chạy run_feature_extractor.py để tạo {TSNE_FILE}")
        return

    df_tsne = pd.read_csv(TSNE_FILE)
    
    # --- FIGURE 5: FEATURE SPACE VISUALIZATION ---
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0)) # Subplots 2 hình cạnh nhau
    
    # --- Hình 5a: Classic ResNet (512D) ---
    sns.scatterplot(
        data=df_tsne, x='C_x', y='C_y', hue='Label', 
        ax=axes[0], palette='viridis', s=10, alpha=0.7, 
        legend='full'
    )
    axes[0].set_title('(a) Classic ResNet Feature Space (512D)', fontsize=9)
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    axes[0].get_legend().set_title('Class')
    
    # --- Hình 5b: Q-ResNet (4D) ---
    sns.scatterplot(
        data=df_tsne, x='Q_x', y='Q_y', hue='Label', 
        ax=axes[1], palette='viridis', s=10, alpha=0.7, 
        legend='full'
    )
    axes[1].set_title('(b) Q-ResNet Quantum Feature Space (4D)', fontsize=9)
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('') # Bỏ trục Y để gọn hơn
    axes[1].get_legend().set_title('Class')
    
    plt.tight_layout()
    output_path_tsne = os.path.join(OUTPUT_DIR, 'Figure_5_Feature_Space_TSNE.pdf')
    plt.savefig(output_path_tsne, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"[SUCCESS] Saved Figure 5: Feature Space TSNE PDF")



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()

    # FIGURE 2: GAUSSIAN NOISE
    df_noise = df[df['Attack'] == 'Gaussian_Noise']
    plot_line_comparison(
        df=df_noise, x_col='Severity', y_col='Accuracy', hue_col='Model',
        title='Resilience Against Gaussian Noise ($\sigma$)',
        xlabel='Gaussian Noise Standard Deviation ($\sigma$)',
        ylabel='Accuracy (\%)',
        output_path=os.path.join(OUTPUT_DIR, 'Figure_2_Noise_Robustness.pdf'),
        custom_annotations=[(0.4, 61.06, 'red'), (0.4, 90.45, 'blue')] # Đánh dấu điểm sụp đổ
    )

    # FIGURE 3: OCCLUSION
    df_occlusion = df[df['Attack'] == 'Occlusion']
    plot_line_comparison(
        df=df_occlusion, x_col='Severity', y_col='Accuracy', hue_col='Model',
        title='Resilience Against Physical Occlusion',
        xlabel='Occlusion Block Size (pixels)',
        ylabel='Accuracy (\%)',
        output_path=os.path.join(OUTPUT_DIR, 'Figure_3_Occlusion_Robustness.pdf'),
        y_min=90, y_max=101
    )

    # FIGURE 6a & 6b: TRAINING DYNAMICS
    plot_training_dynamics()

    #plot_tsne_comparison()


    # TABLE 2: ROBUSTNESS KEY METRICS
    generate_robustness_table(df)

if __name__ == '__main__':
    main()
