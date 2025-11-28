import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập đường dẫn thư mục gốc của repo scientific-figures-template
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'src'))

from publication_style import set_publication_style
# Giả sử hàm plot_line_comparison đã được định nghĩa trong plot_templates.py

DATA_FILE = "../final_paper_results.csv" # Đường dẫn tương đối từ examples/

def create_plots():
    # Sử dụng style chuẩn (Times New Roman, 600 DPI PDF)
    set_publication_style(context='paper', font_size=10) 

    # 1. Load Data
    df = pd.read_csv(DATA_FILE)

    # 2. Chuẩn bị dữ liệu cho Gaussian Noise (Figure 2)
    df_noise = df[df['Attack'] == 'Gaussian_Noise'].rename(
        columns={'Severity': 'Noise Level ($\sigma$)', 'Accuracy': 'Accuracy (%)'}
    )
    
    # Chuẩn bị dữ liệu cho Occlusion (Figure 3)
    df_occlusion = df[df['Attack'] == 'Occlusion'].rename(
        columns={'Severity': 'Occlusion Block Size (pixels)', 'Accuracy': 'Accuracy (%)'}
    )

    # Dùng melt để đưa hai cột model vào một cột "Model"
    df_noise_melt = df_noise.melt(
        id_vars=['Noise Level ($\sigma$)'],
        value_vars=['Classic_ResNet', 'Quantum_ResNet'],
        var_name='Model',
        value_name='Accuracy (%)'
    ).replace({'Classic_ResNet': 'Classic ResNet-18', 'Quantum_ResNet': 'Q-ResNet (Proposed)'})

    df_occlusion_melt = df_occlusion.melt(
        id_vars=['Occlusion Block Size (pixels)'],
        value_vars=['Classic_ResNet', 'Quantum_ResNet'],
        var_name='Model',
        value_name='Accuracy (%)'
    ).replace({'Classic_ResNet': 'Classic ResNet-18', 'Quantum_ResNet': 'Q-ResNet (Proposed)'})
    
    
    # --- FIGURE 2: GAUSSIAN NOISE ---
    fig1, ax1 = plt.subplots(figsize=(3.4, 2.5)) # Kích thước chuẩn 1 cột IEEE
    sns.lineplot(data=df_noise_melt, x='Noise Level ($\sigma$)', y='Accuracy (%)', 
                 hue='Model', style='Model', markers=True, dashes=False, ax=ax1)
    ax1.set_title('Robustness Against Gaussian Noise ($\sigma$)', fontsize=10)
    ax1.set_ylim(40, 105)
    ax1.legend(loc='upper right', frameon=True, fontsize=8)
    
    # Highlight điểm sụp đổ của Classic (0.4) và điểm sống sót của Quantum
    ax1.scatter([0.4, 0.4], [61.06, 90.45], color=['red', 'blue'], marker='o', zorder=10)
    
    plt.savefig('../figures/Figure_2_Noise_Robustness.pdf', dpi=600, bbox_inches='tight', format='pdf')
    print("Saved Figure 2: Noise Robustness")


    # --- FIGURE 3: OCCLUSION ---
    fig2, ax2 = plt.subplots(figsize=(3.4, 2.5))
    sns.lineplot(data=df_occlusion_melt, x='Occlusion Block Size (pixels)', y='Accuracy (%)', 
                 hue='Model', style='Model', markers=True, dashes=False, ax=ax2)
    ax2.set_title('Resilience Against Physical Occlusion', fontsize=10)
    ax2.set_ylim(90, 101) # Zoom vào vùng quan trọng
    ax2.legend(loc='lower left', frameon=True, fontsize=8)
    
    plt.savefig('../figures/Figure_3_Occlusion_Robustness.pdf', dpi=600, bbox_inches='tight', format='pdf')
    print("Saved Figure 3: Occlusion Robustness")

if __name__ == '__main__':
    create_plots()
