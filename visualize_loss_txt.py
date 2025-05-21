# 使用Pandas进行数据分析
import pandas as pd

import numpy as np
from scipy.signal import savgol_filter
import plotly.graph_objects as go


# path = "/home/hjx/hjx_file/MuJoCo/act-mujoco-sim/dataset/ckpt/sim_insertion_scripted/train_val_kl_seed_0.txt"
path = "/home/hjx/hjx_file/MuJoCo/act-mujoco-sim/dataset/ckpt/sim_insertion_scripted/train_val_l1_seed_0.txt"
# path = "/home/hjx/hjx_file/MuJoCo/act-mujoco-sim/dataset/ckpt/sim_insertion_scripted/train_val_loss_seed_0.txt"

df = pd.read_csv(path, sep='\t')

# test code
# 绘制交互式图表
# import plotly.express as px
# fig = px.line(df, x="epoch", y="value", color="data_type")
# fig.show()



# last code
# ==== 学术图表配置 ====
ACADEMIC_CONFIG = {
    "font": {"family": "Times New Roman", "size": 18},
    "colors": {"train": "#1f77b4", "validation": "#ff7f0e"},
    "line": {"width": 3, "dash": "solid"},
    "figure": {"width": 1200, "height": 800, "dpi": 300},
    "grid": {"color": "#D3D3D3", "width": 1},
    "legend": {"x": 0.78, "y": 0.95}
}


def validate_data(df):
    """数据校验函数"""
    required_columns = ['data_type', 'epoch', 'value']
    missing = set(required_columns) - set(df.columns)

    if missing:
        available = ", ".join(df.columns)
        raise KeyError(f"缺失必要列: {missing}，当前列名: {available}\n"
                       f"请检查：1.数据分隔符是否为制表符 2.列名是否包含空格")

    # 检查数据类型
    if not np.issubdtype(df['epoch'].dtype, np.number):
        raise ValueError("epoch列包含非数值数据")

    return df


def load_and_process(path):
    """增强版数据加载"""
    try:
        # 尝试不同编码格式读取
        for encoding in ['utf-8', 'gbk']:
            try:
                df = pd.read_csv(path, sep='\t', engine='python', encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("无法解码文件，尝试转换文件编码为UTF-8")

        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # 去除空格
        df = validate_data(df)  # 校验数据

        # 自动识别训练/验证数据
        df['data_type'] = df['data_type'].str.lower().str.strip()
        valid_types = {'train', 'validation'}
        if not set(df['data_type'].unique()).issubset(valid_types):
            found = df['data_type'].unique()
            raise ValueError(f"未知的数据类型: {found}，应为train/validation")

        # 数据类型转换
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # 滑动窗口平滑
        def smooth_group(group):
            n = len(group)
            window = min(15, n // 2)
            if window % 2 == 0:
                window = max(3, window - 1)
            if window > 3:
                group['smoothed'] = savgol_filter(group['value'], window, 3)  # naive: 3
            else:  # 数据点太少时直接使用原始值
                group['smoothed'] = group['value']
            return group

        return df.groupby('data_type', group_keys=False).apply(smooth_group)

    except Exception as e:
        print(f"数据处理失败: {str(e)}")
        print("调试信息：")
        print(f"文件路径: {path}")
        print("前5行数据:")
        print(df.head() if 'df' in locals() else "无法读取数据")
        raise


def create_plot(df):
    """学术风格强化版绘图函数"""
    fig = go.Figure()

    color_map = {
        'train': ACADEMIC_CONFIG['colors']['train'],
        'validation': ACADEMIC_CONFIG['colors']['validation']
    }

    for dtype in ['train', 'validation']:
        subset = df[df['data_type'] == dtype]
        if subset.empty:
            continue

        # 原始数据（设置更专业的透明度）
        fig.add_trace(go.Scatter(
            x=subset['epoch'], y=subset['value'],
            name=f'{dtype} Raw',
            line=dict(
                color=color_map[dtype],
                width=1.5,
                dash='dot'
            ),
            opacity=0.2,
            hoverinfo='x+y+name',
            showlegend=False
        ))

        # 平滑数据（优化线型）
        fig.add_trace(go.Scatter(
            x=subset['epoch'], y=subset['smoothed'],
            name=f'{dtype}',
            line=dict(
                color=color_map[dtype],
                width=ACADEMIC_CONFIG['line']['width'] + 0.5,
                shape='spline'  # 添加自然样条曲线
            ),
            mode='lines'
        ))

    fig.update_layout(
        template='none',  # 禁用所有默认样式
        font=dict(
            family=ACADEMIC_CONFIG['font']['family'],
            size=ACADEMIC_CONFIG['font']['size'],
            color='black'
        ),
        xaxis=dict(
            title='Training Epochs',
            showgrid=False,  # 彻底移除网格线
            showline=True,
            linecolor='black',
            linewidth=1.5,
            mirror=True,  # 显示顶部轴线
            ticks='outside',
            tickwidth=2,
            zeroline=False
        ),
        yaxis=dict(
            title='Loss Value',
            showgrid=False,
            showline=True,
            linecolor='black',
            linewidth=1.5,
            mirror=True,  # 显示右侧轴线
            ticks='outside',
            tickwidth=2,
            zeroline=False
        ),
        legend=dict(
            x=0.78,
            y=0.95,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=16)
        ),
        plot_bgcolor='white',  # 设置纯白背景
        paper_bgcolor='white',
        width=1200,
        height=800
    )

    # 添加刻度线扩展
    fig.update_xaxes(tickson="boundaries", ticklen=8)
    fig.update_yaxes(tickson="boundaries", ticklen=8)

    return fig


if __name__ == "__main__":

    # data_path = "/home/hjx/hjx_file/MuJoCo/act-mujoco-sim/dataset/ckpt/sim_insertion_scripted/train_val_kl_seed_0.txt"
    # data_path = "/home/hjx/hjx_file/MuJoCo/act-mujoco-sim/dataset/ckpt/sim_insertion_scripted/train_val_l1_seed_0.txt"
    # data_path = "/home/hjx/hjx_file/MuJoCo/act-mujoco-sim/dataset/ckpt/sim_insertion_scripted/train_val_loss_seed_0.txt"

    # data_path = "/home/hjx/hjx_file/MuJoCo/act-mujoco-sim/dataset/ckpt/sim_transfer_cube_scripted/train_val_kl_seed_0.txt"
    data_path = "/home/hjx/hjx_file/MuJoCo/act-mujoco-sim/dataset/ckpt/sim_transfer_cube_scripted/train_val_l1_seed_0.txt"
    # data_path = "/home/hjx/hjx_file/MuJoCo/act-mujoco-sim/dataset/ckpt/sim_transfer_cube_scripted/train_val_loss_seed_0.txt"

    try:
        processed_df = load_and_process(data_path)
        fig = create_plot(processed_df)
        fig.show()

        # 保存图片
        fig.write_image("training_curve.png", scale=2, engine="kaleido")
        print("可视化完成，图片已保存")

    except Exception as e:
        print(f"运行失败: {str(e)}")
        print("排查建议：")
        print("1. 用文本编辑器确认文件内容")
        print("2. 检查文件分隔符是否为制表符（推荐用Excel查看）")
        print("3. 运行前执行：sed -i 's/\r//g' your_data.txt （处理Windows换行符）")
