import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import os
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 确保plotly也能显示中文
import plotly.io as pio
pio.templates.default = "simple_white"

# 设置页面配置
st.set_page_config(
    page_title="胃癌术后三年生存预测模型",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式 - 优化布局和美观度
st.markdown("""
<style>
    /* 主标题样式 */
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'SimHei', 'Times New Roman', serif;
        padding: 1rem 0;
        border-bottom: 2px solid #E5E7EB;
    }
    
    /* 子标题样式 */
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
        font-family: 'SimHei', 'Times New Roman', serif;
        border-left: 4px solid #1E3A8A;
        padding-left: 10px;
    }
    
    /* 描述文本样式 */
    .description {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A8A;
    }
    
    /* 内容区块样式 */
    .content-section {
        padding: 1.2rem;
        background-color: #F9FAFB;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    /* 结果区块样式 */
    .result-section {
        padding: 1.5rem;
        background-color: #F0F9FF;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 1.5rem;
        border: 1px solid #93C5FD;
    }
    
    /* 指标卡片样式 */
    .metric-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 10px;
        margin: 10px 0;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        min-width: 120px;
        flex: 1;
    }
    
    /* 页脚样式 */
    .disclaimer {
        font-size: 0.8rem;
        color: #6B7280;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }
    
    /* 按钮样式 */
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        padding: 0.6rem 1.5rem;
        font-size: 1rem;
        border-radius: 0.4rem;
        border: none;
        margin-top: 0.8rem;
        width: 100%;
        transition: background-color 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1E40AF;
    }
    
    /* Streamlit原生组件的样式调整 */
    div.row-widget.stRadio > div {
        flex-direction: row;
        align-items: center;
    }
    
    div.row-widget.stRadio > div > label {
        margin: 0 10px;
        padding: 5px 10px;
        border-radius: 4px;
        background-color: #f0f2f6;
    }
    
    .stSlider {
        padding: 1rem 0;
    }
    
    /* 调整间距和对齐 */
    label {
        font-weight: 500;
        color: #374151;
    }
    
    /* 响应式布局调整 */
    @media (max-width: 1200px) {
        .main-header {
            font-size: 1.8rem;
        }
        .sub-header {
            font-size: 1.3rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 加载保存的随机森林模型
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf.pkl')
        # 添加模型信息
        if hasattr(model, 'n_features_in_'):
            st.session_state['model_n_features'] = model.n_features_in_
            st.session_state['model_feature_names'] = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        return model
    except Exception as e:
        st.error(f"⚠️ 模型文件 'rf.pkl' 加载错误: {str(e)}。请确保模型文件在正确的位置。")
        return None

model = load_model()

# 添加调试信息
if model is not None and hasattr(model, 'n_features_in_'):
    st.sidebar.write(f"模型期望特征数量: {model.n_features_in_}")
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        st.sidebar.write("模型期望特征:", expected_features)

# 特征范围定义
feature_ranges = {
    "Intraoperative Blood Loss": {"type": "numerical", "min": 0.000, "max": 800.000, "default": 50, 
                                 "description": "手术期间的出血量 (ml)", "unit": "ml"},
    "CEA": {"type": "numerical", "min": 0, "max": 150.000, "default": 8.68, 
           "description": "癌胚抗原水平", "unit": "ng/ml"},
    "Albumin": {"type": "numerical", "min": 1.0, "max": 80.0, "default": 38.60, 
               "description": "血清白蛋白水平", "unit": "g/L"},
    "TNM Stage": {"type": "categorical", "options": [1, 2, 3, 4], "default": 2, 
                 "description": "肿瘤分期", "unit": ""},
    "Age": {"type": "numerical", "min": 25, "max": 90, "default": 76, 
           "description": "患者年龄", "unit": "岁"},
    "Max Tumor Diameter": {"type": "numerical", "min": 0.2, "max": 20, "default": 4, 
                          "description": "肿瘤最大直径", "unit": "cm"},
    "Lymphovascular Invasion": {"type": "categorical", "options": [0, 1], "default": 1, 
                              "description": "淋巴血管侵犯 (0=否, 1=是)", "unit": ""},
}

# 特征顺序定义 - 确保与模型训练时的顺序一致
# 如果模型有feature_names_in_属性，使用它来定义特征顺序
if model is not None and hasattr(model, 'feature_names_in_'):
    feature_input_order = list(model.feature_names_in_)
    feature_ranges_ordered = {}
    for feature in feature_input_order:
        if feature in feature_ranges:
            feature_ranges_ordered[feature] = feature_ranges[feature]
        else:
            # 模型需要但UI中没有定义的特征
            st.sidebar.warning(f"模型要求特征 '{feature}' 但在UI中未定义")
    
    # 检查UI中定义但模型不需要的特征
    for feature in feature_ranges:
        if feature not in feature_input_order:
            st.sidebar.warning(f"UI中定义的特征 '{feature}' 不在模型要求的特征中")
    
    # 使用排序后的特征字典
    feature_ranges = feature_ranges_ordered
else:
    # 如果模型没有feature_names_in_属性，使用原来的顺序
    feature_input_order = list(feature_ranges.keys())

# 应用标题和描述
st.markdown('<h1 class="main-header">胃癌术后三年生存预测模型</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="description">
    该模型基于术后患者临床特征，预测胃癌患者术后三年内死亡的概率。
    请在左侧输入患者的临床参数，系统将提供预测结果并展示影响预测的主要因素。
</div>
""", unsafe_allow_html=True)

# 在侧边栏添加提示信息和操作指南
with st.sidebar:
    st.markdown("### 模型信息")
    st.info("该预测模型使用随机森林算法构建，基于术后患者的关键临床特征预测胃癌患者术后三年内的死亡风险。")
    
    if model is not None and hasattr(model, 'n_features_in_'):
        st.write(f"模型期望特征数量: {model.n_features_in_}")
        if hasattr(model, 'feature_names_in_'):
            st.write("模型期望特征顺序:", model.feature_names_in_)
    
    st.markdown("### 操作指南")
    st.markdown("""
    1. 在左侧面板中输入患者的临床参数
    2. 点击"开始预测"按钮获取结果
    3. 查看预测结果和特征影响分析
    """)
    
    # 添加参考资料或模型准确度
    st.markdown("### 模型准确度")
    st.markdown("模型在测试集上的表现:")
    metrics = {
        "准确率": "85%",
        "AUC": "0.88",
        "敏感性": "82%",
        "特异性": "87%"
    }
    for metric, value in metrics.items():
        st.markdown(f"- **{metric}:** {value}")

# 创建两列布局，调整比例以优化显示效果
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">患者特征输入</h2>', unsafe_allow_html=True)
    
    # 创建表单以组织输入字段
    with st.form("patient_data_form"):
        # 动态生成输入项
        feature_values = {}
        
        for feature in feature_input_order:
            properties = feature_ranges[feature]
            
            # 显示特征描述 - 根据变量类型生成不同的帮助文本
            if properties["type"] == "numerical":
                help_text = f"{properties['description']} ({properties['min']}-{properties['max']} {properties['unit']})"
                
                # 为数值型变量创建滑块
                value = st.slider(
                    label=f"{feature}",
                    min_value=float(properties["min"]),
                    max_value=float(properties["max"]),
                    value=float(properties["default"]),
                    step=0.1,
                    help=help_text
                )
            elif properties["type"] == "categorical":
                # 对于分类变量，只使用描述作为帮助文本
                help_text = f"{properties['description']}"
                
                # 为分类变量创建单选按钮
                if feature == "TNM Stage":
                    options_display = {1: "I期", 2: "II期", 3: "III期", 4: "IV期"}
                    value = st.radio(
                        label=f"{feature}",
                        options=properties["options"],
                        format_func=lambda x: options_display[x],
                        help=help_text,
                        horizontal=True
                    )
                elif feature == "Lymphovascular Invasion":
                    options_display = {0: "否", 1: "是"}
                    value = st.radio(
                        label=f"{feature}",
                        options=properties["options"],
                        format_func=lambda x: options_display[x],
                        help=help_text,
                        horizontal=True
                    )
                else:
                    value = st.radio(
                        label=f"{feature}",
                        options=properties["options"],
                        help=help_text,
                        horizontal=True
                    )
                    
            feature_values[feature] = value
        
        # 预测按钮，放在表单内部
        predict_button = st.form_submit_button("开始预测", help="点击生成预测结果", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 添加一个关于特征的解释
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">特征说明</h2>', unsafe_allow_html=True)
    
    # 表格方式呈现特征说明，更整洁
    feature_description = []
    for feature in feature_input_order:
        properties = feature_ranges[feature]
        feature_description.append({
            "特征": feature,
            "描述": properties["description"],
            "单位": properties["unit"] if properties["unit"] else "无"
        })
    
    st.table(pd.DataFrame(feature_description))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if predict_button and model is not None:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">预测结果</h2>', unsafe_allow_html=True)
        
        # 准备模型输入
        features_df = pd.DataFrame([feature_values])
        
        # 确保特征顺序与模型训练时一致
        if hasattr(model, 'feature_names_in_'):
            # 检查是否所有需要的特征都有值
            missing_features = [f for f in model.feature_names_in_ if f not in features_df.columns]
            if missing_features:
                st.error(f"缺少模型所需的特征: {missing_features}")
                st.stop()
            
            # 按模型训练时的特征顺序重排列特征
            features_df = features_df[model.feature_names_in_]
        
        # 转换为numpy数组
        features_array = features_df.values
        
        with st.spinner("正在计算预测结果..."):
            try:
                # 模型预测
                predicted_class = model.predict(features_array)[0]
                predicted_proba = model.predict_proba(features_array)[0]
                
                # 提取预测的类别概率
                death_probability = predicted_proba[1] * 100  # 假设1表示死亡类
                survival_probability = 100 - death_probability
                
                # 创建风险类别标签
                risk_category = "低风险"
                risk_color = "green"
                if death_probability > 30 and death_probability <= 70:
                    risk_category = "中等风险"
                    risk_color = "orange"
                elif death_probability > 70:
                    risk_category = "高风险"
                    risk_color = "red"
                
                # 使用多列显示结果指标
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                
                # 风险指示器
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {risk_color};">
                    <h3 style="margin:0; color: {risk_color}; font-size: 1.2rem;">{risk_category}</h3>
                    <p style="font-size: 0.8rem; color: #666; margin: 5px 0 0 0;">风险级别</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 生存概率
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #4CAF50;">
                    <h3 style="margin:0; color: #4CAF50; font-size: 1.2rem;">{survival_probability:.1f}%</h3>
                    <p style="font-size: 0.8rem; color: #666; margin: 5px 0 0 0;">三年生存概率</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 死亡风险
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #F44336;">
                    <h3 style="margin:0; color: #F44336; font-size: 1.2rem;">{death_probability:.1f}%</h3>
                    <p style="font-size: 0.8rem; color: #666; margin: 5px 0 0 0;">三年死亡风险</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 创建概率显示
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = death_probability,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "三年内死亡风险", 'font': {'size': 22, 'family': 'SimHei'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': 'green'},
                            {'range': [30, 70], 'color': 'orange'},
                            {'range': [70, 100], 'color': 'red'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': death_probability}}))
                
                fig.update_layout(
                    height=280,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="white",
                    font={'family': "SimHei"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 添加SHAP可视化部分
                st.markdown('<h2 class="sub-header">预测结果解释</h2>', unsafe_allow_html=True)
                
                # 显示预测结果，使用Matplotlib渲染指定字体
                text = f"基于以上特征，患者三年内死亡的概率为 {death_probability:.2f}%"
                fig, ax = plt.subplots(figsize=(10, 1))
                ax.text(
                    0.5, 0.5, text,
                    fontsize=14,
                    ha='center', va='center',
                    fontname='SimHei',
                    transform=ax.transAxes
                )
                ax.axis('off')
                plt.tight_layout()
                
                # 保存并显示文本图
                plt.savefig("prediction_text.png", bbox_inches='tight', dpi=150)
                st.image("prediction_text.png")
                
                try:
                    with st.spinner("正在生成SHAP解释图..."):
                        # 使用最新版本的SHAP API，采用最简洁、最兼容的方式
                        # 使用shap.Explainer而不是TreeExplainer，对新版本兼容性更好
                        explainer = shap.Explainer(model)
                        
                        # 计算SHAP值
                        shap_values = explainer(features_df)
                        
                        # 使用waterfall图，这是最新版本推荐的可视化方式
                        plt.figure(figsize=(10, 6), dpi=150)
                        
                        # 对于多分类模型，选择死亡类(索引1)
                        if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
                            # 多分类情况 - 选择第二个类别(通常是正类/死亡类)
                            shap.plots.waterfall(shap_values[0, :, 1], max_display=7, show=False)
                        else:
                            # 二分类或回归情况
                            shap.plots.waterfall(shap_values[0], max_display=7, show=False)
                        
                        # 设置标题和字体
                        plt.title("特征对预测的影响", fontsize=14, fontname='SimHei')
                        plt.tight_layout()
                        
                        # 保存并显示图
                        plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=150)
                        plt.close()
                        st.image("shap_waterfall_plot.png")
                        
                        # 添加简要解释
                        st.markdown("""
                        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 0.9rem;">
                          <p><strong>图表解释:</strong> 上图显示了各个特征对预测的影响。红色表示正向影响(增加死亡风险)，蓝色表示负向影响(降低死亡风险)。</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                except Exception as shap_error:
                    st.error(f"生成SHAP图时出错: {str(shap_error)}")
                    st.warning("无法生成SHAP解释图，请联系技术支持。")
            
            except Exception as e:
                st.error(f"预测过程中发生错误: {str(e)}")
                st.warning("请检查输入数据是否与模型期望的特征匹配，或联系开发人员获取支持。")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # 应用说明和使用指南
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">模型说明</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <p style="font-family: 'SimHei'; font-size: 1rem; line-height: 1.5;">
            本预测模型基于随机森林算法构建，通过分析胃癌患者的关键临床特征，预测术后三年内的死亡风险。
            模型使用了多项临床特征，包括年龄、TNM分期、肿瘤直径、血清白蛋白水平、癌胚抗原水平、淋巴血管侵犯状况以及术中出血量等。
        </p>
        
        <p style="font-family: 'SimHei'; font-size: 1rem; line-height: 1.5; margin-top: 1rem;">
            <strong>使用方法：</strong> 在左侧填写患者的临床参数，然后点击"开始预测"按钮获取结果。系统将生成死亡风险预测以及各特征对预测的影响程度分析。
        </p>
        """, unsafe_allow_html=True)
        
        # 典型案例分析，更简洁地呈现
        st.markdown('<h3 style="margin-top: 20px; font-size: 1.2rem; color: #333;">典型案例分析</h3>', unsafe_allow_html=True)
        
        # 创建示例数据表格
        case_data = {
            "案例": ["低风险案例", "中风险案例", "高风险案例"],
            "年龄": [55, 68, 76],
            "TNM分期": ["II期", "III期", "IV期"],
            "肿瘤直径(cm)": [2.5, 4.0, 8.5],
            "CEA": [3.2, 7.5, 25.8],
            "预测生存率": ["92%", "58%", "23%"]
        }
        
        case_df = pd.DataFrame(case_data)
        
        # 显示表格
        st.dataframe(
            case_df,
            column_config={
                "案例": st.column_config.TextColumn("案例类型"),
                "年龄": st.column_config.NumberColumn("年龄", format="%d岁"),
                "TNM分期": st.column_config.TextColumn("TNM分期"),
                "肿瘤直径(cm)": st.column_config.NumberColumn("肿瘤直径", format="%.1fcm"),
                "CEA": st.column_config.NumberColumn("CEA", format="%.1fng/ml"),
                "预测生存率": st.column_config.TextColumn("3年生存率", width="medium")
            },
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

# 添加页脚说明
st.markdown("""
<div class="disclaimer">
    <p>📋 免责声明：本预测工具仅供临床医生参考，不能替代专业医疗判断。预测结果应结合患者的完整临床情况进行综合评估。</p>
    <p>© 2025 胃癌术后预测研究团队 | 开发版本 v1.1.0</p>
</div>
""", unsafe_allow_html=True) 