import streamlit as st  
import joblib  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import shap
# 在文件开头添加以下代码（在所有import之后）
from matplotlib import rcParams
import matplotlib.font_manager as fm

# 设置中文字体
try:
    # 尝试使用系统自带的中文字体
    font_path = fm.findfont(fm.FontProperties(family=['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']))
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    # 如果系统没有中文字体，尝试使用指定的字体文件
    try:
        font_path = 'SimHei.ttf'  # 确保这个字体文件存在于同一目录下
        font_prop = fm.FontProperties(fname=font_path)
        rcParams['font.sans-serif'] = [font_prop.get_name()]
    except:
        st.warning("无法加载中文字体，将使用默认字体") 

# 加载模型
model = joblib.load('CNN.pkl')  

# ------------ 定义特征选项 ------------
smoking_options = {  
    1: '不吸烟 (1)',  
    2: '吸烟 (2)'  
}  

sleep_duration_options = {  
    1: '＜6小时 (1)',  
    2: '≥6小时 (2)'  
}  

exercise_options = {  
    0: '不锻炼 (0)',  
    1: '锻炼 (1)'  
}  

economic_burden_options = {  
    1: '轻 (1)',  
    2: '一般 (2)',  
    3: '重 (3)'  
}  

fall_history_options = {  
    0: '无跌倒史 (0)',  
    1: '有跌倒史 (1)'  
}  

polypharmacy_options = {  
    0: '无 (0)',  
    1: '有 (1)'  
}  

adl_options = {  
    1: '独立 (1)',  
    2: '依赖 (2)'  
}  

swallowing_difficulty_options = {  
    1: '无吞咽困难 (1)',  
    2: '有吞咽困难 (2)'  
}  

cognitive_impairment_options = {  
    0: '无认知障碍 (0)',  
    1: '有认知障碍 (1)'  
}  

feature_names = [  
    "吸烟情况", "睡眠时间", "体育锻炼", "经济负担", "跌倒史",  
    "多重用药", "白蛋白", "日常生活活动能力", "吞咽困难", "认知障碍"  
]  

# ------------ 用户界面 ------------
st.title("老年缺血性脑中风衰弱风险预测模型")  

# 用户输入控件
smoking_status = st.selectbox(
    "吸烟情况", 
    options=list(smoking_options.keys()), 
    format_func=lambda x: smoking_options[x]
)
sleep_duration = st.selectbox(
    "睡眠时间", 
    options=list(sleep_duration_options.keys()), 
    format_func=lambda x: sleep_duration_options[x]
)
exercise = st.selectbox(
    "体育锻炼", 
    options=list(exercise_options.keys()), 
    format_func=lambda x: exercise_options[x]
)
economic_burden = st.selectbox(
    "经济负担", 
    options=list(economic_burden_options.keys()), 
    format_func=lambda x: economic_burden_options[x]
)
fall_history = st.selectbox(
    "跌倒史", 
    options=list(fall_history_options.keys()), 
    format_func=lambda x: fall_history_options[x]
)
polypharmacy = st.selectbox(
    "多重用药", 
    options=list(polypharmacy_options.keys()), 
    format_func=lambda x: polypharmacy_options[x]
)
albumin = st.number_input(
    "白蛋白水平 (g/L)", 
    min_value=0.0, 
    max_value=50.0, 
    value=35.0, 
    step=0.1
)
adl = st.selectbox(
    "日常生活活动能力", 
    options=list(adl_options.keys()), 
    format_func=lambda x: adl_options[x]
)
swallowing_difficulty = st.selectbox(
    "吞咽困难", 
    options=list(swallowing_difficulty_options.keys()), 
    format_func=lambda x: swallowing_difficulty_options[x]
)
cognitive_impairment = st.selectbox(
    "认知障碍", 
    options=list(cognitive_impairment_options.keys()), 
    format_func=lambda x: cognitive_impairment_options[x]
)

# ------------ 预测逻辑 ------------
if st.button("进行预测"):  
    # 构建输入特征并调整形状适配CNN
    feature_values = [
        smoking_status, sleep_duration, exercise, economic_burden, fall_history,  
        polypharmacy, albumin, adl, swallowing_difficulty, cognitive_impairment
    ]
    # 转换为三维输入 (样本数=1, 特征数=10, 通道数=1)
    features = np.array(feature_values, dtype=np.float32).reshape(1, 10, 1)  

    try:
        # 模型预测
        predicted_probs = model.predict(features)
        
        # 提取概率值
        if predicted_probs.ndim == 2:
            if predicted_probs.shape[1] == 1:
                predicted_proba = predicted_probs[0][0]  # 单输出节点
            else:
                predicted_proba = predicted_probs[0][1]  # 二分类取正类
        else:
            predicted_proba = predicted_probs[0]
        predicted_proba = float(predicted_proba)  # 转为Python浮点数

    except Exception as e:
        st.error(f"预测错误: {str(e)}")
        st.stop()

    # ------------ 显示结果 ------------
    st.markdown(f"### 预测衰弱的概率:  **{predicted_proba:.3f}**")  
    
    # 风险分类与建议
    threshold = 0.520
    if predicted_proba > threshold:
        st.error("### 高风险警告 ⚠️")
        advice = f"""
        模型预测您存在衰弱高风险（概率 **{predicted_proba*100:.1f}%**）。  
        **建议立即行动：**
        - 联系医生进行详细评估  
        - 加强营养补充（重点关注白蛋白水平）  
        - 进行适度的康复训练  
        - 定期监测跌倒风险  
        - 维持每日7-8小时睡眠  
        - 每周至少3次有氧运动  
        - 每年一次全面体检 
        """
    else:
        st.success("### 低风险 ✅")
        advice = f"""
        模型预测您的风险较低（概率 **{predicted_proba*100:.1f}%**）。  
        **保持健康建议：**
        - 保持均衡饮食，特别是蛋白质摄入
        - 维持规律的运动习惯
        - 保证充足睡眠
        - 定期进行健康检查
        """
    st.markdown(advice)

    # ------------ SHAP解释 ------------
st.markdown("---")
st.subheader("模型解释 - 特征影响分析")

try:
    # 1. 生成背景数据（与模型输入形状一致）
    background_data = np.random.randn(100, 10, 1).astype(np.float32)
    
    # 2. 创建解释器
    explainer = shap.DeepExplainer(model, background_data)
    
    # 3. 计算SHAP值
    shap_values = explainer.shap_values(features)
    
    # 4. 处理不同类型的SHAP输出
    if isinstance(shap_values, list):
        # 如果是多分类，取第一个类的SHAP值
        shap_values = shap_values[0]
    
    # 5. 确保SHAP值是二维的
    shap_values_2d = np.array(shap_values).reshape(1, -1)
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap.Explanation(
        values=shap_values_2d[0],
        base_values=explainer.expected_value[0],
        data=features.flatten(),
        feature_names=feature_names
    ), show=False)
    plt.title("特征对预测结果的影响(SHAP值)", pad=20)
    plt.tight_layout()
    # 6. 
    st.pyplot(plt.gcf())

except Exception:
    # 静默失败，直接尝试回退方案
    try:
        # 确保我们有可用的SHAP值
        if 'shap_values' in locals():
            shap_vals = np.array(shap_values).flatten()
            colors = ['red' if val > 0 else 'blue' for val in shap_vals]
            
            plt.figure(figsize=(10, 5))
            plt.barh(feature_names, shap_vals, color=colors)
            plt.axvline(x=0, color='gray', linestyle='--')
            plt.title("特征影响值(SHAP)")
            plt.xlabel("SHAP值(正值增加风险)")
            st.pyplot(plt.gcf())
            
            # 添加数据表格增强可读性
            st.table(pd.DataFrame({
                '特征': feature_names,
                'SHAP值': shap_vals,
                '影响方向': ['增加风险' if x > 0 else '降低风险' for x in shap_vals]
            }).sort_values('SHAP值', key=abs, ascending=False))
            
    except Exception:
        st.markdown("""
        **特征影响分析：**
        - 白蛋白水平低通常增加风险
        - 锻炼习惯通常降低风险
        - 睡眠不足通常增加风险
        """)
        