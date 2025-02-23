import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# โหลดโมเดล Machine Learning
svm_air_quality_model = joblib.load('air_quality_svm_model.pkl')
air_quality_scaler = joblib.load('air_quality_scaler.pkl')
# โหลดโมเดลที่สอง (Random Forest)
rf_air_quality_model = joblib.load('air_quality_rf_model.pkl')

# โหลดโมเดล Neural Network
mlp_fruit_model = joblib.load('fruit_mlp_model.pkl')
fruit_scaler = joblib.load('fruit_scaler.pkl')
fruit_label_encoder = joblib.load('fruit_label_encoder.pkl')

# สร้างแอป Streamlit
st.title("Machine Learning & Neural Network Web Application")

# เมนูนำทาง
menu = [
    "Model Development Overview", 
    "SVM Model Development", 
    "MLP Model Development",
    "Air Quality Prediction (ML)", 
    "Fruit Classification (NN)", 
    "Batch Prediction"
]
choice = st.sidebar.selectbox("Select Page", menu)

# ----------------------------------------------------------------------
# Page 1: Model Development Overview
if choice == "Model Development Overview":
    st.header("Model Development Overview")
    
    st.write("""
    ## 🔹 Overview
    This section provides an **overall explanation** of the development process for both **SVM** and **MLP models**, including:
    - **Data Preparation**
    - **Theoretical Background**
    - **General Model Development Process**
    
    ---

    ## Data Preparation
    High-quality data is the foundation of a successful model. The following steps were followed:
    - **Data Acquisition**: The dataset was generated using ChatGPT for diverse and realistic input scenarios.
    - **Data Cleaning**: Irrelevant or duplicate data was removed, ensuring accuracy.
    - **Feature Selection**: Only the most influential features were used for training.
    - **Data Scaling**: Standardization and normalization techniques were applied for balanced data processing.

    ---

    ## Theoretical Background
    **Support Vector Machine (SVM)**
    - Finds an optimal hyperplane to separate data classes.
    - Effective for **structured, classification-based tasks**.
    
    **Multilayer Perceptron (MLP)**
    - A type of artificial neural network with multiple layers.
    - Best for **complex, deep-learning-based pattern recognition**.

    ---

    ## Model Development Process
    - **SVM** is used for air quality classification.
    - **MLP** is used for fruit classification.
    - Both models underwent **data processing, hyperparameter tuning, and performance evaluation**.

    **For details on each model, navigate to the specific SVM or MLP development pages.**
    """)

# ----------------------------------------------------------------------
# Page 2: Support Vector Machine (SVM) Model Development
if choice == "SVM Model Development":
    st.header("Support Vector Machine (SVM) Model Development")
    
    st.write("""
    ## 🔹 Overview
    The **SVM model** was developed to classify air quality into categories such as "Good," "Moderate," and "Poor."

    ---  
    ## Theoretical Background of SVM  
    - **Concept**: SVM is a supervised learning algorithm used for classification and regression.  
    - **Hyperplane**: SVM finds the best **decision boundary** that separates different classes.
    - **Mathematical Representation**:
        - Decision function:  
        \[
        f(x) = w \cdot x + b
        \]
        - **Maximizing the Margin**: The goal is to maximize the distance between the hyperplane and the nearest data points.

    ### 🔹 **SVM Hyperplane Visualization**
    - The image below illustrates **how SVM finds an optimal hyperplane to separate different classes**:
    """)

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/450px-SVM_margin.png",
        caption="SVM Hyperplane: Maximizing Margin Between Two Classes",
        use_column_width=True
    )

    st.write("""
    - **Kernel Trick**:
        - If data is **not linearly separable**, SVM **maps it to a higher-dimensional space** using **kernels** such as:
            - **Linear Kernel**: Used for simple linear separability.
            - **Polynomial Kernel**: Useful for more complex decision boundaries.
            - **RBF (Radial Basis Function) Kernel**: Handles non-linear relationships.

    ---  
    ## Model Development Process for SVM
    ### **Step 1: Data Preprocessing**
    - **Data Cleaning**: Removed missing values and standardized numerical features.
    - **Feature Selection**: Selected key features like PM2.5, PM10, temperature, humidity, and wind speed.

    ### **Step 2: Model Selection**
    - Chose **SVM** due to its effectiveness in classification.
    - Used **RBF Kernel** for better adaptability to complex patterns.

    ### **Step 3: Hyperparameter Tuning**
    - Used **Grid Search** to find the best combination of C (Regularization parameter) and gamma.
    - Applied **k-fold Cross-Validation** to ensure the model generalizes well.

    ### **Step 4: Training and Evaluation**
    - Trained the model using the preprocessed dataset.
    - Evaluated using **Accuracy, Precision, Recall, and F1-score**.

    📚 **References**  
    - [Support Vector Machines: Theory and Applications](https://link.springer.com/book/10.1007/b138428)  
    - [Scikit-learn SVM documentation](https://scikit-learn.org/stable/modules/svm.html)  
    """)

# ----------------------------------------------------------------------
# Page 3: Multilayer Perceptron (MLP) Model Development
if choice == "MLP Model Development":
    st.header("Multilayer Perceptron (MLP) Model Development")
    
    st.write("""
    ## 🔹 Overview
    The **MLP model** was developed for **fruit classification**, predicting categories such as "Apple," "Banana," and "Orange."

    ---  
    ## Theoretical Background of MLP  
    - **Concept**: Multilayer Perceptron (MLP) is a type of artificial neural network with multiple layers.  
    - **Network Architecture**:
        - **Input Layer**: Takes numerical feature inputs.
        - **Hidden Layers**: Apply activation functions like **ReLU (Rectified Linear Unit)** to introduce non-linearity.
        - **Output Layer**: Uses **Softmax (for multi-class classification)** or **Sigmoid (for binary classification)** to generate the final prediction.

    ### 🔹 **MLP Network Structure Visualization**
    - The following image illustrates **the structure of a Multilayer Perceptron (MLP)**:
    """)

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Artificial_neural_network.svg/600px-Artificial_neural_network.svg.png",
        caption="MLP Structure: Input Layer, Hidden Layers, and Output Layer",
        use_column_width=True
    )

    st.write("""
    - **Backpropagation Algorithm**:
        - A method to update weights using **gradient descent** by minimizing the loss function.
        - Works with optimizers like **SGD (Stochastic Gradient Descent)** and **Adam**.

    ---  
    ## Model Development Process for MLP
    ### **Step 1: Data Preprocessing**
    - **Data Cleaning**: Removed missing values and normalized numerical features.
    - **Feature Engineering**: Selected relevant features like weight, length, circumference, and color.

    ### **Step 2: Network Architecture Design**
    - **Input Layer**: Receives numerical features.
    - **Hidden Layers**: Uses **ReLU activation** to introduce non-linearity.
    - **Output Layer**: Uses **Softmax activation** for multi-class classification.

    ### **Step 3: Model Training**
    - Used **Backpropagation** to adjust weights and minimize the loss function.
    - Optimized using **Adam optimizer** for better performance.

    ### **Step 4: Evaluation and Optimization**
    - Evaluated the model using **categorical accuracy and loss reduction analysis**.
    - Used **Dropout Regularization** to prevent overfitting.

    📚 **References**  
    - [Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](https://www.deeplearningbook.org/)  
    - [TensorFlow documentation](https://www.tensorflow.org/overview/)  
    """)

# ----------------------------------------------------------------------
# ฟังก์ชันสำหรับ input แบบทั้งกรอกและเลื่อนค่าได้
def hybrid_input(label, min_value, max_value, default):
    return st.number_input(label, min_value=min_value, max_value=max_value, value=default, step=0.1, format="%.2f")

# ----------------------------------------------------------------------
# Page 4: Air Quality Prediction (ML)
if choice == "Air Quality Prediction (ML)":
    st.header("Air Quality Prediction (Machine Learning)")

    # ให้ผู้ใช้เลือกโมเดลที่ต้องการเปรียบเทียบ
    selected_models = st.multiselect(
        "Select Models for Prediction",
        ["SVM (Current)", "Random Forest"],
        default=["SVM (Current)"]
    )

    # ใช้ฟังก์ชัน hybrid_input เพื่อรับค่าอินพุต
    pm25 = hybrid_input("PM2.5", 0.0, 300.0, 50.0)
    pm10 = hybrid_input("PM10", 0.0, 300.0, 100.0)
    temp = hybrid_input("Temperature (C)", -50.0, 50.0, 25.0)
    humidity = hybrid_input("Humidity (%)", 0.0, 100.0, 50.0)
    wind_speed = hybrid_input("Wind Speed (km/h)", 0.0, 50.0, 10.0)

    # ตรวจสอบว่าผู้ใช้ได้กรอกข้อมูลครบถ้วนหรือไม่
    if st.button("Predict Air Quality"):
        if any(value == 0.0 for value in [pm25, pm10, temp, humidity, wind_speed]):
            st.error("⚠️ กรุณากรอกค่าทุกช่องก่อนทำการพยากรณ์!")
        else:
            with st.spinner('Predicting...'):
                # เตรียมข้อมูลเข้าโมเดล
                input_data = air_quality_scaler.transform(
                    np.array([[pm25, pm10, temp, humidity, wind_speed]])
                )

                predictions = {}
                probabilities = {}

                # ทำนายด้วย SVM
                if "SVM (Current)" in selected_models:
                    pred_svm = svm_air_quality_model.predict(input_data)[0]
                    predictions["SVM"] = pred_svm
                    probabilities["SVM"] = list(svm_air_quality_model.predict_proba(input_data)[0])

                # ทำนายด้วย Random Forest
                if "Random Forest" in selected_models:
                    pred_rf = rf_air_quality_model.predict(input_data)[0]
                    predictions["Random Forest"] = pred_rf
                    probabilities["Random Forest"] = list(rf_air_quality_model.predict_proba(input_data)[0])

                quality_mapping = {0: "Good", 1: "Moderate", 2: "Poor"}

                # แสดงผลลัพธ์ของแต่ละโมเดลพร้อม Confidence Score
                for model_name, pred_class in predictions.items():
                    confidence = max(probabilities[model_name]) * 100  # เปลี่ยนเป็นเปอร์เซ็นต์
                    st.success(f"{model_name} Predicted Air Quality: {quality_mapping[pred_class]} (Confidence: {confidence:.2f}%)")

                # จัดรูปแบบข้อมูลสำหรับกราฟ
                categories = ["Good", "Moderate", "Poor"]
                prob_data = {"Air Quality": categories}
                models_selected = []

                for model_name in selected_models:
                    if model_name in probabilities:
                        prob_data[model_name] = probabilities[model_name]
                        models_selected.append(model_name)

                prob_df = pd.DataFrame(prob_data)

                # แปลงข้อมูลให้อยู่ในรูปแบบ Melted DataFrame
                prob_df_melted = prob_df.melt(
                    id_vars=["Air Quality"], 
                    var_name="Model", 
                    value_name="Probability"
                )

                prob_df_melted["Probability"] = prob_df_melted["Probability"].round(3)
                prob_df_melted["Air Quality"] = pd.Categorical(
                    prob_df_melted["Air Quality"], 
                    categories=["Good", "Moderate", "Poor"], 
                    ordered=True
                )

                # พล็อตกราฟ
                st.subheader("📊 Model Prediction Confidence")

                if len(models_selected) > 1:
                    fig = px.line(
                        prob_df_melted,
                        x="Air Quality",
                        y="Probability",
                        color="Model",
                        markers=True,
                        title="Comparison of Prediction Confidence"
                    )
                else:
                    fig = px.bar(
                        prob_df_melted,
                        x="Air Quality",
                        y="Probability",
                        color="Model",
                        barmode="group",
                        title="Prediction Confidence"
                    )

                fig.update_layout(
                    yaxis=dict(
                        range=[0, 1],
                        tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        tickformat=".3f"
                    ),
                    xaxis=dict(type='category')
                )

                st.plotly_chart(fig)

                # ✅ กราฟเส้นเปรียบเทียบค่าอินพุตกับ Good Standard
                factors = ['PM2.5', 'PM10', 'Temperature', 'Humidity', 'Wind Speed']
                values = [pm25, pm10, temp, humidity, wind_speed]
                standards = [12, 50, 25, 50, 10]  

                comparison_df = pd.DataFrame({
                    'Factors': factors, 
                    'Your Input': values,
                    'Good Standard': standards
                })

                fig_comparison = go.Figure()
                fig_comparison.add_trace(go.Scatter(
                    x=factors, y=values,
                    mode='lines+markers', name='Your Input',
                    line=dict(color='blue')
                ))
                fig_comparison.add_trace(go.Scatter(
                    x=factors, y=standards,
                    mode='lines+markers', name='Good Standard',
                    line=dict(color='red', dash='dash', width=2),
                    marker=dict(size=8)
                ))
                fig_comparison.update_layout(
                    title="Air Quality Factors Comparison",
                    xaxis_title="Factors",
                    yaxis_title="Value",
                    yaxis=dict(range=[0, max(values + standards) + 10])
                )
                st.plotly_chart(fig_comparison)

# ----------------------------------------------------------------------
# Page 5: Fruit Classification (NN)
if choice == "Fruit Classification (NN)":
    st.header("Fruit Classification (Neural Network)")

    # ข้อมูลตัวอย่างสำหรับผลไม้แต่ละประเภท
    fruit_examples = {
        "Apple": {"Weight": 180.0, "Length": 8.0, "Circumference": 28.0, "Color": 0},
        "Banana": {"Weight": 120.0, "Length": 18.0, "Circumference": 12.0, "Color": 1},
        "Orange": {"Weight": 220.0, "Length": 7.0, "Circumference": 23.0, "Color": 2}
    }

    selected_fruit_example = st.selectbox("Select Example Fruit", ["Custom", "Apple", "Banana", "Orange"])
    example_data_fruit = fruit_examples.get(
        selected_fruit_example, 
        {"Weight": 0.0, "Length": 0.0, "Circumference": 0.0, "Color": 0}
    )

    weight = hybrid_input("Weight (g)", 0.0, 500.0, float(example_data_fruit["Weight"]))
    length = hybrid_input("Length (cm)", 0.0, 30.0, float(example_data_fruit["Length"]))
    circumference = hybrid_input("Circumference (cm)", 0.0, 40.0, float(example_data_fruit["Circumference"]))
    color = st.selectbox(
        "Color", 
        [0, 1, 2], 
        format_func=lambda x: ["Green", "Yellow", "Orange"][x], 
        index=int(example_data_fruit["Color"])
    )

    if st.button("Classify Fruit"):
        with st.spinner('Classifying...'):
            input_data = fruit_scaler.transform(np.array([[weight, length, circumference, color]]))
            
            # ตรวจสอบว่า predict_proba() คืนค่าถูกต้อง
            probabilities = mlp_fruit_model.predict_proba(input_data)
            
            if probabilities is not None and probabilities.shape[1] == len(fruit_label_encoder.classes_):
                prediction = mlp_fruit_model.predict(input_data)[0]
                fruit_name = fruit_label_encoder.inverse_transform([prediction])[0]
                st.success(f"Predicted Fruit Type: {fruit_name}")

                # ✅ สร้าง DataFrame อย่างถูกต้อง
                prob_df_fruit = pd.DataFrame({
                    'Fruit Type': fruit_label_encoder.classes_,
                    'Probability': probabilities[0]
                })

                # ✅ ตรวจสอบ Debugging DataFrame
                st.subheader("🔍 Debugging: Probability Data")
                st.write(prob_df_fruit)

                # ✅ วาดกราฟแท่งให้ถูกต้อง
                fig_fruit = px.bar(
                    prob_df_fruit, 
                    x='Fruit Type', 
                    y='Probability', 
                    color='Fruit Type', 
                    title='Prediction Confidence'
                )

                fig_fruit.update_layout(
                    yaxis=dict(range=[0, 1]),
                    xaxis_title="Fruit Type",
                    yaxis_title="Probability"
                )

                st.plotly_chart(fig_fruit)
            else:
                st.error("Error: Probability output is incorrect. Check the model output format.")

# ----------------------------------------------------------------------
# Page 6: Batch Prediction
if choice == "Batch Prediction":
    st.header("Batch Prediction with CSV Upload")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(df)

        # ตรวจสอบว่ามีคอลัมน์สำหรับ Air Quality ครบหรือไม่
        if (
            'PM2.5' in df.columns and 'PM10' in df.columns and 
            'Temperature' in df.columns and 'Humidity' in df.columns and 
            'Wind_Speed' in df.columns
        ):
            st.write("Performing Air Quality Prediction...")
            input_data = air_quality_scaler.transform(
                df[['PM2.5', 'PM10', 'Temperature', 'Humidity', 'Wind_Speed']]
            )
            predictions = svm_air_quality_model.predict(input_data)
            df['Predicted_Air_Quality'] = [
                "Good" if pred == 0 else "Moderate" if pred == 1 else "Poor" 
                for pred in predictions
            ]
            st.write("Predicted Results:")
            st.table(df)

        # ตรวจสอบว่ามีคอลัมน์สำหรับผลไม้ครบหรือไม่
        elif (
            'Weight' in df.columns and 'Length' in df.columns and 
            'Circumference' in df.columns and 'Color' in df.columns
        ):
            st.write("Performing Fruit Classification...")
            input_data = fruit_scaler.transform(
                df[['Weight', 'Length', 'Circumference', 'Color']]
            )
            predictions = mlp_fruit_model.predict(input_data)
            df['Predicted_Fruit_Type'] = fruit_label_encoder.inverse_transform(predictions)
            st.write("Predicted Results:")
            st.table(df)

        else:
            st.error("The uploaded file does not have the required columns for prediction.")
