#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)



# In[60]:


df = pd.read_csv("irrigation_prediction.csv")



# In[61]:





# In[62]:


# In[63]:




# In[64]:




# In[65]:




# In[66]:




# In[67]:


numerical_column=df.select_dtypes(include=["float64"]).columns
categorical_column = df.select_dtypes(include=["object"]).columns 



# In[68]:


plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Irrigation_Need",
    palette="viridis"
)

plt.title("Distribution of Irrigation Need")

plt.show()


# In[69]:


df[numerical_column].hist(
    figsize=(18,12),
    bins=25
)
plt.tight_layout()
plt.show()


# In[70]:


plt.figure(figsize=(20,12))

for i,col in enumerate(numerical_column):

    plt.subplot(3,4,i+1)

    sns.boxplot(y=df[col])

    plt.title(col)
plt.tight_layout()
plt.show()


# In[71]:


plt.figure(figsize=(12,8))
sns.heatmap(
    df[numerical_column].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.show()


# In[72]:


sns.pairplot(
    df[
        [
            "Temperature_C",
            "Humidity",
            "Rainfall_mm",
            "Soil_Moisture",
            "Irrigation_Need"
        ]
    ],
    hue="Irrigation_Need"
)
plt.show()


# In[73]:





# In[74]:


plt.figure(figsize=(10,5))
sns.countplot(
    data=df,
    x="Crop_Type"
)
plt.xticks(rotation=45)
plt.show()


# In[75]:


plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    x="Soil_Type"
)
plt.show()


# In[76]:


plt.figure(figsize=(10,5))
sns.countplot(
    data=df,
    x="Region"
)
plt.xticks(rotation=45)
plt.show()


# In[77]:


plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Irrigation_Need",
    y="Previous_Irrigation_mm"
)
plt.show()


# In[78]:


plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Irrigation_Need",
    y="Rainfall_mm"
)
plt.show()


# In[79]:


plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Irrigation_Need",
    y="Soil_Moisture"
)
plt.show()


# In[80]:


df.to_csv("clean_irrigation.csv",index=False)


# In[81]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib



# In[82]:


df = pd.read_csv("clean_irrigation.csv")


# In[83]:



# In[84]:


X = df.drop("Irrigation_Need", axis=1)
y = df["Irrigation_Need"]





# In[85]:


target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)


# In[86]:


joblib.dump(target_encoder,"target_encoder.pkl")


# In[87]:


categorical_column = X.select_dtypes(include="object").columns



# In[88]:


feature_encoders = {}
for col in categorical_column:
   encoder = LabelEncoder()
   X[col] = encoder.fit_transform(X[col])
   feature_encoders[col] = encoder


# In[89]:


joblib.dump(feature_encoders,"feature_encoders.pkl")


# In[90]:




# In[91]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# In[92]:


stratify=y


# In[93]:



# In[94]:



# In[95]:




# In[96]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[97]:


joblib.dump(scaler,"scaler.pkl")


# In[98]:




# In[99]:


joblib.dump(X_train,"X_train.pkl")
joblib.dump(X_test,"X_test.pkl")
joblib.dump(y_train,"y_train.pkl")
joblib.dump(y_test,"y_test.pkl")


# In[100]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings("ignore")


# In[101]:


X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")
scaler = joblib.load("scaler.pkl")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[102]:


models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            random_state=42,
            eval_metric="mlogloss"
        )

}


# In[103]:


results = []
trained_models = {}


# In[104]:


for name, model in models.items():
    
    if name == "Logistic Regression":
       model.fit(X_train_scaled, y_train)
       predictions = model.predict(X_test_scaled)
    else:

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

    trained_models[name] = model

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])


  


# In[105]:


results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)
results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)



# In[106]:


plt.figure(figsize=(10,5))
sns.barplot(
    data=results_df,
    x="Model",
    y="F1 Score"
)
plt.xticks(rotation=20)
plt.title("Model Comparison")
plt.show()


# In[107]:


best_model_name = results_df.iloc[0]["Model"]


# In[108]:


best_model = trained_models[best_model_name]
joblib.dump(best_model,"irrigation_model.pkl")



# In[115]:


from sklearn.metrics import ConfusionMatrixDisplay
best_model = trained_models["XGBoost"]
predictions = best_model.predict(X_test)
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    cmap="Blues"
)
plt.title("Confusion Matrix")
plt.show()


# In[116]:


feature_importance = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance":
        best_model.feature_importances_

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)



# In[111]:


plt.figure(figsize=(10,6))

sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")

plt.show()


# In[117]:


results_df.to_csv("model_results.csv",index=False)


# In[118]:




# In[119]:





# In[122]:


import shap 
best_model = joblib.load("irrigation_model.pkl")
X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")


# In[123]:


explainer = shap.TreeExplainer(best_model)


# In[124]:


shap_values = explainer.shap_values(X_test)


# In[125]:








# In[128]:


shap.summary_plot(
    shap_values[:, :, 0],
    X_test
)


# In[130]:


shap.summary_plot(
    shap_values[:,:,0],
    X_test,
    plot_type="bar"
)


# In[140]:


import random
index = random.randint(0, len(X_test)-1)
sample = X_test.iloc[[index]]
predicted_class = best_model.predict(sample)[0]
explanation = explainer(sample)
shap.plots.waterfall(explanation[0,:,predicted_class])


# In[146]:





# In[147]:





# In[151]:


import requests
def get_coordinates(place):

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": place,
        "count": 10,
        "language": "en",
        "format": "json",
        "countryCode": "IN"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "results" not in data:
        return None

    # Prefer exact name match
    for location in data["results"]:
        if location["name"].lower() == place.lower():
            return (
                location["latitude"],
                location["longitude"],
                location["name"],
                location.get("admin1", "")
            )

    # Otherwise return first Indian result
    location = data["results"][0]

    return (
        location["latitude"],
        location["longitude"],
        location["name"],
        location.get("admin1", "")
    )

def get_weather(latitude, longitude):

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    params = {
    "latitude": latitude,
    "longitude": longitude,
    "current": "temperature_2m,relative_humidity_2m,rain,wind_speed_10m",
    "timezone": "auto"
}

    response = requests.get(url, params=params)

    data = response.json()

    if "current" not in data:
        return None

    return {

        "Temperature_C":
            data["current"]["temperature_2m"],

        "Humidity":
            data["current"]["relative_humidity_2m"],

        "Wind_Speed_kmh":
            data["current"]["wind_speed_10m"],

        "Rainfall_mm":
            data["current"]["rain"]

    }

# In[154]:



model = joblib.load("irrigation_model.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")


# In[167]:


import streamlit as st
st.set_page_config(
    page_title="Smart Irrigation Prediction",
    layout="wide"
)
st.title("🌾 Smart Irrigation Prediction System")
st.write("Developed by **ABHISHEK**")


# In[13]:


place = st.text_input(
    "📍 Enter Village / Town / District / City / State",
    placeholder="Example: Leh, Ladakh or Rourkela, Odisha"
)

soil_ph = st.number_input("Soil pH", 5.5, 9.0, 6.5, key="soil_ph")

soil_moisture = st.number_input("Soil Moisture", 0.0, 100.0, 25.0, key="soil_moisture")

organic_carbon = st.number_input("Organic Carbon", 0.0, 10.0, 0.5, key="organic_carbon")

ec = st.number_input("Electrical Conductivity", 0.0, 10.0, 2.0, key="electrical_conductivity")

sunlight = st.number_input("Sunlight Hours", 0.0, 15.0, 8.0, key="sunlight_hours")

field_area = st.number_input("Field Area (ha)", 0.1, 100.0, 5.0, key="field_area")

previous_irrigation = st.number_input(
    "Previous Irrigation (mm)",
    0.0,
    100.0,
    10.0,
    key="previous_irrigation"
)

soil = st.selectbox(
    "Soil Type",
    feature_encoders["Soil_Type"].classes_
)

crop = st.selectbox(
    "Crop Type",
    feature_encoders["Crop_Type"].classes_
)

growth = st.selectbox(
    "Crop Growth Stage",
    feature_encoders["Crop_Growth_Stage"].classes_
)

season = st.selectbox(
    "Season",
    feature_encoders["Season"].classes_
)

irrigation = st.selectbox(
    "Irrigation Type",
    feature_encoders["Irrigation_Type"].classes_
)

water = st.selectbox(
    "Water Source",
    feature_encoders["Water_Source"].classes_
)

mulching = st.selectbox(
    "Mulching Used",
    feature_encoders["Mulching_Used"].classes_
)

region = st.selectbox(
    "Region",
    feature_encoders["Region"].classes_
)

if st.button("Predict"):

    coordinates = get_coordinates(place)

    if coordinates is None:
        st.error("Location not found.")
        st.stop()

    latitude, longitude, location_name, state_name = coordinates

    st.success(f"📍 Weather Location: {location_name}, {state_name}")

    weather = get_weather(latitude, longitude)

    if weather is None:
        st.error("Unable to fetch weather.")
        st.stop()

    st.subheader("Live Weather")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Temperature", f"{weather['Temperature_C']} °C")
    col2.metric("Humidity", f"{weather['Humidity']} %")
    col3.metric("Wind Speed", f"{weather['Wind_Speed_kmh']:.2f} km/h")
    col4.metric("Rainfall", f"{weather['Rainfall_mm']} mm")

    input_df = pd.DataFrame({

        "Soil_Type":[feature_encoders["Soil_Type"].transform([soil])[0]],

        "Soil_pH":[soil_ph],

        "Soil_Moisture":[soil_moisture],

        "Organic_Carbon":[organic_carbon],

        "Electrical_Conductivity":[ec],

        "Temperature_C":[weather["Temperature_C"]],

        "Humidity":[weather["Humidity"]],

        "Rainfall_mm":[weather["Rainfall_mm"]],

        "Sunlight_Hours":[sunlight],

        "Wind_Speed_kmh":[weather["Wind_Speed_kmh"]],

        "Crop_Type":[feature_encoders["Crop_Type"].transform([crop])[0]],

        "Crop_Growth_Stage":[feature_encoders["Crop_Growth_Stage"].transform([growth])[0]],

        "Season":[feature_encoders["Season"].transform([season])[0]],

        "Irrigation_Type":[feature_encoders["Irrigation_Type"].transform([irrigation])[0]],

        "Water_Source":[feature_encoders["Water_Source"].transform([water])[0]],

        "Field_Area_hectare":[field_area],

        "Mulching_Used":[feature_encoders["Mulching_Used"].transform([mulching])[0]],

        "Previous_Irrigation_mm":[previous_irrigation],

        "Region":[feature_encoders["Region"].transform([region])[0]]

    })

    prediction = model.predict(input_df)

    label = target_encoder.inverse_transform(prediction)[0]

    st.success("Prediction Completed")

    st.metric("Predicted Irrigation Need", label)





