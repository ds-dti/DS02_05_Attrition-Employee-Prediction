import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)

#Dictionary for getting value from form webpage
list_kolom = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
Department_dict = {'Sales': 0, 'Research & Development': 1, 'Human Resources':2}
EducationField_dict = {'Life Sciences': 0, 'Other': 1, 'Medical':2,'Marketing':3,'Technical Degree':4,'Human Resources':5}
BusinessTravel_dict = {'Travel_Rarely': 0,'Travel_Frequently': 1,'Non-Travel':2}
Gender_dict = {'Female': 0,'Male': 1}
JobRole_dict = {'Sales Executive': 0,'Research Scientist': 1, 'Laboratory Technician':2,'Manufacturing Director': 3,'Healthcare Representative': 4, 
                                    'Manager':5,'Sales Representative': 6,'Research Director': 7, 'Human Resources':8}
MaritalStatus_dict = {'Single': 0,'Married': 1, 'Divorced':2}
OverTime_dict = {'Yes': 0,'No': 1}
Condition_dict = {'Low':0, 'Medium':1, 'High':2, 'Very High':3}
Education_dict = {'Bellow College':0, 'College':1, 'Bachelor':2, 'Master':3, 'Doctor':4}
WorkLifeBalance_dict = {'Bad':0, 'Good':1, 'Better':2, 'Best':3}

#Loading Model Random Forest
classifier = joblib.load("random_forest.pkl")

#CSV for visualization (result from encoding)
visualisasi = pd.read_csv("visualisasi.csv")
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

#CSV for Feature Importance visualization (result from random forest (RandomForest.feature.importance_))
graph_df = pd.read_csv("draft.csv")

#Summary from encoded dataframe before train and test splitting
info = pd.read_csv("info.csv")
#DataFrame processing
df['Attrition'] = df['Attrition'].map( {'Yes': 0,'No': 1} ).astype(int)
df.drop(['EmployeeCount', 'EmployeeNumber','StandardHours'], axis=1, inplace=True)

#Streamlit 
st.sidebar.header('Attrition Prediction')

if not st.sidebar.checkbox("Data Visualization", True, key='2'):
    st.title('Attrition Employee Predictions')
    
    age = st.number_input("Age:")
    BusinessTravel = st.selectbox('Business Travel',('Travel_Rarely', 'Travel_Frequently', 'Non-Travel'))
    DailyRate = st.number_input("Daily Rate:")
    Department = st.selectbox('Department',('Sales', 'Research & Development', 'Human Resources'))
    DistanceFromHome = st.number_input("Distance From Home:")
    Education = st.selectbox('Education',('Bellow College', 'College', 'Bachelor', 'Master', 'Doctor'))
    EducationField = st.selectbox('Education Field',('Life Sciences', 'Other', 'Medical','Marketing','Technical Degree','Human Resources'))
    EnvironmentSatisfaction = st.selectbox('Environment Satisfaction',('Low', 'Medium', 'High', 'Very High'))
    Gender = st.selectbox('Gender',('Male', 'Female'))
    HourlyRate = st.number_input("Hourly Rate:")
    JobInvolvement = st.selectbox('Job Involvement',('Low', 'Medium', 'High', 'Very High'))
    JobLevel = st.number_input("Job Level:")
    JobRole = st.selectbox('Job Role',('Sales Executive',
                                    'Research Scientist', 
                                    'Laboratory Technician',
                                    'Manufacturing Director',
                                    'Healthcare Representative', 
                                    'Manager',
                                    'Sales Representative',
                                    'Research Director', 
                                    'Human Resources'))
    JobSatisfaction = st.selectbox('Job Satisfaction',('Low', 'Medium', 'High', 'Very High'))
    MaritalStatus = st.selectbox('Marital Status',('Single', 'Married', 'Divorced')) 
    MonthlyIncome = st.number_input("Monthly Income:")
    MonthlyRate = st.number_input("Monthly Rate:")
    NumCompaniesWorked = st.number_input("Number Companies Worked:")
    OverTime = st.selectbox('OverTime',('Yes', 'No'))
    PercentSalaryHike = st.number_input("Percent Salary Hike:")
    PerformanceRating = st.selectbox('Performance Rating',('Low', 'Medium', 'High', 'Very High'))
    RelationshipSatisfaction = st.selectbox('Relationship Satisfaction',('Low', 'Medium', 'High', 'Very High'))
    StockOptionLevel = st.number_input("Stock Option Level:")
    TotalWorkingYears = st.number_input("Total Working Years:")
    TrainingTimesLastYear = st.number_input("Training Times Last Year:")
    WorkLifeBalance = st.selectbox('Work Life Balance',('Bad', 'Good', 'Better', 'Best'))
    YearsAtCompany = st.number_input("Years At Company:")
    YearsInCurrentRole = st.number_input("Years In Current Role:")
    YearsSinceLastPromotion = st.number_input("Years Since Last Promotion:")
    YearsWithCurrManager = st.number_input("Years With Current Manager:")


    
    dict_df = dict()

    submit = st.button('Predict')
    if submit:
        #getting value from dictionary
        bt_lr = BusinessTravel_dict[BusinessTravel]
        Dept_lr = Department_dict[Department]
        Edu_lr = Education_dict[Education]
        Edu_field_lr = EducationField_dict[EducationField]
        env_sat_lr = Condition_dict[EnvironmentSatisfaction]
        Gender_lr = Gender_dict[Gender]
        job_inv_lr = Condition_dict[JobInvolvement]
        jobrole_lr = JobRole_dict[JobRole]
        js_lr = Condition_dict[JobSatisfaction]
        marit_lr = MaritalStatus_dict[MaritalStatus]
        ot_lr = OverTime_dict[OverTime]
        pr_lr = Condition_dict[PerformanceRating]
        rs_lr = Condition_dict[RelationshipSatisfaction]
        wb_lr = WorkLifeBalance_dict[WorkLifeBalance]
        list_prediksi = [age,bt_lr,DailyRate,Dept_lr,DistanceFromHome,Edu_lr,Edu_field_lr,env_sat_lr,Gender_lr,HourlyRate,job_inv_lr,JobLevel,jobrole_lr,js_lr,marit_lr,MonthlyIncome,MonthlyRate,NumCompaniesWorked,ot_lr,PercentSalaryHike,pr_lr,rs_lr,StockOptionLevel,TotalWorkingYears,TrainingTimesLastYear,wb_lr,
                    YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,YearsWithCurrManager]
        #encoding
        for i in range(len(list_prediksi)):
            dict_df.update({list_kolom[i]:[list_prediksi[i]]})
        df_pred = pd.DataFrame(dict_df)
        df_pred.to_csv("df_pred.csv")
        st.write("Test Parameter")
        st.write(df_pred)
        #prediction
        prediction = classifier.predict(df_pred)
        prob = classifier.predict_proba(df_pred)
        
        if prediction == 0:
            st.markdown("<h3 style='text-align: center;'>Prediction Result = Attrition Yes</h3>", unsafe_allow_html=True)
            if np.amax(prob) >= 0.8:
                st.markdown("<h3 style='text-align: center;'>High Risk</h3>", unsafe_allow_html=True)
            elif np.amax(prob) <= 0.8 and np.amax(prob) > 0.6:
                st.markdown("<h3 style='text-align: center;'>Medium Risk</h3>", unsafe_allow_html=True)
            elif np.amax(prob) <= 0.6:
                st.markdown("<h3 style='text-align: center;'>Low Risk</h3>", unsafe_allow_html=True)
            #Making plot for most important features in Random Forest
            importances = classifier.feature_importances_
            indices = np.argsort(importances)[::-1] # Sort feature importances in descending order
            names = [graph_df.columns[i] for i in indices] # Rearrange feature names so they match the sorted feature importances
            plt.figure(figsize=(15, 7)) # Create plot
            plt.title("Feature Importance") # Create plot title
            plt.bar(range(graph_df.shape[1]), importances[indices]) # Add bars
            plt.xticks(range(graph_df.shape[1]), names, rotation=90) # Add feature names as x-axis labels
            st.markdown("<h3 style='text-align: center;'>Importance Feature Plot</h3>", unsafe_allow_html=True)
            st.pyplot()
            #Parameter explanation
            st.write("Parameter dengan koefisien terbesar:")
            for i in range(10):
                st.write("Parameter ke-{}: {}".format(i+1,names[i]))
            #Reccomendation
            st.markdown("<h3 style='text-align: center;'>Rekomendasi</h3>", unsafe_allow_html=True)
            for i in df_pred.columns:
                if i == "Overtime ":
                    if df_pred[i][0] == 0:
                        st.write("Planning project secara tepat dengan dukungan tenaga kerja yang memadai, mengurangi overtime")
                if i == "MonthlyIncome":
                    if df_pred[i][0] < info["{}_mean".format(i)][0]:
                        st.write("Memberikan gaji sesuai dengan effort yang dikeluarkan")
                elif i == "TotalWorkingYears":
                    if df_pred[i][0] < info["{}_mean".format(i)][0]:
                        st.write("Memberikan reward yang menarik")
                elif i == "Age":
                    if df_pred[i][0] < info["{}_mean".format(i)][0]:
                        st.write("Umur pegawai masih dibawah rata rata umur pegawai lain, mungkin dapat menawarkan kompensasi dan benefit")
                    elif df_pred[i][0] > info["{}_mean".format(i)][0]:
                        st.write("Umur pegawai sudah diatas rata rata umur pegawai lain, mungkin dapat menawarkan kompensasi dan benefit")
                elif i == "TotalWorkingYears":
                    if df_pred[i][0] < info["{}_mean".format(i)][0]:
                        st.write("Memberikan reward yang menarik")
                elif i == "MaritalStatus":
                    if df_pred[i][0] == 1:
                        st.write("Memberikan komisi atau bonus bagi yang berkeluarga")
                elif i == "YearsAtCompany":
                    if df_pred[i][0] < info["{}_mean".format(i)][0]:
                        st.write("Kesempatan pengembangan karir")
                elif i == "JobLevel":
                    if df_pred[i][0] == 1 or df_pred[i][0] == 2:
                        st.write("Memberikan reward yang menarik")
                elif i == "EnvironmentSatisfaction":
                    if df_pred[i][0] == 1 or df_pred[i][0] == 2:
                        st.write("Memberikan reward yang menarik")            
        else:
            st.markdown("<h2 style='text-align: center;'>Prediction Result = Attrition No</h2>", unsafe_allow_html=True)

else:
    #Data Visualisasi
    st.markdown("<h1 style='text-align: center;'>Data Visualization</h1>", unsafe_allow_html=True)
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    
    st.markdown("<h3 style='text-align: center;'>Heatmap Data Plot </h3>", unsafe_allow_html=True)
    # Heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr,
            vmax=.5,
            mask=mask,
            annot=True, fmt='.2f',
            linewidths=.2, cmap="YlGnBu")
 
    st.pyplot()
    #Bar plot
    st.markdown("<h3 style='text-align: center;'>Histogram Plot</h3>", unsafe_allow_html=True)
    bar_plot = df.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20))
    st.pyplot()
    
    st.markdown("<h3 style='text-align: center;'>KDE Plot</h3>", unsafe_allow_html=True)
    f, axes = plt.subplots(3, 3, figsize=(10, 8), 
                       sharex=False, sharey=False)

    # Defining our colormap scheme
    s = np.linspace(0, 3, 10)
    cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)

    # Generate and plot
    x = visualisasi['Age'].values
    y = visualisasi['TotalWorkingYears'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=axes[0,0])
    axes[0,0].set( title = 'Age against Total working years')

    cmap = sns.cubehelix_palette(start=0.333333333333, light=1, as_cmap=True)
    # Generate and plot
    x = visualisasi['Age'].values
    y = visualisasi['DailyRate'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,1])
    axes[0,1].set( title = 'Age against Daily Rate')

    cmap = sns.cubehelix_palette(start=0.666666666667, light=1, as_cmap=True)
    # Generate and plot
    x = visualisasi['YearsInCurrentRole'].values
    y = visualisasi['Age'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,2])
    axes[0,2].set( title = 'Years in role against Age')

    cmap = sns.cubehelix_palette(start=1.0, light=1, as_cmap=True)
    # Generate and plot
    x = visualisasi['DailyRate'].values
    y = visualisasi['DistanceFromHome'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,0])
    axes[1,0].set( title = 'Daily Rate against DistancefromHome')

    cmap = sns.cubehelix_palette(start=1.333333333333, light=1, as_cmap=True)
    # Generate and plot
    x = visualisasi['DailyRate'].values
    y = visualisasi['JobSatisfaction'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,1])
    axes[1,1].set( title = 'Daily Rate against Job satisfaction')

    cmap = sns.cubehelix_palette(start=1.666666666667, light=1, as_cmap=True)
    # Generate and plot
    x = visualisasi['YearsAtCompany'].values
    y = visualisasi['JobSatisfaction'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,2])
    axes[1,2].set( title = 'Daily Rate against distance')

    cmap = sns.cubehelix_palette(start=2.0, light=1, as_cmap=True)
    # Generate and plot
    x = visualisasi['YearsAtCompany'].values
    y = visualisasi['DailyRate'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,0])
    axes[2,0].set( title = 'Years at company v Daily Rate')

    cmap = sns.cubehelix_palette(start=2.333333333333, light=1, as_cmap=True)
    # Generate and plot
    x = visualisasi['RelationshipSatisfaction'].values
    y = visualisasi['YearsWithCurrManager'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,1])
    axes[2,1].set( title = 'Relation_Satis vs years_manager')

    cmap = sns.cubehelix_palette(start=2.666666666667, light=1, as_cmap=True)
    # Generate and plot
    x = visualisasi['WorkLifeBalance'].values
    y = visualisasi['JobSatisfaction'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,2])
    axes[2,2].set( title = 'WorklifeBalance v Satisfaction')

    f.tight_layout()
    st.pyplot()
