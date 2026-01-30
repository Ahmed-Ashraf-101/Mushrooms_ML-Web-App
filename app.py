import streamlit as st
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

def main():
    st.title("Binary Classification Web Application")
    st.sidebar.title('Binary Classifiers')
    st.markdown('Are your mushrooms edible or poisonous?')
    st.sidebar.markdown('Mushrooms Mushrooms...')

    @st.cache_data(persist= True)
    def load_data():
        data = pd.read_csv(r"D:\Career\Coursera Projects\Build ML Web App with Streamlit\mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist= True)
    def split(df):
        y = df.type
        x = df.drop(columns= ['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= .25, stratify= y, shuffle= True, random_state= 0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix:')
            fig1 = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels= class_names, cmap= 'viridis')
            st.pyplot(fig1.figure_)
        
        if 'ROC Curve' in metrics_list:
            st.subheader('Receiver Operating Characteristic Curve:')
            fig2 = RocCurveDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(fig2.figure_)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            fig3 = PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(fig3.figure_)

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['Edible', 'Poisonous']
    st.sidebar.subheader('Choose Classifer:')
    classifier = st.sidebar.selectbox('Classification Model:', ('Support Vector Machine', 'Random Forest', 'Logistic Regression'))

    if classifier == 'Support Vector Machine':
        st.sidebar.subheader('Model Hyper-parameters')
        C = st.sidebar.number_input('C (Regularization Parameter):', 0.01, 10.0, step= 0.01, key= 'C', )
        kernel = st.sidebar.radio('Kernel Type:', ('rbf', 'poly', 'linear', 'sigmoid', 'precomputed'), key= 'kernel')
        gamma = st.sidebar.radio('Gamma (Kernel Coefficient):', ('scale', 'auto'), key= 'gamma')

        metrics = st.sidebar.multiselect('Valuation Metrics', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key= 'Classify'):
            st.subheader('(SVM) Results')
            model = SVC(C= C, kernel= kernel, gamma= gamma)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            st.write(f'Accuracy = {model.score(x_test, y_test):.2f}')
            st.write(f'Precision = {precision_score(y_test, y_pred, labels= class_names):.2f}')
            st.write(f'Recall = {recall_score(y_test, y_pred, labels= class_names):.2f}')
            st.write(f'F1_Score = {f1_score(y_test, y_pred, labels= class_names):.2f}')
            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader('Model Hyper-parameters')
        C_LR = st.sidebar.number_input('C (Regularization Parameter):', 0.01, 10.0, step= 0.01, key= 'C_LR')
        max_iter = st.sidebar.slider('Maximum Iterations:', 50, 500, key= 'max_iter')

        metrics = st.sidebar.multiselect('Valuation Metrics', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key= 'Classify'):
            st.subheader('Logistic Regression Results')
            model = LogisticRegression(C= C_LR, max_iter= max_iter, )
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            st.write(f'Accuracy = {model.score(x_test, y_test):.2f}')
            st.write(f'Precision = {precision_score(y_test, y_pred, labels= class_names):.2f}')
            st.write(f'Recall = {recall_score(y_test, y_pred, labels= class_names):.2f}')
            st.write(f'F1_Score = {f1_score(y_test, y_pred, labels= class_names):.2f}')
            plot_metrics(metrics)

    if classifier == 'Random Forest':
            st.sidebar.subheader('Model Hyper-parameters')
            n_estimators = st.sidebar.slider('# of Decision Trees:', 50, 500, key= 'n_estimators')
            max_depth = st.sidebar.number_input('Maximum Depth of each Decision Tree:', 5, 50, step= 5, key= 'max_depth')
            min_samples_split = st.sidebar.number_input('Minimum Samples to Split:', 2, 20, step= 2, key= 'min_samples_split')
            criterion = st.sidebar.selectbox('Criterion:', ('gini', 'entropy', 'log_loss'))
            bootstrap = st.sidebar.radio('Bootstrapping Samples:', [True, False], key= 'bootstrap', format_func= bool)

            metrics = st.sidebar.multiselect('Valuation Metrics', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button('Classify', key= 'Classify'):
                st.subheader('Random Forest Results')
                model = RandomForestClassifier(n_estimators= n_estimators, criterion= criterion, min_samples_split= min_samples_split, max_depth= max_depth, bootstrap= bootstrap, n_jobs= -1)
                model.fit(x_train, y_train)

                y_pred = model.predict(x_test)
                st.write(f'Accuracy = {model.score(x_test, y_test):.2f}')
                st.write(f'Precision = {precision_score(y_test, y_pred, labels= class_names):.2f}')
                st.write(f'Recall = {recall_score(y_test, y_pred, labels= class_names):.2f}')
                st.write(f'F1_Score = {f1_score(y_test, y_pred, labels= class_names):.2f}')
                plot_metrics(metrics)

    if st.sidebar.checkbox('Show raw data', False):
        st.subheader('Mushroom Dataset (Classification)')
        st.write(df)

if __name__ == '__main__':
    main()
