import streamlit as st
from operator import index
import plotly.express as px
from pycaret.classification import setup, compare_models, pull, save_model, load_model
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import ydata_profiling
import pickle
from PIL import Image
import pandas as pd
from streamlit import columns
from streamlit_pandas_profiling import st_profile_report
import os
from streamlit_option_menu import option_menu
from toolz.curried import unique

st.set_page_config(
    page_title="AutoBot",
    page_icon="🤖",
    layout="wide",

)

# Nav Bar
selected = option_menu(
    menu_title="AutoBot",
    options=["Dashboard", "Classification",
             "Regression", "Sample Applications"],
    icons=["boxes", "layout-wtf", "graph-up-arrow", "grid-fill"],
    menu_icon="stack",
    orientation="horizontal")

if selected == "Dashboard":
    body = """
    <h1> <marquee behavior="alternate">Hi! there👋 I am Autobot</marquee></h1>
    <p>Embark on your ML journey with me, and don't you worry you dont need to have any superpowers.
    I will be there helping you out just bring the data to me I am your Sherlock😉</p>
    <h3>Who am I, you ask❓</h3>
    <p>I am Autobot, an innovative AutoML web application designed to make machine learning accessible to everyone. With
     me, you can train machine learning models without writing a single line of complex code. I am user-friendly and 
     intuitive, making it easy for both beginners and experts to create, train, and deploy models. Whether you’re 
     looking to predict sales, find out more about a dataset, or anything in between,I’m here to help you achieve your goals with
      the power of machine learning. Let’s start this exciting journey together!</p>
    <h3>How can I make your life easier❓</h3>
    <p>I simplify your life by enabling on-the-go dataset analysis and model training without complex coding. I save 
    you time and effort by handling the technical aspects of machine learning, so you can focus on interpreting results 
    and making data-driven decisions. I make machine learning accessible to everyone, regardless of their background in
     data science or programming. By democratizing machine learning, I aim to make it beneficial for all. Let’s start 
     this exciting journey together!</p>
    """
    la1, la2 = st.columns(2)
    with la1:
        st.image("assets/autobot.gif")
    with la2:
        st.markdown(body, unsafe_allow_html=True)
    st.markdown("<hr style='border:1px dashed #FF4B4B'>",
                unsafe_allow_html=True)
    aa1, aa2 = st.columns(2)
    with aa1:
        body = """
            <h1>Here is how you can gain my superpowers✨</h1>
            <p>Together we can train a machine learning model in a few simple steps:</p>
            <h3>Upload Your Dataset📑:</h3>
            <p>Begin by uploading your dataset using the "Upload" section. Click on the "Upload Your Dataset" button and
             select your dataset <b>CSV<b> file.</p>
            <h3>I will generate a detailed EDA📋:</h3>
            <p>Once your dataset is uploaded, explore its characteristics through the "Profiling" section.
            Autobot performs EDA on the dataset so that you know more about it.</p>
            <h3>Train Your Model🧮:</h3>
            <p>Move on to the "Modelling" section and choose the target column then from here Autobot does all the work 
            it trains, compares different models, and selects the best-performing one.</p>
            <h3>Download Your Model💾:</h3>
            <p>After all this, a best performing trained model is ready, download it and use wherever you want.</p> 
            """
        st.markdown(body, unsafe_allow_html=True)
    with aa2:
        ins = Image.open("assets/manual.png")
        st.image(ins.resize((700, 700)))

# Classification Trainer Code

if selected == "Classification":

    if os.path.exists('./dataset.csv'):
        df = pd.read_csv('dataset.csv', index_col=None)

    with st.sidebar:
        st.image("assets/Core.gif")
        st.title("AutoBot : Classification Trainer")
        choice = st.radio(
            "Workflow 👇", ["Upload", "Profiling", "Modelling", "Download"])
        st.info("This is an AutoML app for Classification problems just upload a dataset and go through the selection "
                "steps only this time let our app do all the hardworking.")

    if choice == "Upload":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)

    if choice == "Profiling":
        st.title("Exploratory Data Analysis")
        profile_df = df.profile_report()
        st_profile_report(profile_df)

    if choice == "Modelling":
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            setup(df, target=chosen_target)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')

    if choice == "Download":
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")

# Regression Trainer Code

if selected == "Regression":

    if os.path.exists('./dataset.csv'):
        df = pd.read_csv('dataset.csv', index_col=None)

    with st.sidebar:
        st.image("assets/Core.gif", use_column_width="always")
        st.title("AutoBot : Regression Trainer")
        choice = st.radio(
            "Workflow 👇", ["Upload", "Profiling", "Modelling", "Download"])
        st.info("This is an AutoML app for Regression problems just upload a dataset and go"
                " through the selection steps only this time let our app do all the hardworking.")

    if choice == "Upload":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)

    if choice == "Profiling":
        st.title("Exploratory Data Analysis")
        profile_df = df.profile_report()
        st_profile_report(profile_df)

    if choice == "Modelling":
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            setup(df, target=chosen_target)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')

    if choice == "Download":
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")

if selected == "Sample Applications":
    options = option_menu(
        menu_title=None,
        options=["About", "Employee Churn Analysis","Data Analytics", "Announcements"],
        icons=["body-text", "briefcase","bar-chart-line", "megaphone"], orientation="horizontal")

    # About
    if options == "About":
        x, la1, la2 = st.columns(3)
        with la1:
            element = """
            <h1 style="padding-top:65px">Sample Model Significance</h1>
            <p>I will also provides some sample models just to showcase how machine learning can really enable 
            you to see into the future by making 
            predictions on data by learning from it. These are just just basic examples, you can do a lot more than that.</p>"""
            st.markdown(element, unsafe_allow_html=True)
        with la2:
            ins = Image.open("assets/manual.png")
            st.image(ins.resize((400, 400)))
        st.markdown("<hr style='border:2px dotted #FF4B4B'>",
                    unsafe_allow_html=True)

    # Data Analytics
    if options == "Data Analytics":
        st.markdown("""<h1 style="padding-top:65px">Data analysis and visualization.""", unsafe_allow_html=True) 
        st.markdown("""<p>This app is designed to help data scientists, machine learning engineers, and business analysts to analyze and visualize data.
        It's a comprehensive tool that allows you to explore, understand, and gain insights from your data.
        </p>""", unsafe_allow_html=True)
        st.markdown("<hr style='border:2px dotted #FF4B4B'>", unsafe_allow_html=True)

        upload_file = st.file_uploader("Choose a CSV file", type='csv')

        if upload_file is not None:
            df = pd.read_csv(upload_file)
            st.subheader("Data Preview")
            st.write(df.head())

            st.subheader("Data Summary")
            st.write(df.describe())

            st.subheader('Filter Data')
            columns = df.columns.tolist()
            selected_column = st.selectbox("Select Columns to filter by", columns)
            unique_values = df[selected_column].unique()
            selected_value = st.selectbox("Select value", unique_values)

            filtered_df = df[df[selected_column] == selected_value]
            st.write(filtered_df)

            st.subheader("Plot Data")
            x_column = st.selectbox("Select x-axis column", columns)
            y_column = st.selectbox("Select y-axis column", columns)

            if st.button("Generate Plot"):
                st.line_chart(filtered_df.set_index(x_column)[y_column])

        else:
            st.write("Waiting for a file upload")

    # ANNOUNCEMENTS
    if options == "Announcements":
        col1, col2, col3 = st.columns([2, 6, 2])

        with col1:
            st.write("")

        with col2:
            st.markdown("""<br>""", unsafe_allow_html=True)
            st.image("assets/updates.gif")
            st.markdown("""<h2>More Updates on the way...</h2>""",
                        unsafe_allow_html=True)

    # EMPLOYEE CHURN ANALYSIS
    if options == "Employee Churn Analysis":
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.cluster import KMeans
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn import metrics

        # Load the pre-trained model
        model = pickle.load(
            open('pretrained models/employee_churn_model.pkl', 'rb'))

        # Load the CSV file
        if os.path.exists('./datasets/HR_comma_sep.csv'):
            df = pd.read_csv('datasets/HR_comma_sep.csv', index_col=None)

        features_for_prediction = ['satisfaction_level', 'last_evaluation', 'number_project',
                                   'average_montly_hours', 'time_spend_company', 'Work_accident',
                                   'promotion_last_5years', 'Departments ', 'salary']

        def user_report():
            # Assuming df is your DataFrame containing historical employee data
            if os.path.exists('./datasets/HR_comma_sep.csv'):
                df = pd.read_csv('datasets/HR_comma_sep.csv', index_col=None)

            user_data = {}

            # Iterate over features to get user input
            for feature in features_for_prediction:
                if df[feature].dtype == 'float64':
                    # If the feature is of type float, use float values for slider
                    user_data[feature] = st.sidebar.slider(f'Select {feature}', float(df[feature].min()),
                                                           float(df[feature].max()), float(df[feature].mean()))
                elif df[feature].dtype == 'int64':
                    # If the feature is of type int, use int values for slider
                    user_data[feature] = st.sidebar.slider(f'Select {feature}', int(df[feature].min()),
                                                           int(df[feature].max()), int(df[feature].mean()))
                else:
                    # Handle other data types as needed
                    user_data[feature] = st.sidebar.text_input(
                        f'Enter {feature}', df[feature].iloc[0])

            return pd.DataFrame(user_data, index=[0])

        def descriptive_statistics():
            st.title(" Employee Attrition Analysis")
            b = df.describe()
            st.dataframe(b)

        df['Departments '] = LabelEncoder().fit_transform(df['Departments '])
        df['salary'] = LabelEncoder().fit_transform(df['salary'])

        if df.isnull().any().any():
            df = df.fillna(df.mean())

        with st.sidebar:
            st.image('assets/employee.gif')
            st.title("Employee Churn Analysis")
            choice = st.radio("Navigation",
                              ["Profiling", "Stayed vs. Left: Employee Data Comparison",
                               "Descriptive Statistics Overview",
                               "Employees Left", "Show Value Counts", "Number of Projects Distribution",
                               "Time Spent in Company",
                               "Employee Count by Features", "Clustering of Employees who Left",
                               "Employee Clustering Analysis", "Predict Churn"])
            st.info("This Module is a user-friendly application for data analytics. It enables"
                    " exploration of employee data and predicts turnover, aiding HR professionals and data"
                    " enthusiasts in making informed decisions.")

        if choice == "Profiling":
            st.title("Data Profiling Dashboard")
            a = df.head()
            st.dataframe(a)

        if choice == "Stayed vs. Left: Employee Data Comparison":
            st.title(
                "Employee Retention Analysis: Comparing Characteristics of Stayed and Left Groups")
            left = df.groupby('left')
            b = left.mean()
            st.dataframe(b)

        if choice == "Descriptive Statistics Overview":
            descriptive_statistics()

        if choice == "Employees Left":
            st.title("Data Visualization")
            left_count = df.groupby('left').count()
            st.bar_chart(left_count['satisfaction_level'])

        if choice == "Show Value Counts":
            st.title("Employee Left Counts")
            left_counts = df.left.value_counts()
            st.write(left_counts)
            st.bar_chart(left_counts)

        if choice == "Number of Projects Distribution":
            st.title("Employees' Project Distribution")
            num_projects = df.groupby('number_project').count()
            plt.bar(num_projects.index.values,
                    num_projects['satisfaction_level'])
            plt.xlabel('Number of Projects')
            plt.ylabel('Number of Employees')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

        if choice == "Time Spent in Company":
            st.title("Data Visualization")
            time_spent = df.groupby('time_spend_company').count()
            plt.bar(time_spent.index.values, time_spent['satisfaction_level'])
            plt.xlabel('Number of Years Spent in Company')
            plt.ylabel('Number of Employees')
            st.pyplot()

        if choice == "Employee Count by Features":
            st.title("Data Visualization")
            features = ['number_project', 'time_spend_company', 'Work_accident', 'left', 'promotion_last_5years',
                        'Departments ', 'salary']

            fig, axes = plt.subplots(4, 2, figsize=(10, 15))

            for i, j in enumerate(features):
                row, col = divmod(i, 2)
                sns.countplot(x=j, data=df, ax=axes[row, col])
                axes[row, col].tick_params(axis="x", rotation=90)
                axes[row, col].set_title(f"No. of Employees - {j}")

            plt.tight_layout()
            st.pyplot(fig)

        if choice == "Clustering of Employees who Left":
            X = df[['satisfaction_level', 'last_evaluation', 'number_project',
                    'average_montly_hours', 'time_spend_company', 'Work_accident',
                    'promotion_last_5years', 'Departments ', 'salary']]
            y = df['left']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42)
            gb = GradientBoostingClassifier()
            gb.fit(X_train, y_train)
            y_pred = gb.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            st.title("Gradient Boosting Classifier Model Evaluation")
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision)
            st.write("Recall:", recall)

            y_pred_all = gb.predict(X)
            diff_all_df = pd.DataFrame({
                'Sample': range(len(y)),
                'Actual': y,
                'Predicted': y_pred_all
            })
            diff_all_df['Correct'] = (
                diff_all_df['Actual'] == diff_all_df['Predicted']).astype(int)
            diff_counts = diff_all_df.groupby(
                'Correct').size().reset_index(name='Count')
            fig_diff_all = px.bar(diff_counts, x='Correct', y='Count', color='Correct',
                                  labels={'Correct': 'Prediction Correctness',
                                          'Count': 'Number of Samples'},
                                  title='Actual vs Predicted for All Data',
                                  color_discrete_map={0: 'red', 1: 'green'})
            fig_diff_all.update_layout(showlegend=False)
            st.plotly_chart(fig_diff_all)

        if choice == "Employee Clustering Analysis":
            st.title("Employee Clustering Analysis")
            user_data = user_report()
            user_data['Departments '] = LabelEncoder(
            ).fit_transform(user_data['Departments '])
            user_data['salary'] = LabelEncoder(
            ).fit_transform(user_data['salary'])
            if user_data.isnull().any().any():
                user_data = user_data.fillna(user_data.mean())
            features_for_clustering = ['satisfaction_level', 'last_evaluation', 'number_project',
                                       'average_montly_hours', 'time_spend_company', 'Work_accident',
                                       'promotion_last_5years', 'Departments ', 'salary']
            scaler = StandardScaler()
            user_data[['satisfaction_level', 'last_evaluation', 'average_montly_hours']] = scaler.fit_transform(
                user_data[['satisfaction_level', 'last_evaluation', 'average_montly_hours']])
            X = df[features_for_clustering]
            num_clusters = st.slider(
                "Select Number of Clusters", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X)
            fig_clusters = px.scatter_3d(df, x='satisfaction_level', y='last_evaluation', z='average_montly_hours',
                                         color='Cluster', opacity=0.7, title='Employee Clusters')
            st.plotly_chart(fig_clusters)

        if choice == "Predict Churn":
            st.title("Employee Churn Prediction")
            user_report_data = {
                'salary': st.sidebar.selectbox('Salary', df['salary'].unique())
            }
            user_data = pd.DataFrame(user_report_data, index=[0])
            st.header('Employee Data for Prediction')
            st.write(user_data)
            features_for_prediction = ['satisfaction_level', 'last_evaluation', 'number_project',
                                       'average_montly_hours', 'time_spend_company', 'Work_accident',
                                       'promotion_last_5years', 'Departments ', 'salary']
            missing_columns = set(features_for_prediction) - \
                set(user_data.columns)

            if missing_columns:
                st.error(f"Columns {missing_columns} not found in user data.")
            else:
                X_pred = user_data[features_for_prediction]
                X_pred.columns = ['satisfaction_level', 'last_evaluation', 'number_project',
                                  'average_montly_hours', 'time_spend_company', 'Work_accident',
                                  'promotion_last_5years', 'Departments ', 'salary']
                churn_prediction = model.predict(X_pred)
                st.subheader('Churn Prediction Result')
                st.write(churn_prediction)
