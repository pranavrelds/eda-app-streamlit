#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#Set Title
st.title('Machine Learning App by Pranav')

#Show Image
image  = Image.open('data.jpg')
st.image(image, use_column_width = True)

#Set Subtitle
st.write("""## A Streamlit app to explore various classifiers on your dataset""")

def main():
	stage = ['EDA', 'Data Visualization', 'ML model Building']
	sidebar_option = st.sidebar.selectbox('Stage', stage)

# EDA stage
	if sidebar_option == 'EDA':
		st.subheader('Exploratory Data Analysis')
		file = st.file_uploader('Upload Dataset', type = ['csv', 'xlsx','txt','json'])
		if file is not None:
			st.success('Data upload successfully')

		if file is not None:
			df = pd.read_csv(file)
			st.dataframe(df)

			if st.checkbox('Shape of Dataset'):
				st.write(df.shape)

			if st.checkbox('Display Summary'):
				st.write(df.describe().T)

			if st.checkbox('Features in Dataset'):
				st.write(df.columns)

			if st.checkbox('Select Columns'):
				selected_columns = st.multiselect('Selected Columns : ',df.columns)
				selected_columns_df = df[selected_columns]
				st.dataframe(selected_columns_df)

			if st.checkbox('Check Number of Null values'):
				st.write(df.isnull().sum())

			if st.checkbox('Data Type'):
				st.write(df.dtypes)

			if st.checkbox('Display Correaltion Heatmap'):
				plt.figure(figsize=(15,10))
				st.write(sns.heatmap(df.corr(), cmap='viridis', square=True, vmax=1, annot= True))
				st.pyplot()

# Data Visualization stage
	elif sidebar_option == 'Data Visualization':
		st.subheader('Data Visualization')
		file = st.file_uploader('Upload Dataset', type = ['csv', 'xlsx','txt','json'])
		if file is not None:
			st.success('Data upload successfully')

		if file is not None:
			df = pd.read_csv(file)
			st.dataframe(df)

			if st.checkbox('Display Pairplot on Whole Data'):				
				st.write(sns.pairplot(df,diag_kind='kde'))
				st.pyplot()

			if st.checkbox('Select Columns to Plot'):
				selected_columns = st.multiselect('Selected Columns : ',df.columns)
				selected_columns_df = df[selected_columns]
				st.dataframe(selected_columns_df)

			if st.checkbox('Display Pairplot on Selected Columns'):
				st.write(sns.pairplot(selected_columns_df,diag_kind='kde'))
				st.pyplot()

			if st.checkbox('Display Pie Chart'):
				pie_column =st.selectbox("select column to display Pie Chart",df.columns)
				pieChart=df[pie_column].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pieChart)
				st.pyplot()

			if st.checkbox('Display Bar Chart'):
				bar_chart_column=st.selectbox("select column to display Bar Chart",df.columns)
				st.bar_chart(df[bar_chart_column])
				# st.write(pieChart)
				# st.pyplot()

			if st.checkbox('Display Box Plot'):
				box_plot_column=st.selectbox("select column to display Box Plot",df.columns)
				st.write(sns.boxplot(data = df[box_plot_column] , orient = 'h'))
				st.pyplot()

# ML Model Building stage
	elif sidebar_option == 'ML model Building':
		st.subheader('ML model Building')
		file = st.file_uploader('Upload Dataset', type = ['csv', 'xlsx','txt','json'])
		if file is not None:
			st.success('Data upload successfully')

		if file is not None:
			df = pd.read_csv(file)
			st.dataframe(df.head(50))

		if st.checkbox('Select the Target Feature'):
			target = st.selectbox("Target Feature is",df.columns)
			y = df[target]
			X = df.drop(target,axis=1)

		if st.checkbox('Standardize data'):
			scaled_features = StandardScaler().fit_transform(X)
			scaled_df = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)

		if st.checkbox('Perform train_test_split'):
			test_size = st.slider('Select test size : ',0.1,0.99)
			X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=test_size, random_state=42)
			st.write(pd.DataFrame({'X_train': [X_train.shape],'X_test': [X_test.shape], 'y_train': [y_train.shape],'y_test': [y_test.shape]}))

		list_of_classifiers = ['K Nearest Neighbors','Support Vector Classifier','Logistic Regression','Decision Tree']
		select_classifier = st.sidebar.selectbox('Select the Classifier : ',list_of_classifiers)

		def add_parameter(classifier_name):
			params = dict()
			if classifier_name == 'K Nearest Neighbors':
				n_neighbors = st.sidebar.slider('K-Neighbors', 1,100)
				params['n_neighbors']= n_neighbors

			if classifier_name == 'Support Vector Classifier':
				select_kernel = st.sidebar.selectbox('Select Kernel for SVC', ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')) #Hyper parameter
				C = st.sidebar.slider('C', 0.01,15.0) #Hyper parameter
				gamma = st.sidebar.slider('Gamma',0.01 , 8.0) #Hyper parameter
				params['C'] = C
				params['gamma'] = gamma
				params['kernel'] = select_kernel

			if classifier_name == 'Logistic Regression':
				list_of_penalties = ['l2', 'l1', 'elasticnet', 'none']
				penalty = st.sidebar.selectbox('Select Penalty for Logistic Regression ',list_of_penalties,list_of_penalties.index('l2')) #Hyper parameter
				list_of_solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
				solver = st.sidebar.selectbox('Select Solver for Logistic Regression ', list_of_solver,list_of_solver.index('lbfgs')) #Hyper parameter
				C = st.sidebar.slider('C', 0.01,15.0,1.0)
				params['C'] = C
				params['penalty'] = penalty
				params['solver'] = solver
			
			return params
		params = add_parameter(select_classifier)

		#Access the clssifier after selecting
		def get_classifier(classifier_name, params):
			clf = None
			if classifier_name == 'Support Vector Classifier':
				clf = SVC(C= params['C'], kernel = params['kernel'], gamma= params['gamma']) #

			elif classifier_name=='K Nearest Neighbors':
				clf=KNeighborsClassifier(n_neighbors=params['n_neighbors'])

			elif classifier_name=='Logistic Regression':
				clf=LogisticRegression(penalty = params['penalty'],C = params['C'], solver =params['solver'])

			elif classifier_name=='Decision Tree':
				clf=DecisionTreeClassifier()

			else:
				st.warning("You didn't select any option, please select at least one Classifier")
			return clf

		#call the function
		clf = get_classifier(select_classifier, params)

		if st.checkbox('Fit the training data to classifier'):
			clf.fit(X_train,y_train)

		if st.checkbox('Predict the label'):
			y_pred = clf.predict(X_test)

		if st.checkbox('Get Accuracy Score'):
			st.write('Accuracy Score % : ', accuracy_score(y_test,y_pred)*100)

		if st.checkbox('Get Classification Report'):
			cr = classification_report(y_test,y_pred, output_dict=True)
			cr_df = pd.DataFrame(cr).T
			st.dataframe(cr_df)			

		if st.checkbox('Get Confusion Matrix'):
			cm = confusion_matrix(y_test, y_pred)
			st.write(sns.heatmap(cm, annot=True))
			st.pyplot()			
	else: 
		st.warning('Please select at least one option')

if __name__ == '__main__':
	main()