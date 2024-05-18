from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
import plotly.express as px
import plotly.io as pio
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        df = pd.read_csv(file_path)
        return render_template('select_model.html', file_name=file.filename, columns=df.columns.tolist())
    return redirect(url_for('index'))

@app.route('/select_model', methods=['POST'])
def select_model():
    file_name = request.form['file_name']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    df = pd.read_csv(file_path)
    model_type = request.form['model']
    encoding = request.form['encoding']
    target_variable = request.form['target']
    
    # Encoding
    if encoding == 'One Hot':
        df = pd.get_dummies(df)
    elif encoding == 'Label':
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    if model_type == 'Linear Regression':
        error = mean_squared_error(y_test, predictions)
    elif model_type == 'Logistic Regression':
        error = accuracy_score(y_test, predictions)
    
    return render_template('results.html', file_name=file_name, error=error, columns=df.columns.tolist(), target=target_variable)

@app.route('/plot', methods=['POST'])
def plot_graph():
    file_name = request.form['file_name']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    df = pd.read_csv(file_path)

    fig1 = px.bar(df, x='Product Category', y='Total Amount', title='Total Sales by Product Category')
    plot1 = pio.to_html(fig1, full_html=False)

    fig3 = px.histogram(df, x='Age', title='Distribution of Customer Ages')
    plot3 = pio.to_html(fig3, full_html=False)

    fig4 = px.bar(df, x='Gender', y='Total Amount', title='Total Sales by Gender', color='Gender')
    plot4 = pio.to_html(fig4, full_html=False)

    fig5 = px.scatter(df, x='Quantity', y='Total Amount', color='Product Category', title='Quantity vs Total Amount')
    plot5 = pio.to_html(fig5, full_html=False)

    return render_template('plots.html', plot1=plot1, plot3=plot3, plot4=plot4, plot5=plot5)

if __name__ == '__main__':
    app.run(debug=True)
