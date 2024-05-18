from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import plotly.express as px
import plotly.io as pio
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
        string_columns = df.select_dtypes(include=['object']).columns.tolist()
        return render_template('select_encoding.html', file_name=file.filename, columns=df.columns.tolist(), string_columns=string_columns)
    return redirect(url_for('index'))

@app.route('/select_encoding', methods=['POST'])
def select_encoding():
    file_name = request.form['file_name']
    one_hot_columns = request.form.getlist('one_hot_columns')
    label_columns = request.form.getlist('label_columns')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    df = pd.read_csv(file_path)
    
    # Apply One Hot Encoding
    if one_hot_columns:
        df = pd.get_dummies(df, columns=one_hot_columns)
    
    # Apply Label Encoding
    if label_columns:
        le = LabelEncoder()
        for col in label_columns:
            df[col] = le.fit_transform(df[col])
    
    updated_columns = df.columns.tolist()
    
    # Save the encoded dataframe to a new CSV file
    encoded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'encoded_' + file_name)
    df.to_csv(encoded_file_path, index=False)
    
    return render_template('select_target.html', file_name='encoded_' + file_name, columns=updated_columns)

@app.route('/select_target', methods=['POST'])
def select_target():
    file_name = request.form['file_name']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    df = pd.read_csv(file_path)
    model_type = request.form['model']
    target_variable = request.form['target']
    
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'Linear Regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        score = model.score(X_test, y_test)
        metrics = {
            'Mean Squared Error': mse,
            'R2 Score': score
        }
        return render_template('results.html', file_name=file_name, model_type=model_type, metrics=metrics, columns=df.columns.tolist(), target=target_variable)
    
    elif model_type == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='macro')
        precision = precision_score(y_test, predictions, average='macro')
        recall = recall_score(y_test, predictions, average='macro')
        metrics = {
            'Accuracy': acc,
            'Confusion Matrix': conf_matrix,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall
        }
        return render_template('results.html', file_name=file_name, model_type=model_type, metrics=metrics, columns=df.columns.tolist(), target=target_variable)

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
