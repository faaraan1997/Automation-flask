from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

@app.route('/')
def index():
    # Load the data
    df =  pd.read_csv('retail.csv')

    # Plot 1: Total Sales by Product Category
    fig1 = px.bar(df, x='Product Category', y='Total Amount', title='Total Sales by Product Category')
    plot1 = pio.to_html(fig1, full_html=False)

    # # Plot 2: Sales Over Time
    # fig2 = px.line(df, x='Date', y='Total Amount', title='Sales Over Time')
    # plot2 = pio.to_html(fig2, full_html=False)

    # Plot 3: Distribution of Customer Ages
    fig3 = px.histogram(df, x='Age', title='Distribution of Customer Ages')
    plot3 = pio.to_html(fig3, full_html=False)

    # Plot 4: Total Sales by Gender
    fig4 = px.bar(df, x='Gender', y='Total Amount', title='Total Sales by Gender', color='Gender')
    plot4 = pio.to_html(fig4, full_html=False)

    # Plot 5: Scatter plot of Quantity vs Total Amount, colored by Product Category
    fig5 = px.scatter(df, x='Quantity', y='Total Amount', color='Product Category', title='Quantity vs Total Amount')
    plot5 = pio.to_html(fig5, full_html=False)

    return render_template('index.html', plot1=plot1,  plot3=plot3, plot4=plot4, plot5=plot5)


if __name__ == '__main__':
    app.run(debug=True)
