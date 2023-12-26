import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import random
from geneticalgo import generate_portfolios, calculate_fitness_function, calculate_sharpe_ratio, select_fittest_population, crossover, mutate, genetic_algorithm,generate_pie_chart,calculate_stock_score

# Load the CSV file
csv_path = "EQUITY_L.csv"
df = pd.read_csv(csv_path)

df[' DATE OF LISTING'] = pd.to_datetime(df[' DATE OF LISTING'], format='%d-%b-%y')

# # Filter out entries after the year 2009
df_filtered = df[df[' DATE OF LISTING'].dt.year <= 2013]


df = df_filtered[['SYMBOL', 'NAME OF COMPANY', ' FACE VALUE', 'YahooEquiv', 'Yahoo_Equivalent_Code']]


# Extract stock names and symbols
stock_names = df['NAME OF COMPANY'].tolist()
stock_symbols = df['YahooEquiv'].tolist()

# Streamlit App
st.title("Portfolio Optimization App")

# Get the user's stock selection using a multiselect dropdown
selected_stock_names = st.multiselect("Select Stocks", stock_names)

# Map selected stock names to corresponding symbols
selected_stock_symbols = [stock_symbols[stock_names.index(name)] for name in selected_stock_names]

st.write("")
st.write("")
st.write("")
st.write("")



# fitness score parameters' weights

sharpe_weight = st.slider("Sharpe Ratio Weight", 0.0, 1.0, 0.6, step=0.01)
fundamental_weight = st.slider("Fundamental Analysis Weight", 0.0, 1.0, 0.2, step=0.01)
diversification_weight = st.slider("HHI Diversification Weight", 0.0, 1.0, 0.2, step=0.01)

total_weight = sharpe_weight + fundamental_weight + diversification_weight

if total_weight != 1.0:
    st.error("Error: The sum of weights must be equal to 1.0. Please adjust the weights.")

st.write(f"Current Weights - Sharpe: {sharpe_weight}, Fundamental: {fundamental_weight}, HHI: {diversification_weight}")

st.write("")
st.write("")
st.write("")
st.write("")
# Displaying the selected stocks and their symbols
st.write("Selected Stocks:", selected_stock_names)
st.write("Corresponding Symbols:", selected_stock_symbols)

# Downloading stock data
stocks_data_list = []
start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2023, 1, 1)

st.write("")
st.write("")
st.write("")
st.write("")

if selected_stock_symbols and st.button("Run", key='centered_button'):
    for symbol in selected_stock_symbols:
        df_stock = yf.download(symbol, start=start, end=end)[['Close']]
        df_stock.columns = [symbol]
        stocks_data_list.append(df_stock)

    final_df = pd.concat(stocks_data_list, axis=1)


    # dividing the data on monthly basis 
    monthly_closing_prices = final_df.resample('M').last()
    monthly_returns = monthly_closing_prices.pct_change()
    correlation_matrix = monthly_closing_prices.corr() # corelation/covariance of one stock with another
    expected_returns_mean = monthly_returns.mean() # average of monthly returns

    # Run the genetic algorithm
    num_portfolios = 5000
    generations = 40
    mutation_rate = 0.1


    # stock_scores = [random.uniform(0, 1) for _ in range(len(selected_stock_symbols))]
    stock_scores = []
    for i in range(len(selected_stock_symbols)):
        stock_scores.append(calculate_stock_score(selected_stock_symbols[i]))


    best_portfolio = genetic_algorithm(selected_stock_symbols, correlation_matrix, expected_returns_mean, num_portfolios, generations, mutation_rate, stock_scores,sharpe_weight , fundamental_weight ,diversification_weight)

    # Display portfolio analytics
    sharpe_optimized = calculate_sharpe_ratio(best_portfolio, correlation_matrix, expected_returns_mean, selected_stock_symbols, 0.012)


    benchmark_portfolio = [1/len(selected_stock_symbols) for _ in range(len(selected_stock_symbols))]

    sharpe_benchmark = calculate_sharpe_ratio(benchmark_portfolio, correlation_matrix, expected_returns_mean, selected_stock_symbols, 0.012)

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")


    st.write("**Optimized Portfolio Sharpe Ratio:**", f"**{sharpe_optimized:.2f}**", f"**Risk adjusted returns : {sharpe_optimized*100}**")
    st.write("**Benchmark Portfolio Sharpe Ratio:**", f"**{sharpe_benchmark:.2f}**",f"**Risk adjusted returns : {sharpe_benchmark*100}**")

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")


    # Display pie charts
    st.subheader("Optimized Portfolio Allocation")
    # Create a Streamlit column layout
    col1, col2 = st.columns(2)  # Adjust the number of columns as needed

    # Column 1: Display the pie chart
    with col1:
        pie_chart_fig = generate_pie_chart(best_portfolio, "Best Portfolio", selected_stock_symbols)
        st.pyplot(pie_chart_fig)

    # Column 2: Display the table
    with col2:
        # Create a DataFrame for the table
        data = {'Stock Name': selected_stock_symbols, 'Allocation Percentage': best_portfolio}
        df = pd.DataFrame(data)
        st.table(df)


    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")


    st.subheader("Benchmark Portfolio Allocation")
    col1,col2 = st.columns(2)

    with col1:
        pie_chart_fig = generate_pie_chart(benchmark_portfolio, "Benchmark Portfolio", selected_stock_symbols)
        st.pyplot(pie_chart_fig)

    with col2:
        # Create a DataFrame for the table
        data = {'Stock Name': selected_stock_symbols, 'Allocation Percentage': benchmark_portfolio}
        df = pd.DataFrame(data)
        # Display the table using st.table
        st.table(df)

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
