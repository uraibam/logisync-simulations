# LogiSync - Supply Chain Logistics Simulations

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

LogiSync is an interactive web application that simulates various supply chain logistics scenarios using stochastic processes and Markov chains. The app helps visualize and analyze inventory management, demand forecasting, warehouse operations, and logistics flows.

## Features

- **8 Different Supply Chain Simulations**:
  - Inventory Level Fluctuation (Markov Chains)
  - Demand Forecasting (Poisson Process)
  - Warehouse Queue Simulation
  - Inventory Replenishment Policy
  - Order Arrival Variability (Exponential)
  - Multi-Warehouse Logistics (Sankey Diagrams)
  - Priority-based Order Fulfillment (MDP)
  - Delay Simulation (Binomial Distribution)

- **Interactive Controls**:
  - Adjustable parameters for each simulation
  - Real-time visualization updates
  - Detailed statistical summaries

- **Visualizations**:
  - Matplotlib line charts and histograms
  - Plotly Sankey diagrams for logistics flows
  - Interactive state transition diagrams

## How to Use

1. Select a simulation from the sidebar
2. Adjust parameters using the interactive controls
3. View the visualizations and analysis
4. Explore different scenarios by changing inputs

## Deployment

This app is deployed on Streamlit Community Cloud. You can access it at: [https://logisync-supplychain.streamlit.app](https://logisync-supplychain.streamlit.app)

## Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/logisync-supplychain.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Technologies Used

- Python 3.9+
- Streamlit
- NumPy
- Matplotlib
- Pandas
- Plotly
- SciPy

## Future Enhancements

- Add database integration for saving simulation results
- Implement more complex supply chain networks
- Add machine learning for predictive analytics
- Include real-world dataset integration

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License
