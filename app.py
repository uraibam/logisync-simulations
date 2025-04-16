import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import itertools
from scipy import stats

# Configure page settings
st.set_page_config(
    page_title="LogiSync - Supply Chain Simulations",
    page_icon="🚚",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stMarkdown h1 {
        color: #4b7d83;
        font-family: 'Poppins', sans-serif;
    }
    .stInfo {
        border-left: 5px solid #4b7d83;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown(
    """
<div style='text-align: center; padding: 20px 0; background-color: #0e1117; margin-bottom: 30px; border-radius: 10px;'>
    <h1 style='font-family:Poppins; color:#4b7d83; font-size:60px; margin-bottom: 0;'>
        LogiSync
    </h1> 
    <h1 style='font-family:Poppins; color:#4b7d83; font-size:40px; margin-bottom: 0;'>
        Supply Chain Logistics Simulations
    </h1> 
    <p style='font-size:22px; color:gray; margin-top: 5px;'>
        Explore warehouse flow, delays, and decisions with stochastic models
    </p>
    <hr style='border: 1px solid #ccc; width: 60%; margin: auto; margin-top: 10px;'>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar controls
st.sidebar.title("🔧 Simulation Control")

simulations = [
    "Inventory Level Fluctuation",
    "Demand Forecasting",
    "Warehouse Queue",
    "Inventory Replenishment",
    "Order Arrival Variability",
    "Multi-Warehouse Logistics",
    "Priority-based Order",
    "Delay Simulation",
]

selected_sim = st.sidebar.radio("Select Simulation", simulations)

# Cached functions for better performance
@st.cache_data
def run_inventory_simulation(days, restock_level, daily_consumption):
    inventory = [restock_level]
    for _ in range(days):
        consumed = np.random.poisson(daily_consumption)
        new_level = max(0, inventory[-1] - consumed)
        if new_level < restock_level * 0.4:
            new_level += restock_level
        inventory.append(new_level)
    return inventory

@st.cache_data
def run_demand_forecast(days, lambda_demand):
    return np.random.poisson(lambda_demand, size=days)

@st.cache_data
def run_warehouse_queue(sim_days, arrivals, service):
    queue = [0]
    for _ in range(sim_days):
        incoming = np.random.poisson(arrivals)
        served = np.random.poisson(service)
        queue.append(max(0, queue[-1] + incoming - served))
    return queue

@st.cache_data
def run_replenishment_simulation(days, daily_consumption, replenish_qty, threshold):
    inv = [replenish_qty]
    for _ in range(days):
        consume = np.random.poisson(daily_consumption)
        level = max(0, inv[-1] - consume)
        if level < threshold:
            level += replenish_qty
        inv.append(level)
    return inv

@st.cache_data
def run_order_arrival_simulation(sample_size, mean_interval):
    return np.random.exponential(scale=mean_interval, size=sample_size)

@st.cache_data
def run_priority_simulation(sim_steps):
    states = ["Idle", "Processing High", "Processing Low"]
    actions = ["Serve High", "Serve Low"]
    transitions = {
        ("Idle", "Serve High"): [0.0, 1.0, 0.0],
        ("Idle", "Serve Low"): [0.0, 0.0, 1.0],
        ("Processing High", "Serve High"): [0.6, 0.4, 0.0],
        ("Processing High", "Serve Low"): [0.4, 0.0, 0.6],
        ("Processing Low", "Serve High"): [0.4, 0.5, 0.1],
        ("Processing Low", "Serve Low"): [0.7, 0.0, 0.3],
    }
    
    state = "Idle"
    trace = [state]
    for _ in range(sim_steps):
        action = np.random.choice(actions)
        probs = transitions.get((state, action), [1.0, 0.0, 0.0])
        state = np.random.choice(states, p=probs)
        trace.append(state)
    return trace

@st.cache_data
def run_delay_simulation(days, shipments, delay_prob):
    return np.random.binomial(shipments, delay_prob, size=days)

# SIMULATION 1: Inventory Level Fluctuation
if selected_sim == "Inventory Level Fluctuation":
    st.header("📊 1. Inventory Level Fluctuation (Markov Chains)")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        days_1 = st.slider("Number of Days", 10, 100, 30)
        restock_level = st.number_input("Restock Level (units)", value=100)
        daily_consumption = st.number_input("Average Daily Consumption (Poisson λ)", value=10)
    
    inventory = run_inventory_simulation(days_1, restock_level, daily_consumption)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(inventory, color="green")
        ax1.set_title("Inventory Level Over Time", fontsize=14)
        ax1.set_xlabel("Days", fontsize=12)
        ax1.set_ylabel("Inventory Level (Units)", fontsize=12)
        ax1.grid(True)
        st.pyplot(fig1)

    st.info(
        f"""
        ### 📦 Inventory Analysis Summary
        - 📉 Average Inventory Level: **{np.mean(inventory):.1f} units**
        - ⚠️ Restocked **{sum([1 for i in range(len(inventory)-1) if inventory[i+1] > inventory[i]]):d} times**
        - ⛔ Minimum Level Reached: **{min(inventory)} units**
        - 📈 Standard Deviation: **{np.std(inventory):.1f} units**
        - 📊 Coefficient of Variation: **{(np.std(inventory)/np.mean(inventory)):.2%}**
        - 📈 Peak Inventory Level: **{max(inventory)} units**
        - 🔝 Longest Periods of Low Inventory: **{max([len(list(group)) for _, group in itertools.groupby(inventory, lambda x: x < restock_level * 0.4) if not _]):d} days**

        **Explanation:** This simulation models inventory level fluctuations using a Markov Chain approach. 
        Daily consumption is modeled as a Poisson random variable, representing random demand variations. 
        When inventory falls below 40% of the restock level, it is replenished to simulate restocking actions.
        """
    )

# SIMULATION 2: Demand Forecasting
elif selected_sim == "Demand Forecasting":
    st.header("📈 2. Demand Forecasting (Poisson Process)")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        days_2 = st.slider("Forecast Days", 10, 100, 30)
        lambda_demand = st.number_input("λ for Poisson Demand", value=15)
    
    demand = run_demand_forecast(days_2, lambda_demand)

    with col1:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(range(days_2), demand, color="skyblue")
        ax2.set_title("Simulated Demand Forecast", fontsize=14)
        ax2.set_xlabel("Days", fontsize=12)
        ax2.set_ylabel("Demand (Units)", fontsize=12)
        ax2.grid(True)
        st.pyplot(fig2)

    st.info(
        f"""
        ### 🛍️ Demand Forecast Summary
        - 📦 Total Forecasted Demand: **{np.sum(demand)} units**
        - 📅 Avg Daily Demand: **{np.mean(demand):.1f} units**
        - 📈 Highest Demand Day: **{np.argmax(demand)+1} (with {max(demand)} units)**
        - 📉 Standard Deviation: **{np.std(demand):.1f} units**

        The demand forecasting simulation uses a Poisson process to generate random demand.
        The expected daily demand is set to **{lambda_demand}** units with a standard deviation of **{np.sqrt(lambda_demand):.1f}**.
        """
    )

# SIMULATION 3: Warehouse Queue
elif selected_sim == "Warehouse Queue":
    st.header("🏭 3. Warehouse Queue Simulation")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        sim_days = st.slider("Simulation Days", 10, 100, 30)
        arrivals = st.number_input("Average Daily Arrivals (Poisson)", value=12)
        service = st.number_input("Average Daily Service (Poisson)", value=10)
    
    queue = run_warehouse_queue(sim_days, arrivals, service)

    with col1:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(queue, color="purple")
        ax3.set_title("Warehouse Queue Length Over Time", fontsize=14)
        ax3.set_xlabel("Days", fontsize=12)
        ax3.set_ylabel("Queue Length (Units)", fontsize=12)
        ax3.grid(True)
        st.pyplot(fig3)

    st.info(
        f"""
        ### 🧾 Queue Analysis Summary
        - 📈 Maximum Queue Length: **{max(queue)}**
        - ⏳ Average Queue Length: **{np.mean(queue):.1f}**
        - 🧍 Days Queue Increased: **{sum(np.diff(queue) > 0)}**

        The Warehouse Queue simulation models the arrival and service of items in a warehouse.
        """
    )

# SIMULATION 4: Inventory Replenishment Policy
elif selected_sim == "Inventory Replenishment":
    st.header("🔄 4. Inventory Replenishment Policy")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        days_1 = st.slider("Days to Simulate", 10, 100, 30)
        daily_consumption = st.number_input("Daily Consumption λ", value=10)
        replenish_qty = st.number_input("Replenishment Quantity", value=50)
        threshold = st.number_input("Restock Threshold", value=40)
    
    inv = run_replenishment_simulation(days_1, daily_consumption, replenish_qty, threshold)

    with col1:
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(inv, color="orange")
        ax4.set_title("Inventory Under Replenishment Policy", fontsize=14)
        ax4.set_xlabel("Days", fontsize=12)
        ax4.set_ylabel("Inventory Level (Units)", fontsize=12)
        ax4.grid(True)
        st.pyplot(fig4)

    st.info(
        f"""
        ### 🗃️ Replenishment Summary
        - 🔁 Replenishments Triggered: **{sum([1 for i in range(len(inv)-1) if inv[i+1] > inv[i]]):d}**
        - 📊 Average Inventory: **{np.mean(inv):.1f} units**

        The inventory replenishment policy simulation models the inventory management process.
        """
    )

# SIMULATION 5: Order Arrival Variability
elif selected_sim == "Order Arrival Variability":
    st.header("📬 5. Order Arrival Variability (Exponential)")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        mean_interval = st.slider(
            "Average Interval Between Orders (Days)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
        )
        sample_size = st.slider(
            "Number of Orders", min_value=10, max_value=100, value=30
        )
    
    order_intervals = run_order_arrival_simulation(sample_size, mean_interval)

    with col1:
        fig5, ax5 = plt.subplots(figsize=(10, 4))
        ax5.hist(order_intervals, bins=10, color="teal", edgecolor="black")
        ax5.set_title("Order Arrival Time Distribution", fontsize=14)
        ax5.set_xlabel("Time Between Orders (Days)", fontsize=12)
        ax5.set_ylabel("Frequency", fontsize=12)
        ax5.grid(True)
        st.pyplot(fig5)

    std_dev = np.std(order_intervals)
    cv = std_dev / np.mean(order_intervals)

    st.info(
        f"""
        ### 🕒 Order Variability Summary
        - 📦 Avg Interval: **{np.mean(order_intervals):.2f} days**
        - 📉 Min Interval: **{np.min(order_intervals):.2f} days**
        - 📈 Max Interval: **{np.max(order_intervals):.2f} days**
        - 📐 Standard Deviation: **{std_dev:.2f} days**
        - 🔄 Coefficient of Variation: **{cv:.2f}**

        This simulation explores the variability in order arrival times.
        """
    )

# SIMULATION 6: Multi-Warehouse Logistics
elif selected_sim == "Multi-Warehouse Logistics":
    st.header("🚛 6. Multi-Warehouse Logistics Flow")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        output_factory = st.number_input("Output from Factory to Warehouse A", min_value=0, value=100)
        warehouse_a_to_b = st.number_input("Output from Warehouse A to Warehouse B", min_value=0, value=80)
        warehouse_b_to_retail = st.number_input("Output from Warehouse B to Retail", min_value=0, value=60)
        warehouse_b_to_a = st.number_input("Output from Warehouse B back to Warehouse A", min_value=0, value=30)
        output_factory_to_b = st.number_input("Output from Factory to Warehouse B", min_value=0, value=50)

    labels = ["Factory", "Warehouse A", "Warehouse B", "Retail"]
    source = [0, 0, 1, 1, 2, 2]
    target = [1, 2, 2, 3, 3, 1]
    value = [output_factory, output_factory_to_b, warehouse_a_to_b, warehouse_b_to_retail, warehouse_b_to_a, warehouse_b_to_a]

    with col1:
        fig6 = go.Figure(
            data=[go.Sankey(
                node=dict(
                    pad=10,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    label=labels,
                ),
                link=dict(source=source, target=target, value=value),
            )]
        )
        fig6.update_layout(
            title="Logistics Flow Between Warehouse and Retail",
            title_x=0.5,
            font=dict(size=14),
            height=400,
        )
        st.plotly_chart(fig6, use_container_width=True)

    st.info(
        f"""
        ### 🚚 Logistics Summary
        - 📤 Total Output from Factory: **{output_factory + output_factory_to_b} units**
        - 🧭 Cyclic Movement Detected: **{'Yes' if warehouse_b_to_a > 0 else 'No'}**

        This simulation visualizes the logistics flow of goods across a multi-warehouse supply chain.
        """
    )

# SIMULATION 7: Priority-based Order
elif selected_sim == "Priority-based Order":
    st.header("⚙️ 7. Priority-based Order Fulfillment (Markov Decision Process)")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        sim_steps = st.slider("Simulation Steps", min_value=10, max_value=100, value=30)
    
    trace = run_priority_simulation(sim_steps)
    state_counts = {s: trace.count(s) for s in ["Idle", "Processing High", "Processing Low"]}

    with col1:
        fig7, ax7 = plt.subplots(figsize=(10, 4))
        ax7.plot(trace, marker="o", linestyle="--", color="crimson")
        ax7.set_title("State Transitions in Order Fulfillment", fontsize=14)
        ax7.set_xlabel("Steps", fontsize=12)
        ax7.set_ylabel("State", fontsize=12)
        ax7.set_yticks(range(3))
        ax7.set_yticklabels(["Idle", "Processing High", "Processing Low"])
        ax7.grid(True)
        st.pyplot(fig7)

    st.info(
        f"""
        ### 🧠 Fulfillment Summary (MDP)
        - 🔄 State Distribution: {state_counts}
        - ✅ Most Frequent State: **{max(state_counts, key=state_counts.get)}**

        This simulation models a Priority-based Order Fulfillment System using a Markov Decision Process.
        """
    )

# SIMULATION 8: Delay Simulation
elif selected_sim == "Delay Simulation":
    st.header("🚚 8. Delay Simulation (Markov Chains)")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        delay_prob = st.slider(
            "Probability of Delay", min_value=0.0, max_value=1.0, value=0.2, step=0.05
        )
        shipments = st.slider(
            "Shipments Per Day", min_value=1, max_value=50, value=10
        )
        days_1 = st.slider("Simulation Days", min_value=10, max_value=100, value=30)
    
    delays = run_delay_simulation(days_1, shipments, delay_prob)

    with col1:
        fig8, ax8 = plt.subplots(figsize=(10, 4))
        ax8.plot(range(1, days_1 + 1), delays, marker="o", linestyle="-", color="orange")
        ax8.set_title("Shipment Delays Over Time", fontsize=14)
        ax8.set_xlabel("Day", fontsize=12)
        ax8.set_ylabel("Delayed Shipments", fontsize=12)
        ax8.grid(True)
        st.pyplot(fig8)

    st.info(
        f"""
        ### 📊 Analysis Summary
        - 📦 Total Delayed Shipments: **{np.sum(delays)}**
        - ⏱️ Average Daily Delay: **{np.mean(delays):.1f} shipments**
        - 🚨 Maximum Delay in a Day: **{np.max(delays)} shipments**

        This simulation models shipment delays using a Binomial Distribution.
        """
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>LogiSync Supply Chain Simulations © 2023</div>", 
    unsafe_allow_html=True
)
