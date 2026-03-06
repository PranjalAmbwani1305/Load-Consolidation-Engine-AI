import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import random

st.set_page_config(page_title="LoRRI Load Optimization Engine", layout="wide")

st.title("🚚 LoRRI – AI Load Consolidation Optimization Engine")

# -------------------------
# Load Data
# -------------------------

shipments = pd.read_csv("shipments.csv")
fleet = pd.read_csv("fleet.csv")
routes = pd.read_csv("routes.csv")

# -------------------------
# Executive Metrics
# -------------------------

st.subheader("Executive Overview")

c1,c2,c3 = st.columns(3)

c1.metric("Total Shipments",len(shipments))
c2.metric("Fleet Size",len(fleet))
c3.metric("Routes",len(routes))

st.divider()

# -------------------------
# Shipment Distribution
# -------------------------

st.subheader("Shipment Distribution")

fig = px.histogram(
    shipments,
    x="destination",
    title="Shipments per Destination"
)

st.plotly_chart(fig,use_container_width=True)

# -------------------------
# AI Clustering
# -------------------------

def run_clustering(df):

    coords = df[["destination_lat","destination_lon"]]

    model = KMeans(n_clusters=5,random_state=42)

    df["cluster"] = model.fit_predict(coords)

    return df

# -------------------------
# Compatibility Score
# -------------------------

def compatibility(row):

    score = (
        row["weight_kg"]*0.4 +
        row["volume_m3"]*0.3 +
        row["sla_hours"]*0.3
    )

    return round(score,2)

# -------------------------
# Truck Allocation
# -------------------------

def allocate_trucks(shipments,fleet):

    trucks = fleet["truck_id"].tolist()

    assignments = []

    for _,row in shipments.iterrows():

        truck = random.choice(trucks)

        assignments.append({
            "shipment":row["shipment_id"],
            "truck":truck,
            "destination":row["destination"]
        })

    return pd.DataFrame(assignments)

# -------------------------
# Run Optimization
# -------------------------

if st.button("Run AI Optimization"):

    st.subheader("AI Shipment Clustering")

    shipments_clustered = run_clustering(shipments)

    cluster_fig = px.scatter(
        shipments_clustered,
        x="destination_lat",
        y="destination_lon",
        color="cluster",
        title="Geo-Spatial Shipment Clusters"
    )

    st.plotly_chart(cluster_fig,use_container_width=True)

    st.subheader("Compatibility Analysis")

    shipments_clustered["compatibility"] = shipments_clustered.apply(
        compatibility,
        axis=1
    )

    score_fig = px.histogram(
        shipments_clustered,
        x="compatibility",
        title="Compatibility Score Distribution"
    )

    st.plotly_chart(score_fig,use_container_width=True)

    st.subheader("Truck Allocation")

    allocation = allocate_trucks(shipments_clustered,fleet)

    st.dataframe(allocation.head())

    # -------------------------
    # Optimization Metrics
    # -------------------------

    baseline_trips = len(shipments)
    optimized_trips = allocation["truck"].nunique()

    improvement = round(
        (baseline_trips - optimized_trips) / baseline_trips * 100,
        2
    )

    st.subheader("Optimization Impact")

    col1,col2,col3 = st.columns(3)

    col1.metric("Baseline Trips",baseline_trips)
    col2.metric("Optimized Trips",optimized_trips)
    col3.metric("Trip Reduction",str(improvement)+" %")

    impact = pd.DataFrame({
        "Scenario":["Baseline","Optimized"],
        "Trips":[baseline_trips,optimized_trips]
    })

    impact_fig = px.bar(
        impact,
        x="Scenario",
        y="Trips",
        title="Baseline vs Optimized Trips"
    )

    st.plotly_chart(impact_fig,use_container_width=True)

    st.success("Optimization Completed Successfully")
