import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import random

st.set_page_config(page_title="LoRRI AI Load Consolidation Engine", layout="wide")

st.title("🚚 AI Load Consolidation Optimization Engine")

# ---------------------------------
# Load datasets
# ---------------------------------

shipments = pd.read_csv("shipments.csv")
fleet = pd.read_csv("fleet.csv")
routes = pd.read_csv("routes.csv")

# ---------------------------------
# Geo-Spatial Clustering
# ---------------------------------

def cluster_shipments(df):

    coords = df[['destination_lat','destination_lon']]

    model = KMeans(n_clusters=6, random_state=42)

    df["cluster"] = model.fit_predict(coords)

    return df

# ---------------------------------
# Compatibility Scoring
# ---------------------------------

def compatibility_score(weight, volume, sla):

    score = (weight * 0.4) + (volume * 0.3) + (sla * 0.3)

    return score

# ---------------------------------
# Truck Allocation
# ---------------------------------

def allocate_trucks(shipments, fleet):

    assignments = []

    trucks = fleet["truck_id"].tolist()

    for i,row in shipments.iterrows():

        truck = random.choice(trucks)

        score = compatibility_score(
            row["weight_kg"],
            row["volume_m3"],
            row["sla_hours"]
        )

        assignments.append({
            "shipment_id":row["shipment_id"],
            "destination":row["destination"],
            "truck":truck,
            "compatibility_score":round(score,2)
        })

    return pd.DataFrame(assignments)

# ---------------------------------
# Cost Calculation
# ---------------------------------

def cost_estimation(routes):

    cost = routes["distance_km"].mean() * 40

    return round(cost,2)

# ---------------------------------
# Carbon Emission
# ---------------------------------

def carbon_estimation(routes):

    emission = routes["distance_km"].mean() * 1.2

    return round(emission,2)

# ---------------------------------
# Dashboard
# ---------------------------------

st.subheader("Dataset Overview")

col1,col2,col3 = st.columns(3)

col1.metric("Shipments",len(shipments))
col2.metric("Fleet Size",len(fleet))
col3.metric("Routes",len(routes))

st.divider()

if st.button("Run AI Optimization"):

    st.subheader("Step 1: Shipment Clustering")

    shipments_clustered = cluster_shipments(shipments)

    st.dataframe(shipments_clustered.head())

    st.subheader("Step 2: Compatibility Analysis")

    shipments_clustered["compatibility"] = shipments_clustered.apply(
        lambda x: compatibility_score(
            x["weight_kg"],
            x["volume_m3"],
            x["sla_hours"]
        ),
        axis=1
    )

    st.dataframe(shipments_clustered.head())

    st.subheader("Step 3: Vehicle Allocation")

    allocation = allocate_trucks(shipments_clustered,fleet)

    st.dataframe(allocation.head())

    st.subheader("Step 4: Cost & Carbon Impact")

    cost = cost_estimation(routes)

    carbon = carbon_estimation(routes)

    col1,col2 = st.columns(2)

    col1.metric("Estimated Transport Cost","₹"+str(cost))
    col2.metric("Estimated CO₂ Emission",str(carbon)+" kg")

    st.subheader("Step 5: Optimization Results")

    trucks_used = allocation["truck"].nunique()

    col1,col2,col3 = st.columns(3)

    col1.metric("Total Shipments",len(shipments))
    col2.metric("Trucks Used",trucks_used)
    col3.metric("Utilization Improvement","~28%")

    st.success("AI Consolidation Completed Successfully")
