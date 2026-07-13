import streamlit as st
import ee
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import json
from google.oauth2 import service_account

# 1. Page Configuration
st.set_page_config(
    page_title="Mosquito Breeding & Malaria Surveillance Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Earth Engine Authentication and Initialization
@st.cache_resource
def initialize_ee():
    try:
        # Check for Streamlit Secrets
        if "GEE_SECRET_KEY" in st.secrets:
            secret_dict = json.loads(st.secrets["GEE_SECRET_KEY"])
            credentials = service_account.Credentials.from_service_account_info(secret_dict)
            ee.Initialize(credentials=credentials)
        else:
            # Fallback to local user credentials
            ee.Initialize()
        return True
    except Exception as e:
        st.error(f"Failed to initialize Earth Engine: {e}")
        return False

ee_initialized = initialize_ee()

# 3. Session State Initialization
if "target_district" not in st.session_state:
    st.session_state.target_district = "Karagwe"
if "target_year" not in st.session_state:
    st.session_state.target_year = 2024
if "prediction_triggered" not in st.session_state:
    st.session_state.prediction_triggered = False
if "pixel_data" not in st.session_state:
    st.session_state.pixel_data = None
if "smoothed_prediction_30m" not in st.session_state:
    st.session_state.smoothed_prediction_30m = None
if "map_raster" not in st.session_state:
    st.session_state.map_raster = None
if "aoi" not in st.session_state:
    st.session_state.aoi = None

# Load Region Boundaries
districts = ee.FeatureCollection("FAO/GAUL/2015/level2")
st.session_state.aoi = districts.filter(ee.Filter.eq("ADM2_NAME", st.session_state.target_district))
aoi_geometry = st.session_state.aoi.geometry()

# 4. Sidebar Controls
st.sidebar.title("🧬 Control Panel")
st.sidebar.markdown("Configure environmental parameters and baseline datasets.")

st.session_state.target_district = st.sidebar.selectbox(
    "Target District",
    ["Karagwe", "Kyerwa", "Bukoba", "Misenyi"],
    index=0
)

st.session_state.target_year = st.sidebar.slider(
    "Prediction Year Target",
    min_value=2020,
    max_value=2030,
    value=st.session_state.target_year,
    step=1
)

# 5. Main Application Interface
st.title("🛰️ Mosquito Breeding Sites & Malaria Surveillance Tool")
st.markdown("### Predictive Risk Mapping using Satellite Imagery & Climate Predictors")

tab1, tab2 = st.tabs(["🔮 Workspace & Interactive Risk Map", "ℹ️ About the Application"])

with tab1:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("#### Model Engine")
        st.info(f"📍 **Region:** {st.session_state.target_district} District\n\n📅 **Target Year:** {st.session_state.target_year}")
        
        run_button = st.button("🚀 Run Predictive Model", use_container_width=True)
        
        if run_button:
            with st.spinner("Executing Random Forest Engine & Interpolating Stacks..."):
                current_year = 2024
                base_start = f"{min(st.session_state.target_year, current_year)}-01-01"
                base_end = f"{min(st.session_state.target_year, current_year)}-12-31"
                
                # Fetch baseline Malaria Prevalence Asset (MAP)
                raw_map_raster = ee.ImageCollection("projects/sat-images-atlas/assets/Tanzania_PfPR")\
                    .filterDate(base_start, base_end).median().select("MAP_PfPR").clip(aoi_geometry)
                
                # Safely pull max value server-side down to native Python float
                map_max_val = float(raw_map_raster.reduceRegion(
                    reducer=ee.Reducer.max(), 
                    geometry=aoi_geometry, 
                    scale=5000, 
                    tileScale=4
                ).get("MAP_PfPR").getInfo() or 0)
                
                # Normalize values if expressed as decimals (0.0 - 1.0) instead of percentages
                if map_max_val <= 1.0:
                    st.session_state.map_raster = raw_map_raster.multiply(100.0)
                else:
                    st.session_state.map_raster = raw_map_raster
                
                # Environmental Feature Engineering Stack
                years_list = ee.List([st.session_state.target_year])
                
                def get_annual_rain(y):
                    start = ee.Date.fromYMD(y, 1, 1)
                    end = ee.Date.fromYMD(ee.Number(y).add(1), 1, 1)
                    return ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
                        .filterBounds(aoi_geometry).filterDate(start, end).select("precipitation").sum()
                
                if st.session_state.target_year <= current_year:
                    s2Collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                        .filterBounds(aoi_geometry).filterDate(base_start, base_end)\
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"])
                    s2 = s2Collection.median().clip(aoi_geometry)
                    
                    lst_collection = ee.ImageCollection("MODIS/061/MOD11A1")\
                        .filterBounds(aoi_geometry).filterDate(base_start, base_end).select("LST_Day_1km")
                    lst_raw = lst_collection.mean().clip(aoi_geometry)
                    
                    annual_rain_collection = ee.ImageCollection(years_list.map(get_annual_rain))
                    rainfall = annual_rain_collection.mean().rename("Rainfall").clip(aoi_geometry)
                else:
                    # Scenario Planning (Future Forecasting Multipliers)
                    s2Collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                        .filterBounds(aoi_geometry).filterDate("2024-01-01", "2024-12-31")\
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"])
                    s2 = s2Collection.median().clip(aoi_geometry)
                    
                    lst_collection = ee.ImageCollection("MODIS/061/MOD11A1")\
                        .filterBounds(aoi_geometry).filterDate("2024-01-01", "2024-12-31").select("LST_Day_1km")
                    lst_raw = lst_collection.mean().clip(aoi_geometry).add(0.45 * (st.session_state.target_year - 2024))
                    
                    annual_rain_collection = ee.ImageCollection(years_list.map(get_annual_rain))
                    rainfall = annual_rain_collection.mean().rename("Rainfall").clip(aoi_geometry).multiply(1.03)

                # Remote Sensing Indices
                mndwi = s2.normalizedDifference(["B3", "B11"]).rename("MNDWI")
                ndvi = s2.normalizedDifference(["B8", "B3"]).rename("NDVI")
                lst = lst_raw.multiply(0.02).subtract(273.15).rename("LST")
                
                # Predictor Layer Alignment
                predictors = ee.Image.cat([mndwi, ndvi, lst, rainfall, st.session_state.map_raster.rename("MAP_Baseline")])
                
                # Synthetic Training Sample Engine (Stratified Random Sampling)
                training_points = predictors.sample(
                    region=aoi_geometry,
                    scale=500,
                    numPixels=150,
                    seed=42,
                    geometries=True
                )
                
                # Train Random Forest Regressor Model
                trained_rf_regressor = ee.Classifier.smileRandomForest(numberOfTrees=75)\
                    .setOutputMode("REGRESSION")\
                    .train(features=training_points, classProperty="MAP_Baseline", inputProperties=["MNDWI", "NDVI", "LST", "Rainfall"])
                
                # Predict and Interpolate to 30m Downsampled Resolution
                raw_prediction = predictors.classify(trained_rf_regressor).rename("predicted_PfPR")
                st.session_state.smoothed_prediction_30m = raw_prediction.resample("bilinear")\
                    .focalMean(radius=45, shape="circle", units="meters")\
                    .clip(aoi_geometry)
                
                # Sample Pixel Array Data for UI Tables
                sampled_features = st.session_state.smoothed_prediction_30m.sample(
                    region=aoi_geometry, scale=1000, numPixels=40, seed=42
                ).getInfo()
                
                records = [f["properties"] for f in sampled_features["features"] if "predicted_PfPR" in f["properties"]]
                st.session_state.pixel_data = pd.DataFrame(records)
                st.session_state.prediction_triggered = True
                st.success("Model runs complete! Risk matrices initialized.")

    with col2:
        # Generate clean interactive Folium canvas map instance
        f_map = folium.Map(location=[-1.59, 31.05], zoom_start=9, control_scale=True)
        
        # Add AOI Boundary Vector Layer Safely
        if st.session_state.aoi:
            aoi_map_id = ee.Image().paint(st.session_state.aoi, 0, 2).getMapId()
            folium.TileLayer(
                tiles=str(aoi_map_id['tile_fetcher'].url_format), 
                attr='Google Earth Engine',
                name=f'{st.session_state.target_district} District Boundary', 
                overlay=True
            ).add_to(f_map)
        
        # Add Raster Layers conditionally based on model state
        if st.session_state.prediction_triggered and st.session_state.smoothed_prediction_30m is not None:
            # Force values into strict native floats to eliminate numpy/pandas wrapper classes
            min_val = float(st.session_state.pixel_data["predicted_PfPR"].min())
            max_val = float(st.session_state.pixel_data["predicted_PfPR"].max())
            
            high_contrast_palette = ['#3288bd', '#99d594', '#e6f598', '#fee08b', '#fc8d59', '#d53e4f']
            vis_params = {'min': min_val, 'max': max_val, 'palette': high_contrast_palette}
            
            # Extract and explicitly convert tile URLs to standard string types
            prediction_map_id = st.session_state.smoothed_prediction_30m.getMapId(vis_params)
            folium.TileLayer(
                tiles=str(prediction_map_id['tile_fetcher'].url_format), 
                attr='Google Earth Engine',
                name=f'Predicted PfPR ({st.session_state.target_year})', 
                overlay=True, 
                opacity=0.85
            ).add_to(f_map)
            
            if st.session_state.map_raster is not None:
                map_layer_id = st.session_state.map_raster.getMapId(vis_params)
                folium.TileLayer(
                    tiles=str(map_layer_id['tile_fetcher'].url_format), 
                    attr='Malaria Atlas Project User Asset',
                    name=f'MAP Asset Baseline ({min(st.session_state.target_year, 2024)})', 
                    overlay=True, 
                    opacity=0.65
                ).add_to(f_map)
            
            # Dynamic UI Legend Layout
            css_gradient = ", ".join(high_contrast_palette)
            v_min, v_max = f"{min_val:.2f}%", f"{max_val:.2f}%"
            st.markdown(
                f"""
                <div style="border:1px solid #ddd; background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                    <p style="margin: 0 0 5px 0; font-weight: bold; text-align: center;">Malaria Prevalence Legend (PfPR2-10)</p>
                    <div style="background: linear-gradient(to right, {css_gradient}); width: 100%; height: 15px; border-radius: 3px;"></div>
                    <div style="display: flex; justify-content: space-between; font-weight: bold; font-size: 12px; margin-top: 3px;">
                        <span>Low ({v_min})</span>
                        <span>High ({v_max})</span>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

        # Layer toggles
        folium.LayerControl().add_to(f_map)
        
        # Render map canvas safely using isolated component keying strings
        dynamic_map_key = f"interactive_prediction_map_canvas_yr_{st.session_state.target_year}"
        map_output = st_folium(f_map, height=600, width=None, key=dynamic_map_key)
        
        # Inspector Pixel Matrix Engine
        st.markdown("#### 🔍 Point Inspector Matrix")
        clicked_coords = map_output.get("last_clicked")
        
        if clicked_coords and st.session_state.prediction_triggered and st.session_state.smoothed_prediction_30m is not None:
            lat, lon = clicked_coords["lat"], clicked_coords["lng"]
            inspect_point = ee.Geometry.Point([lon, lat])
            
            pixel_val = st.session_state.smoothed_prediction_30m.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=inspect_point,
                scale=30
            ).get("predicted_PfPR").getInfo()
            
            if pixel_val is not None:
                st.metric(label=f"Inspected Site Risk (Lat: {lat:.4f}, Lon: {lon:.4f})", value=f"{pixel_val:.3f}% Plasmodium falciparum Prevalence")
            else:
                st.warning("Selected point lies outside target raster spatial region boundaries.")
        else:
            st.info("Click anywhere inside the generated risk layer to read localized prevalence percentages.")

with tab2:
    st.markdown("### About the Application Architecture")
    st.write("This tool runs Random Forest Regressors downstream against harmonized Sentinel-2 and MODIS composite stacks across the Kagera Region of Tanzania.")
    
    # Render static map placeholder safely in the About section via pure HTML text streams
    about_static_map = folium.Map(location=[-1.59, 31.05], zoom_start=8)
    if st.session_state.aoi:
        boundary_image = ee.Image().paint(st.session_state.aoi, 0, 3)
        boundary_map_id = boundary_image.getMapId({'palette': '#FF0000'})
        
        folium.TileLayer(
            tiles=str(boundary_map_id['tile_fetcher'].url_format),
            attr='Google Earth Engine GAUL',
            name='District Boundary Overview',
            overlay=True,
            opacity=1.0
        ).add_to(about_static_map)
        
        st.components.v1.html(about_static_map._repr_html_(), height=380)