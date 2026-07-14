import streamlit as st
import sys
import joblib
import json
import numpy as np
import pandas as pd
import ee
import os
import base64
import folium
from streamlit_folium import st_folium  # <--- Clean integration library

# ==============================================================================
# 0. CONFIGURATION SETUP
# ==============================================================================
st.set_page_config(page_title="Malaria Prevalence Prediction", layout="wide")

st.markdown("""
    <style>
    .reportview-container {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eef2f6;
        margin-bottom: 10px;
    }
    .instruction-card {
        background-color: #f1f3f5;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #2b6cb0;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. MODEL LOADING
# ==============================================================================
@st.cache_resource
def load_ml_pipeline():
    return joblib.load("best_rf_reduced_model.joblib")

model_pipeline = load_ml_pipeline()

# ==============================================================================
# 2. EARTH ENGINE AUTHENTICATION SETUP
# ==============================================================================
try:
    b64_credentials = st.secrets["EARTHENGINE_CREDENTIALS_BASE64"]
    decoded_bytes = base64.b64decode(b64_credentials)
    credentials_dict = json.loads(decoded_bytes.decode("utf-8"))
    
    service_account_email = credentials_dict["client_email"]
    
    with open("ee_credentials.json", "w") as f:
        json.dump(credentials_dict, f)
        
    credentials = ee.ServiceAccountCredentials(service_account_email, "ee_credentials.json")
    ee.Initialize(credentials)
    
except Exception as e:
    st.error(f"Earth Engine Authentication Failed. Error details: {e}")

# ==============================================================================
# 3. MAIN INTERFACE HEADER SETUP
# ==============================================================================
st.title("A Web Application for Malaria Prevalence Prediction")

if "map_ready" not in st.session_state:
    st.session_state.map_ready = False
    st.session_state.smoothed_prediction_30m = None
    st.session_state.pixel_data = None
    st.session_state.aoi = None
    st.session_state.target_year = 2026
    st.session_state.target_district = "Karagwe"
    st.session_state.map_raster = None

current_view = st.radio(
    label="Navigation Menu",
    options=["About the Application", "Malaria Prevalence Prediction Workspace"],
    horizontal=True,
    label_visibility="collapsed"
)

st.write("---")

# ==========================================
# View 1: About Panel
# ==========================================
if current_view == "About the Application":
    st.header("About the Application")
    st.write(
        """
        This web application provides an automated platform for predicting the *Plasmodium falciparum* parasite 
        rate for children between 2 and 10 years (**PfPR2-10**) using satellite-derived environmental variables 
        and a trained Random Forest machine learning model.
        """
    )
    
    st.write("### 🗺️ Geographic Domain Workspace Reference: Karagwe")
    about_map = folium.Map(location=[-1.59, 31.05], zoom_start=9, control_scale=True)
    try:
        districts = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
        aoi_karagwe = districts.filter(ee.Filter.eq("ADM2_NAME", "Karagwe"))
        aoi_map_id = ee.Image().paint(aoi_karagwe, 0, 2).getMapId()
        folium.TileLayer(
            tiles=aoi_map_id['tile_fetcher'].url_format, attr='Google Earth Engine',
            name='Karagwe Boundary Layer', overlay=True
        ).add_to(about_map)
    except Exception:
        pass
    
    # Use st_folium for the static map view as well
    st_folium(about_map, height=400, width=800, returned_objects=[])

# ==========================================
# View 2: Prediction Workspace
# ==========================================
elif current_view == "Malaria Prevalence Prediction Workspace":
    st.header("Malaria Prevalence Prediction Workspace")
    
    st.markdown("""
    <div class="instruction-card">
        <h4 style="margin-top:0px; color:#2b6cb0;">📖 How to Use This Module</h4>
        <ol style="margin-bottom:10px;">
            <li><b>Select Target Projection Year:</b> Choose the specific year (between 2020 and 2027) for projection.</li>
            <li><b>Run Predictions:</b> Click <b>"Run Predictions"</b> to generate dynamic Earth Engine layers.</li>
            <li><b>Extract Values:</b> Click <b>any point on the map</b> to instantly retrieve environmental predictors in the right sidebar.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("### 📊 Predictive Model Achieved Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="Mean Absolute Error (MAE)", value="0.0384", delta="Excellent Precision")
    with m_col2:
        st.metric(label="Root Mean Squared Error (RMSE)", value="0.0491", delta="Low Spatial Variance")
    with m_col3:
        st.metric(label="R-squared (R² Coefficient)", value="84.2%", delta="Strong Variance Explained")

    st.write("---")

    def reset_map_state():
        st.session_state.map_ready = False

    available_years = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
    target_year = st.selectbox(
        "Select Target Surveillance / Projection Year", 
        available_years, 
        index=available_years.index(st.session_state.target_year),
        on_change=reset_map_state
    )
    
    st.session_state.target_year = target_year
    st.session_state.target_district = "Karagwe"

    if st.button("Run Predictions", type="primary"):
        current_year = 2026
        spinner_msg = f"Extracting spatial diagnostics from Google Earth Engine for {st.session_state.target_district}..."
            
        with st.spinner(spinner_msg):
            districts = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
            aoi = districts.filter(ee.Filter.eq("ADM2_NAME", st.session_state.target_district))
            aoi_geometry = aoi.geometry()
            
            commonCRS = "EPSG:4326"
            fineScale = 30
            pfprScale = 5000
            commonProjection = ee.Projection(commonCRS).atScale(fineScale)

            base_start = ee.Date.fromYMD(2020, 1, 1)
            base_end = ee.Date.fromYMD(2026, 1, 1)
            years_list = ee.List([2020, 2021, 2022, 2023, 2024, 2025])
            
            def get_annual_rain(y):
                start = ee.Date.fromYMD(y, 1, 1)
                end = ee.Date.fromYMD(ee.Number(y).add(1), 1, 1)
                return ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
                    .filterBounds(aoi_geometry).filterDate(start, end).select("precipitation").sum()

            if st.session_state.target_year == current_year:
                real_start = ee.Date.fromYMD(current_year, 1, 1)
                real_end = ee.Date('2026-07-10')
                
                s2_real = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                    .filterBounds(aoi_geometry).filterDate(real_start, real_end)\
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"])
                lst_real = ee.ImageCollection("MODIS/061/MOD11A1")\
                    .filterBounds(aoi_geometry).filterDate(real_start, real_end).select("LST_Day_1km")
                rain_real = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
                    .filterBounds(aoi_geometry).filterDate(real_start, real_end).select("precipitation").sum()
                
                s2_base = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                    .filterBounds(aoi_geometry).filterDate(base_start, base_end)\
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"])
                lst_base = ee.ImageCollection("MODIS/061/MOD11A1")\
                    .filterBounds(aoi_geometry).filterDate(base_start, base_end).select("LST_Day_1km")
                rain_base = ee.ImageCollection(years_list.map(get_annual_rain)).mean()

                s2 = ee.ImageCollection([s2_real.median(), s2_base.median()]).mean().clip(aoi_geometry)
                lst_raw = ee.ImageCollection([lst_real.mean(), lst_base.mean()]).mean().clip(aoi_geometry)
                rainfall = rain_real.add(rain_base.multiply(0.5)).rename("Rainfall").clip(aoi_geometry)

            elif st.session_state.target_year > current_year:
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
                start_date = ee.Date.fromYMD(st.session_state.target_year, 1, 1)
                end_date = ee.Date.fromYMD(st.session_state.target_year + 1, 1, 1)
                
                s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                    .filterBounds(aoi_geometry).filterDate(start_date, end_date)\
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"])\
                    .median().clip(aoi_geometry)
                lst_raw = ee.ImageCollection("MODIS/061/MOD11A1")\
                    .filterBounds(aoi_geometry).filterDate(start_date, end_date).select("LST_Day_1km")\
                    .mean().clip(aoi_geometry)
                rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
                    .filterBounds(aoi_geometry).filterDate(start_date, end_date).select("precipitation")\
                    .sum().rename("Rainfall").clip(aoi_geometry)

            # Feature Layer Synthesizer
            ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
            ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")
            lst = lst_raw.multiply(0.02).subtract(273.15).rename("LST")
            elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("Elevation").clip(aoi_geometry)
            
            water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(50).clip(aoi_geometry)
            water5km = water.reproject(crs=commonCRS, scale=pfprScale).unmask(0)
            distWater5km = water5km.fastDistanceTransform(256, "pixels", "squared_euclidean").sqrt().multiply(pfprScale).rename("DistWater").clip(aoi_geometry)

            band_names = ["NDWI", "NDMI", "LST", "Rainfall", "Elevation", "DistWater"]
            
            user_map_assets = {
                2020: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2020",
                2021: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2021",
                2022: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2022",
                2023: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2023",
                2024: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2024"
            }
            
            if st.session_state.target_year < 2025:
                map_target_year = min(st.session_state.target_year, 2024)
                selected_asset_path = user_map_assets[map_target_year]
                map_raster = ee.Image(selected_asset_path).select([0]).rename("MAP_PfPR").clip(aoi_geometry)
                raw_stack = ee.Image.cat([ndwi, ndmi, lst, rainfall, elevation, distWater5km, map_raster]).toFloat()
            else:
                map_raster = None
                raw_stack = ee.Image.cat([ndwi, ndmi, lst, rainfall, elevation, distWater5km]).toFloat()

            lonlat_image = ee.Image.pixelLonLat().clip(aoi_geometry)
            data_and_coords = raw_stack.addBands(lonlat_image)
            
            sampling_features = data_and_coords.sample(
                region=aoi_geometry,
                scale=3000, 
                projection=commonCRS,
                geometries=True,
                tileScale=4
            ).getInfo()

            features_list = []
            for feat in sampling_features["features"]:
                props = feat["properties"]
                if "longitude" in props and "latitude" in props:
                    features_list.append(props)
            
            pixel_data = pd.DataFrame(features_list) if features_list else pd.DataFrame()

            if not pixel_data.empty:
                pixel_data = pixel_data.dropna(subset=band_names)
                X_pixels = pixel_data[band_names]
                pixel_data["predicted_PfPR"] = model_pipeline.predict(X_pixels)
                
                if "MAP_PfPR" in pixel_data.columns:
                    if pixel_data["MAP_PfPR"].max() <= 1.0:
                        pixel_data["MAP_PfPR"] = pixel_data["MAP_PfPR"] * 100.0
                
                ee_features = []
                for _, row in pixel_data.iterrows():
                    geom = ee.Geometry.Point([row["longitude"], row["latitude"]])
                    feat = ee.Feature(geom, {"predicted_PfPR": float(row["predicted_PfPR"])})
                    ee_features.append(feat)
                
                predicted_gee_layer = ee.FeatureCollection(ee_features)
                grid_projection = ee.Projection(commonCRS).atScale(pfprScale)
                
                prediction_raster_5km = predicted_gee_layer.reduceToImage(
                    properties=["predicted_PfPR"], reducer=ee.Reducer.mean()
                ).reproject(crs=grid_projection)
                
                smoothed_prediction_30m = prediction_raster_5km.resample('bilinear')\
                    .reproject(crs=commonProjection)\
                    .clip(aoi_geometry)
                
                st.session_state.pixel_data = pixel_data
                st.session_state.smoothed_prediction_30m = smoothed_prediction_30m
                st.session_state.map_raster = map_raster
                st.session_state.aoi = aoi
                st.session_state.map_ready = True

    # ==========================================
    # Rendering Interface Split Columns
    # ==========================================
    if st.session_state.map_ready:
        st.write("---")
        
        col_map, col_profile = st.columns([2, 1])

        with col_map:
            st.subheader("🗺️ Spatial Prediction Canvas")
            
            map_center = [-1.59, 31.05]
            map_zoom = 9
            f_map = folium.Map(location=map_center, zoom_start=map_zoom, control_scale=True)
            
            aoi_map_id = ee.Image().paint(st.session_state.aoi, 0, 2).getMapId()
            folium.TileLayer(
                tiles=aoi_map_id['tile_fetcher'].url_format, attr='Google Earth Engine',
                name=f'{st.session_state.target_district} Border', overlay=True
            ).add_to(f_map)
            
            min_val = float(st.session_state.pixel_data["predicted_PfPR"].min())
            max_val = float(st.session_state.pixel_data["predicted_PfPR"].max())
            
            high_contrast_palette = ['#3288bd', '#99d594', '#e6f598', '#fee08b', '#fc8d59', '#d53e4f']
            vis_params = {'min': min_val, 'max': max_val, 'palette': high_contrast_palette}
            
            prediction_map_id = st.session_state.smoothed_prediction_30m.getMapId(vis_params)
            folium.TileLayer(
                tiles=prediction_map_id['tile_fetcher'].url_format, attr='Google Earth Engine',
                name=f'Predicted PfPR ({st.session_state.target_year})', overlay=True, opacity=0.85
            ).add_to(f_map)
            
            if st.session_state.map_raster is not None:
                map_layer_id = st.session_state.map_raster.getMapId(vis_params)
                folium.TileLayer(
                    tiles=map_layer_id['tile_fetcher'].url_format, attr='Malaria Atlas Project User Asset',
                    name=f'MAP Asset Baseline ({min(st.session_state.target_year, 2024)})', overlay=True, opacity=0.65
                ).add_to(f_map)
            
            folium.LayerControl().add_to(f_map)
            
            # --- ST_FOLIUM INTERFACE BINDING ---
            # Using st_folium triggers native Python state return on-click instantly
            map_data = st_folium(f_map, height=600, width=700, key="prediction_map")

        with col_profile:
            st.subheader("🎯 Real-Time Query Console")
            
            # Extract point coordinate from map_data click event natively
            clicked_coords = map_data.get("last_clicked") if map_data else None
            
            if clicked_coords:
                click_lat = clicked_coords["lat"]
                click_lon = clicked_coords["lng"]
                
                # Search local Pandas spatial grid dataset for the nearest pixel match
                spatial_grid_dataset = st.session_state.pixel_data
                distances = np.sqrt(
                    (spatial_grid_dataset["latitude"] - click_lat)**2 + 
                    (spatial_grid_dataset["longitude"] - click_lon)**2
                )
                nearest_idx = distances.idxmin()
                minimal_distance = distances.min()
                
                # Check bounding proximity threshold (~6.6km bounding box filter)
                if minimal_distance < 0.06:
                    pt = spatial_grid_dataset.iloc[nearest_idx]
                    
                    # Coordinate Card
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="margin:0; font-size: 0.8rem; color:#666; font-weight: bold; text-transform: uppercase;">Selected Coordinate</p>
                        <h3 style="margin:5px 0 0 0; color:#2b6cb0;">Lat: {pt['latitude']:.4f}, Lon: {pt['longitude']:.4f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Model Predicted PfPR Card
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid #fc8d59;">
                        <p style="margin:0; font-size: 0.85rem; color:#666; font-weight: bold;">MODEL PREDICTED PfPR2-10</p>
                        <h2 style="margin:5px 0 0 0; color:#d53e4f; font-size:2rem;">{pt['predicted_PfPR']:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Baseline Card
                    if "MAP_PfPR" in pt and not pd.isna(pt["MAP_PfPR"]):
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 4px solid #3288bd;">
                            <p style="margin:0; font-size: 0.85rem; color:#666; font-weight: bold;">MALARIA ATLAS PROJECT BASELINE</p>
                            <h2 style="margin:5px 0 0 0; color:#3288bd; font-size:2rem;">{pt['MAP_PfPR']:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="metric-card" style="background-color: #fafafa; border-left: 4px dashed #ccc;">
                            <p style="margin:0; font-size: 0.85rem; color:#777; font-weight: bold;">MALARIA ATLAS PROJECT BASELINE</p>
                            <h4 style="margin:5px 0 0 0; color:#777;">N/A (Projection Period)</h4>
                            <span style="font-size:0.75rem; color:#888;">MAP assets do not extend past 2024.</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Environmental Telemetry Grid
                    st.write("#### 🌿 Localized Environmental Telemetry")
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        st.metric("LST Temp", f"{pt['LST']:.2f} °C")
                        st.metric("Elevation", f"{pt['Elevation']:.1f} m")
                        st.metric("NDWI", f"{pt['NDWI']:.4f}")
                    with v_col2:
                        st.metric("Rainfall", f"{pt['Rainfall']:.2f} mm")
                        st.metric("Dist to Water", f"{pt['DistWater']:.1f} m")
                        st.metric("NDMI", f"{pt['NDMI']:.4f}")
                else:
                    st.warning("⚠️ The click was registered outside of the target bounding dataset region.")
            else:
                st.info("ℹ️ Click any point inside the map boundary to instantly extract and profile variables.")

        # 💾 Export Spatial Products Code block
        st.write("---")
        st.write("### 💾 Export Spatial Products:")
        try:
            raw_download_url = st.session_state.smoothed_prediction_30m.getDownloadURL({
                'name': f'PfPR_Output_{st.session_state.target_district}_{st.session_state.target_year}',
                'scale': 5000, 
                'crs': 'EPSG:4326', 
                'filePerBand': False
            })
            st.markdown(f"[📥 Download Native 5km Model Raster (.tiff)]({raw_download_url})")
        except Exception:
            st.info("Download link generation timed out on remote GEE servers.")