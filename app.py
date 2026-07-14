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
import streamlit.components.v1 as components
from branca.element import Template, MacroElement

# ==============================================================================
# 0. CONFIGURATION SETUP
# ==============================================================================
st.set_page_config(page_title="Malaria Prevalence Prediction", layout="wide")

st.markdown("""
    <style>
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
# 3. MAIN INTERFACE
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
    components.html(about_map._repr_html_(), height=400, scrolling=False)

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
            <li><b>Run Predictions:</b> Click the <b>"Run Predictions"</b> button to fetch the dynamic Earth Engine products.</li>
            <li><b>Interact with the Map:</b> Use the layers panel, pan, or zoom into the high-resolution layers.</li>
            <li><b>Extract Values:</b> Click <b>any point on the map</b> to instantly retrieve environmental predictors and model calculations via the popup.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("### 📊 Predictive Model Achieved Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1: st.metric(label="MAE", value="0.0384")
    with m_col2: st.metric(label="RMSE", value="0.0491")
    with m_col3: st.metric(label="R-squared", value="84.2%")

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
        with st.spinner(f"Extracting spatial diagnostics for {st.session_state.target_district}..."):
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
                return ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi_geometry).filterDate(start, end).select("precipitation").sum()

            if st.session_state.target_year == current_year:
                real_start = ee.Date.fromYMD(current_year, 1, 1)
                real_end = ee.Date('2026-07-10')
                s2_real = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi_geometry).filterDate(real_start, real_end).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"])
                lst_real = ee.ImageCollection("MODIS/061/MOD11A1").filterBounds(aoi_geometry).filterDate(real_start, real_end).select("LST_Day_1km")
                rain_real = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi_geometry).filterDate(real_start, real_end).select("precipitation").sum()
                s2_base = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi_geometry).filterDate(base_start, base_end).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"])
                lst_base = ee.ImageCollection("MODIS/061/MOD11A1").filterBounds(aoi_geometry).filterDate(base_start, base_end).select("LST_Day_1km")
                rain_base = ee.ImageCollection(years_list.map(get_annual_rain)).mean()
                s2 = ee.ImageCollection([s2_real.median(), s2_base.median()]).mean().clip(aoi_geometry)
                lst_raw = ee.ImageCollection([lst_real.mean(), lst_base.mean()]).mean().clip(aoi_geometry)
                rainfall = rain_real.add(rain_base.multiply(0.5)).rename("Rainfall").clip(aoi_geometry)

            elif st.session_state.target_year > current_year:
                s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi_geometry).filterDate(base_start, base_end).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"]).median().clip(aoi_geometry)
                lst_raw = ee.ImageCollection("MODIS/061/MOD11A1").filterBounds(aoi_geometry).filterDate(base_start, base_end).select("LST_Day_1km").mean().clip(aoi_geometry)
                rainfall = ee.ImageCollection(years_list.map(get_annual_rain)).mean().rename("Rainfall").clip(aoi_geometry)
                
            else:
                start_date = ee.Date.fromYMD(st.session_state.target_year, 1, 1)
                end_date = ee.Date.fromYMD(st.session_state.target_year + 1, 1, 1)
                s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi_geometry).filterDate(start_date, end_date).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B3", "B8", "B11"]).median().clip(aoi_geometry)
                lst_raw = ee.ImageCollection("MODIS/061/MOD11A1").filterBounds(aoi_geometry).filterDate(start_date, end_date).select("LST_Day_1km").mean().clip(aoi_geometry)
                rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi_geometry).filterDate(start_date, end_date).select("precipitation").sum().rename("Rainfall").clip(aoi_geometry)

            ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
            ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")
            lst = lst_raw.multiply(0.02).subtract(273.15).rename("LST")
            elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("Elevation").clip(aoi_geometry)
            water5km = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(50).reproject(crs=commonCRS, scale=pfprScale).unmask(0).clip(aoi_geometry)
            distWater5km = water5km.fastDistanceTransform(256, "pixels", "squared_euclidean").sqrt().multiply(pfprScale).rename("DistWater").clip(aoi_geometry)

            band_names = ["NDWI", "NDMI", "LST", "Rainfall", "Elevation", "DistWater"]
            
            user_map_assets = {2020: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2020", 2021: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2021", 2022: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2022", 2023: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2023", 2024: "projects/ee-joanithakaijage/assets/MAP_PfPR2_10_2024"}
            
            if st.session_state.target_year < 2025:
                map_raster = ee.Image(user_map_assets[min(st.session_state.target_year, 2024)]).select([0]).rename("MAP_PfPR").clip(aoi_geometry)
                raw_stack = ee.Image.cat([ndwi, ndmi, lst, rainfall, elevation, distWater5km, map_raster]).toFloat()
            else:
                map_raster = None
                raw_stack = ee.Image.cat([ndwi, ndmi, lst, rainfall, elevation, distWater5km]).toFloat()

            lonlat_image = ee.Image.pixelLonLat().clip(aoi_geometry)
            data_and_coords = raw_stack.addBands(lonlat_image)
            sampling_features = data_and_coords.sample(region=aoi_geometry, scale=3000, projection=commonCRS, geometries=True, tileScale=4).getInfo()

            features_list = [f["properties"] for f in sampling_features["features"] if "longitude" in f["properties"]]
            pixel_data = pd.DataFrame(features_list)
            
            if not pixel_data.empty:
                pixel_data = pixel_data.dropna(subset=band_names)
                X_pixels = pixel_data[band_names]
                pixel_data["predicted_PfPR"] = model_pipeline.predict(X_pixels)
                
                if "MAP_PfPR" in pixel_data.columns and pixel_data["MAP_PfPR"].max() <= 1.0:
                    pixel_data["MAP_PfPR"] = pixel_data["MAP_PfPR"] * 100.0
                
                ee_features = [ee.Feature(ee.Geometry.Point([r["longitude"], r["latitude"]]), {"predicted_PfPR": float(r["predicted_PfPR"])}) for _, r in pixel_data.iterrows()]
                prediction_raster_5km = ee.FeatureCollection(ee_features).reduceToImage(properties=["predicted_PfPR"], reducer=ee.Reducer.mean()).reproject(crs=ee.Projection(commonCRS).atScale(pfprScale))
                smoothed_prediction_30m = prediction_raster_5km.resample('bilinear').reproject(crs=commonProjection).clip(aoi_geometry)
                
                st.session_state.pixel_data = pixel_data
                st.session_state.smoothed_prediction_30m = smoothed_prediction_30m
                st.session_state.map_raster = map_raster
                st.session_state.aoi = aoi
                st.session_state.map_ready = True

    if st.session_state.map_ready:
        st.write("---")
        st.subheader("🗺️ Spatial Prediction Canvas")
        
        f_map = folium.Map(location=[-1.59, 31.05], zoom_start=9, control_scale=True)
        aoi_map_id = ee.Image().paint(st.session_state.aoi, 0, 2).getMapId()
        folium.TileLayer(tiles=aoi_map_id['tile_fetcher'].url_format, attr='Google Earth Engine', name='Karagwe Border', overlay=True).add_to(f_map)
        
        min_val, max_val = float(st.session_state.pixel_data["predicted_PfPR"].min()), float(st.session_state.pixel_data["predicted_PfPR"].max())
        vis_params = {'min': min_val, 'max': max_val, 'palette': ['#3288bd', '#99d594', '#e6f598', '#fee08b', '#fc8d59', '#d53e4f']}
        
        folium.TileLayer(tiles=st.session_state.smoothed_prediction_30m.getMapId(vis_params)['tile_fetcher'].url_format, attr='Google Earth Engine', name=f'Predicted PfPR ({st.session_state.target_year})', overlay=True).add_to(f_map)
        
        if st.session_state.map_raster is not None:
            folium.TileLayer(tiles=st.session_state.map_raster.getMapId(vis_params)['tile_fetcher'].url_format, attr='MAP Asset', name='MAP Baseline', overlay=True, opacity=0.6).add_to(f_map)
            
        json_data_payload = st.session_state.pixel_data.to_json(orient="records")
        click_macro_template = f"""
        {{% macro html(this, kwargs) %}}
        <script>
            document.addEventListener("DOMContentLoaded", function() {{
                setTimeout(function() {{
                    var map_instance = Object.values(window).find(val => val instanceof L.Map);
                    if (!map_instance) return;
                    var dataset = {json_data_payload};
                    map_instance.on('click', function(e) {{
                        var click_lat = e.latlng.lat, click_lon = e.latlng.lng;
                        var nearest = null, min_d = Infinity;
                        dataset.forEach(p => {{
                            var d = Math.sqrt(Math.pow(p.latitude - click_lat, 2) + Math.pow(p.longitude - click_lon, 2));
                            if (d < min_d) {{ min_d = d; nearest = p; }}
                        }});
                        if (nearest && min_d < 0.06) {{
                            var baseline = nearest.MAP_PfPR ? nearest.MAP_PfPR.toFixed(2) + "%" : "N/A";
                            L.popup().setLatLng(e.latlng).setContent(`
                                <div style="font-family:sans-serif; font-size:12px; width:200px;">
                                    <b>Predicted PfPR:</b> ${{nearest.predicted_PfPR.toFixed(2)}}%<br/>
                                    <b>MAP Baseline:</b> ${{baseline}}<br/>
                                    <b>LST:</b> ${{nearest.LST.toFixed(1)}}°C<br/>
                                    <b>Rainfall:</b> ${{nearest.Rainfall.toFixed(1)}}mm<br/>
                                    <b>Elevation:</b> ${{nearest.Elevation.toFixed(0)}}m
                                </div>`).openOn(map_instance);
                        }}
                    }});
                }}, 1000);
            }});
        </script>
        {{% endmacro %}}
        """
        macro = MacroElement()
        macro._template = Template(click_macro_template)
        f_map.add_child(macro)
        
        folium.LayerControl().add_to(f_map)
        components.html(f_map._repr_html_(), height=600, scrolling=True)