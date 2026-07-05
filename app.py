import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ee
import geemap
import folium
from branca.element import Template, MacroElement
import streamlit.components.v1 as components

# ==========================================
# 1. Initialize Earth Engine Authentication
# ==========================================
@st.cache_resource
def ee_authenticate():
    try:
        # Attempt standard execution initialization
        ee.Initialize()
    except Exception:
        # Fallback to Streamlit Secret cloud token handling
        if "EARTHENGINE_TOKEN" in st.secrets:
            import os
            token_dir = os.path.expanduser("~/.config/earthengine")
            os.makedirs(token_dir, exist_ok=True)
            with open(os.path.join(token_dir, "credentials"), "w") as f:
                f.write(st.secrets["EARTHENGINE_TOKEN"])
            ee.Initialize()
        else:
            st.error("Earth Engine credentials missing. Please check your Streamlit Secrets configuration profile.")
            st.stop()

ee_authenticate()

# ==========================================
# 2. Load the Pre-Trained Random Forest Model
# ==========================================
@st.cache_resource
def load_model():
    # Looks for your model pipeline payload inside your repository directory
    return joblib.load("best_rf_reduced_model.joblib")

try:
    model_pipeline = load_model()
except Exception as e:
    st.error(f"Failed to load the predictive model file structure: {str(e)}")
    st.stop()

# ==========================================
# 3. User Interface Layout & Session State
# ==========================================
st.title("Automated GEE Malaria Surveillance Platform")
st.write("Extracting automated environmental indices for spatial risk mapping in Tanzania.")

# Initialize persistent storage keys so the UI components don't reset on browser redraws
if "map_ready" not in st.session_state:
    st.session_state.map_ready = False
    st.session_state.smoothed_prediction_30m = None
    st.session_state.pixel_data = None
    st.session_state.aoi = None
    st.session_state.target_year = None
    st.session_state.target_district = None

# User selection controls
target_year = st.selectbox("Select Target Surveillance Year", [2020, 2021, 2022, 2023, 2024, 2025])
target_district = st.selectbox("Select Target District Boundary View", ["Karagwe", "Kyerwa"])

if st.button("Generate 30m Visual Risk Map Profile"):
    with st.spinner(f"Extracting remote sensing data and evaluating model layers for {target_district} ({target_year})..."):
        
        try:
            # ------------------------------------------
            # 4. Define Geographic Spatial Boundaries
            # ------------------------------------------
            districts = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
            
            # Dynamically pull the boundary matching the user's dropdown choice
            aoi = districts.filter(ee.Filter.eq("ADM2_NAME", target_district))
            aoi_geometry = aoi.geometry()
            
            start_date = ee.Date.fromYMD(target_year, 1, 1)
            end_date = ee.Date.fromYMD(target_year + 1, 1, 1)
            
            commonCRS = "EPSG:4326"
            fineScale = 30
            pfprScale = 5000
            commonProjection = ee.Projection(commonCRS).atScale(fineScale)

            # ------------------------------------------
            # 5. Extract Predictor Variables from GEE
            # ------------------------------------------
            s2Collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                .filterBounds(aoi_geometry)\
                .filterDate(start_date, end_date)\
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))\
                .select(["B3", "B8", "B11"])
            
            s2 = s2Collection.median().clip(aoi_geometry)
            
            ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
            ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")
            
            lst = ee.ImageCollection("MODIS/061/MOD11A1")\
                .filterBounds(aoi_geometry)\
                .filterDate(start_date, end_date)\
                .select("LST_Day_1km")\
                .mean()\
                .multiply(0.02).subtract(273.15).rename("LST").clip(aoi_geometry)
                
            rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
                .filterBounds(aoi_geometry)\
                .filterDate(start_date, end_date)\
                .select("precipitation")\
                .sum().rename("Rainfall").clip(aoi_geometry)
                
            elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("Elevation").clip(aoi_geometry)
            
            water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(50).clip(aoi_geometry)
            water5km = water.reproject(crs=commonCRS, scale=pfprScale).unmask(0)
            distWater5km = water5km.fastDistanceTransform(256, "pixels", "squared_euclidean").sqrt().multiply(pfprScale).rename("DistWater").clip(aoi_geometry)

            # ------------------------------------------
            # 6. Create clean 5km Grid Points for Sampling
            # ------------------------------------------
            bounds = aoi_geometry.bounds().getInfo()['coordinates'][0]
            lons = [pt[0] for pt in bounds]
            lats = [pt[1] for pt in bounds]
            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)
            
            step = pfprScale / 111000.0
            grid_points = []
            lon_iter = min_lon
            while lon_iter <= max_lon:
                lat_iter = min_lat
                while lat_iter <= max_lat:
                    grid_points.append(ee.Feature(ee.Geometry.Point([lon_iter, lat_iter])))
                    lat_iter += step
                lon_iter += step
                
            raw_grid_collection = ee.FeatureCollection(grid_points)
            grid_samples = raw_grid_collection.filterBounds(aoi)

            # ------------------------------------------
            # 7. Extract Predictor Variables Safely (Point-Based Reduction)
            # ------------------------------------------
            band_names = ["NDWI", "NDMI", "LST", "Rainfall", "Elevation", "DistWater"]
            raw_stack = ee.Image.cat([ndwi, ndmi, lst, rainfall, elevation, distWater5km]).toFloat()

            pixel_samples = raw_stack.reduceRegions(
                collection=grid_samples,
                reducer=ee.Reducer.mean(),
                scale=fineScale,
                crs=commonCRS,
                tileScale=4
            ).getInfo()

            features_list = []
            for feat in pixel_samples["features"]:
                props = feat["properties"]
                if "geometry" in feat and feat["geometry"] is not None:
                    coords = feat["geometry"]["coordinates"]
                    props["longitude"] = coords[0]
                    props["latitude"] = coords[1]
                    features_list.append(props)
            
            if not features_list:
                pixel_data = pd.DataFrame()
            else:
                pixel_data = pd.DataFrame(features_list)

            if not pixel_data.empty:
                valid_cols = band_names + ["longitude", "latitude"]
                pixel_data = pixel_data[[col for col in pixel_data.columns if col in valid_cols]]
                pixel_data = pixel_data.dropna(subset=band_names)
                
                # ------------------------------------------
                # 8. Compute Model Predictions
                # ------------------------------------------
                X_pixels = pixel_data[band_names]
                pixel_data["predicted_PfPR"] = model_pipeline.predict(X_pixels)
                
                # ------------------------------------------
                # 9. Re-upload Predictions to GEE and Rasterize Cleanly
                # ------------------------------------------
                predicted_gee_layer = geemap.pandas_to_ee(pixel_data[["longitude", "latitude", "predicted_PfPR"]], latitude="latitude", longitude="longitude")
                
                # Construct an explicit scale projection for the rasterizer engine
                grid_projection = ee.Projection(commonCRS).atScale(pfprScale)
                
                # Convert points back to images respecting the 5km grid properties
                prediction_raster_5km = predicted_gee_layer.reduceToImage(
                    properties=["predicted_PfPR"], 
                    reducer=ee.Reducer.first()
                ).reproject(crs=grid_projection)
                
                # Resample and bilinear smooth to downscale natively to 30m
                smoothed_prediction_30m = prediction_raster_5km.resample('bilinear').reproject(crs=commonProjection).clip(aoi_geometry)
                
                # Save structured outputs to session state
                st.session_state.pixel_data = pixel_data
                st.session_state.smoothed_prediction_30m = smoothed_prediction_30m
                st.session_state.aoi = aoi
                st.session_state.target_year = target_year
                st.session_state.target_district = target_district
                st.session_state.map_ready = True
            else:
                st.error("No valid pixel coordinates could be extracted from GEE bounds. Check the spatial resolution parameters.")
                st.session_state.map_ready = False

        except Exception as err:
            st.error(f"An error occurred within the execution pipeline: {str(err)}")
            st.session_state.map_ready = False

# ==========================================
# 11. Persistent Map Rendering Render Loop
# ==========================================
if st.session_state.map_ready:
    st.success(f"Successfully processed {st.session_state.target_district} District for {st.session_state.target_year}!")
    st.write("### Interactive 30m Smoothed Risk Map:")
    
    # Dynamically shift map camera coordinate centers depending on chosen district dropdown
    if st.session_state.target_district == "Kyerwa":
        map_center = [-1.30, 30.85]
        map_zoom = 10
    else:
        map_center = [-1.59, 31.21]
        map_zoom = 9

    # Initialize raw Folium map canvas
    f_map = folium.Map(location=map_center, zoom_start=map_zoom, control_scale=True)
    
    # Extract Vector boundary tiles
    aoi_map_id = ee.Image().paint(st.session_state.aoi, 0, 2).getMapId()
    folium.TileLayer(
        tiles=aoi_map_id['tile_fetcher'].url_format,
        attr='Google Earth Engine Boundary Layer',
        name=f'{st.session_state.target_district} Border',
        overlay=True,
        control=True
    ).add_to(f_map)
    
    # Define high-contrast palette gradient variables
    min_val = float(st.session_state.pixel_data["predicted_PfPR"].min())
    max_val = float(st.session_state.pixel_data["predicted_PfPR"].max())
    
    high_contrast_palette = ['#3288bd', '#99d594', '#e6f598', '#fee08b', '#fc8d59', '#d53e4f']
    
    vis_params = {
        'min': min_val,
        'max': max_val,
        'palette': high_contrast_palette
    }
    
    # Extract prediction model layer tiles from Earth Engine
    prediction_map_id = st.session_state.smoothed_prediction_30m.getMapId(vis_params)
    folium.TileLayer(
        tiles=prediction_map_id['tile_fetcher'].url_format,
        attr='Google Earth Engine Model Prediction',
        name=f'Predicted PfPR ({st.session_state.target_year})',
        overlay=True,
        control=True,
        opacity=0.85
    ).add_to(f_map)
    
    # Construct Continuous SHAP-Style colorbar strings and labels
    v_min = f"{min_val:.1f}%"
    v_max = f"{max_val:.1f}%"
    css_gradient = ", ".join(high_contrast_palette)

    legend_template = f"""
    {{% macro html(this, kwargs) %}}
    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px solid #bbb; background-color:rgba(255, 255, 255, 0.95);
        border-radius:8px; padding: 12px 15px; font-size:13px; right: 20px; bottom: 30px; width: 280px;
        font-family: "Source Sans Pro", sans-serif; box-shadow: 0 0 15px rgba(0,0,0,0.2);'>
      
      <div class='legend-title' style='font-weight: bold; margin-bottom: 8px; text-align: center; color: #333;'>
        Malaria Parasite Rate (PfPR)
      </div>
      
      <div class='gradient-bar' style='
        background: linear-gradient(to right, {css_gradient}); 
        width: 100%; height: 18px; border-radius: 4px; border: 1px solid #777;'>
      </div>
      
      <div class='legend-labels' style='margin-top: 5px; font-weight: 600; color: #444; display: flex; justify-content: space-between;'>
        <span>Low Risk ({v_min})</span>
        <span>High Risk ({v_max})</span>
      </div>
    </div>

    <div id='export-container' style='position: absolute; z-index:9999; top: 10px; left: 50px;'>
      <button onclick="window.print()" style='padding: 6px 12px; background: white; border: 2px solid #ccc; 
        border-radius: 4px; cursor: pointer; font-weight: bold; font-family: "Source Sans Pro", sans-serif; font-size: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
         📷 Save Map View
      </button>
    </div>
    {{% endmacro %}}
    """
    macro = MacroElement()
    macro._template = Template(legend_template)
    f_map.add_child(macro)

    folium.LayerControl().add_to(f_map)
    
    # Render the compiled interactive HTML component frame to the Streamlit page view canvas
    map_html = f_map._repr_html_()
    components.html(map_html, height=650, scrolling=True)