import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import ee
import geemap
import os
import base64

# ==========================================
# 1. Earth Engine Authentication Setup
# ==========================================

try:
    # 1. Pull the clean, flat base64 sequence string from secrets
    b64_credentials = st.secrets["EARTHENGINE_CREDENTIALS_BASE64"]
    
    # 2. Decode the Base64 bytes back into a standard string layout
    decoded_bytes = base64.b64decode(b64_credentials)
    credentials_dict = json.loads(decoded_bytes.decode("utf-8"))
    
    service_account_email = credentials_dict["client_email"]
    
    # 3. Generate the local file mapping for the environment container
    with open("ee_credentials.json", "w") as f:
        json.dump(credentials_dict, f)
        
    # 4. Execute structural validation sequence
    credentials = ee.ServiceAccountCredentials(service_account_email, "ee_credentials.json")
    ee.Initialize(credentials)
    
except Exception as e:
    st.error(f"Earth Engine Authentication Failed. Error details: {e}")

# ==========================================
# 2. Model Loading (Cached)
# ==========================================
@st.cache_resource
def load_ml_pipeline():
    return joblib.load("best_rf_reduced_model.joblib")

model_pipeline = load_ml_pipeline()

# ==========================================
# 3. User Interface Layout & Session State
# ==========================================
st.title("Automated GEE Malaria Surveillance Platform")
st.write("Extracting automated environmental indices for Karagwe District, Tanzania.")

# Initialize session state so the map doesn't disappear on rerun
if "map_ready" not in st.session_state:
    st.session_state.map_ready = False
    st.session_state.smoothed_prediction_30m = None
    st.session_state.pixel_data = None
    st.session_state.aoi = None
    st.session_state.target_year = None

# User inputs the target year for the surveillance update
target_year = st.selectbox("Select Target Surveillance Year", [2020, 2021, 2022, 2023, 2024, 2025])

# NEW: Add dynamic district selection widget
target_district = st.selectbox("Select Target District", ["Karagwe", "Kyerwa"])

if st.button("Generate 30m Visual Risk Map Profile"):
        
        # ------------------------------------------
        # 4. Define Geographic Spatial Boundaries
        # ------------------------------------------
        districts = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
        
        # FIX: Dynamically filter strictly based on the user's dropdown choice
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
            
            # FIX: We construct a proper 5km scale projection for the rasterizer engine
            grid_projection = ee.Projection(commonCRS).atScale(pfprScale)
            
            # Convert points to images using the explicit 5km grid scale properties
            prediction_raster_5km = predicted_gee_layer.reduceToImage(
                properties=["predicted_PfPR"], 
                reducer=ee.Reducer.first()
            ).reproject(crs=grid_projection) # Forces Earth Engine to render true 5km grid cells
            
            # CHOOSE YOUR VIEW MODE BELOW:
            # Mode A: Comment this out if you want to see raw, sharp 5km squares.
            # Mode B: Keep it to see a true high-resolution 30m downscaled map based on real 5km data blocks.
            smoothed_prediction_30m = prediction_raster_5km.resample('bilinear').reproject(crs=commonProjection).clip(aoi_geometry)
            
            # Save elements to session state so they persist across canvas renders
            st.session_state.pixel_data = pixel_data
            st.session_state.smoothed_prediction_30m = smoothed_prediction_30m
            st.session_state.aoi = aoi
            st.session_state.target_year = target_year
            st.session_state.map_ready = True

# ==========================================
# 11. Persistent Map Rendering Render Loop
# ==========================================
if st.session_state.map_ready:
    st.success(f"Successfully processed {st.session_state.target_year} model pipeline!")
    st.write("### Interactive 30m Smoothed Risk Map:")
    
    import folium
    import streamlit.components.v1 as components
    from branca.element import Template, MacroElement

    # FIX: Dynamically adjust the map camera center coordinates based on selected district
    if st.session_state.target_district == "Kyerwa":
        map_center = [-1.30, 30.85] # Centered over Kyerwa's northern landscape
        map_zoom = 10
    else:
        map_center = [-1.59, 31.21] # Centered over Karagwe's southern landscape
        map_zoom = 9

    # Initialize a pure Folium map object with dynamic positioning
    f_map = folium.Map(location=map_center, zoom_start=map_zoom, control_scale=True)
    
    # 2. Extract and add the authenticated map tile URLs directly from Google Earth Engine
    # Vector Boundary Tile
    aoi_map_id = ee.Image().paint(st.session_state.aoi, 0, 2).getMapId()
    folium.TileLayer(
        tiles=aoi_map_id['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        name='Karagwe Border',
        overlay=True,
        control=True
    ).add_to(f_map)
    
    # 3. Define High-Contrast Color Gradients (Highly distinct blue/yellow/red spectrum)
    min_val = float(st.session_state.pixel_data["predicted_PfPR"].min())
    max_val = float(st.session_state.pixel_data["predicted_PfPR"].max())
    
    # Perceptually distinct, high-contrast palette values
    high_contrast_palette = ['#3288bd', '#99d594', '#e6f598', '#fee08b', '#fc8d59', '#d53e4f']
    
    vis_params = {
        'min': min_val,
        'max': max_val,
        'palette': high_contrast_palette
    }
    
    # Prediction Raster Tile
    prediction_map_id = st.session_state.smoothed_prediction_30m.getMapId(vis_params)
    folium.TileLayer(
        tiles=prediction_map_id['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        name=f'Predicted PfPR ({st.session_state.target_year})',
        overlay=True,
        control=True,
        opacity=0.85
    ).add_to(f_map)
    
    # 4. Inject a Continuous SHAP-Style Color Bar Legend & Print Layout
    v_min = f"{min_val:.1f}%"
    v_max = f"{max_val:.1f}%"

    # Turn the palette list into a CSS linear gradient string
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

    # 5. Add standard Layer Control toggle
    folium.LayerControl().add_to(f_map)
    
    # 6. Compile map asset to a raw HTML text stream string natively
    map_html = f_map._repr_html_()
    
    # Render the container frame directly onto the Streamlit application canvas
    components.html(map_html, height=650, scrolling=True)