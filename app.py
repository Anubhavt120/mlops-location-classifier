import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
from PIL import Image
from src.config import CLASSES, SAMPLE_IMAGES_DIR
from src.predict import predict_location
from src.utils import geocode_location_api

# Page setup
st.set_page_config(
    page_title="Location Image Classifier",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Class Metadata (Pure English, no Hindi)
CLASS_METADATA = {
    "glacier": {
        "icon": "🧊",
        "title": "Glacier",
        "color": "#38bdf8",
        "desc": "Glacial ice formations, snow fields, and frozen polar terrain.",
    },
    "mountain": {
        "icon": "🏔️",
        "title": "Mountain",
        "color": "#a855f7",
        "desc": "Rocky peaks, high-altitude ranges, and mountainous cliffs.",
    },
    "sea": {
        "icon": "🌊",
        "title": "Sea / Ocean",
        "color": "#0ea5e9",
        "desc": "Open ocean waters, coastal horizons, and marine shorelines.",
    },
    "forest": {
        "icon": "🌲",
        "title": "Forest",
        "color": "#22c55e",
        "desc": "Dense woodlands, green tree canopies, and natural vegetation.",
    },
    "buildings": {
        "icon": "🏙️",
        "title": "Buildings",
        "color": "#f97316",
        "desc": "Urban city skylines, architecture, and structural high-rises.",
    },
    "street": {
        "icon": "🛣️",
        "title": "Street",
        "color": "#6366f1",
        "desc": "City roadways, asphalt avenues, traffic lanes, and urban streets.",
    },
}

# Pure Black Dark Theme CSS
st.markdown(
    """
    <style>
    /* Pure Black Background */
    .stApp {
        background-color: #030712 !important;
        background: radial-gradient(circle at 50% 0%, #111827 0%, #030712 100%) !important;
        color: #f9fafb !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main Container Cards */
    .dark-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.7);
    }

    /* Header Styling */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #93c5fd 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }

    .main-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 22px;
    }

    /* Prediction Card */
    .result-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 20px 24px;
        margin: 20px 0 16px 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
    }

    .result-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        color: #64748b;
    }

    .result-name {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 4px 0 8px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .confidence-tag {
        font-size: 0.95rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }

    /* Geographical Location Box */
    .geo-box {
        background: linear-gradient(135deg, #0b1329 0%, #0f172a 100%);
        border: 1px solid #2563eb;
        border-radius: 16px;
        padding: 20px 24px;
        margin: 18px 0 16px 0;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.15);
    }

    .geo-tag {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 4px;
    }

    .geo-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
    }

    .geo-coords {
        font-size: 0.9rem;
        color: #94a3b8;
        font-family: monospace;
        margin-bottom: 6px;
    }

    .geo-desc {
        font-size: 0.95rem;
        color: #cbd5e1;
        line-height: 1.4;
    }

    /* Animation Warning Banner */
    .animation-warning {
        background: #422006;
        border: 1px solid #b45309;
        border-radius: 12px;
        padding: 14px 18px;
        color: #fef3c7;
        margin: 16px 0;
        font-size: 0.95rem;
    }

    /* Sidebar Dark Theme Overrides */
    section[data-testid="stSidebar"] {
        background-color: #070d19 !important;
        border-right: 1px solid #1e293b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.title("⚙️ Model Controls")
selected_model = st.sidebar.selectbox(
    "Architecture:",
    ["MobileNetV2 (High Accuracy - 86.1%)", "Custom Deep CNN (Baseline)"],
    index=0,
)
model_name = "MobileNetV2" if "MobileNetV2" in selected_model else "Custom CNN"

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 1-Click Instant Samples")
st.sidebar.caption("Click any preset photo to load and test instantly:")

sample_buttons = {
    "🧊 Glacier Sample": "glacier.jpg",
    "🏔️ Mountain Sample": "mountain.jpg",
    "🌊 Sea Sample": "sea.jpg",
    "🌲 Forest Sample": "forest.jpg",
    "🏢 Buildings Sample": "buildings.jpg",
    "🚗 Street Sample": "street.jpg",
}

for label, fname in sample_buttons.items():
    if st.sidebar.button(label, use_container_width=True):
        st.session_state["chosen_mode"] = "Upload Image"
        st.session_state["active_input"] = str(SAMPLE_IMAGES_DIR / fname)
        st.session_state["input_type"] = "sample"
        st.session_state["active_geo_query"] = ""  # Auto-refresh geo for selected sample

# Header
st.markdown('<div class="main-title">Location Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Deep learning location and landscape scene recognition powered by MobileNetV2</div>', unsafe_allow_html=True)

# 2 Choices for Input (Pure English)
input_choice = st.radio(
    "Choose Input Method:",
    options=["Upload Image", "Enter Image URL"],
    index=0 if st.session_state.get("chosen_mode") != "Enter Image URL" else 1,
    horizontal=True,
    key="input_choice_radio",
)

st.session_state["chosen_mode"] = input_choice
image_source = None
source_name_hint = ""

# CHOICE 1: Upload Image Mode
if input_choice == "Upload Image":
    st.markdown("##### 📁 Select or Drop an Image:")
    uploaded_file = st.file_uploader(
        label="Click or Drag & Drop photo here (JPG, JPEG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="main_file_uploader",
    )
    if uploaded_file is not None:
        image_source = uploaded_file
        source_name_hint = uploaded_file.name
    elif st.session_state.get("input_type") == "sample":
        image_source = st.session_state.get("active_input")
        source_name_hint = Path(str(image_source)).name
    else:
        # Default starting sample
        default_sample = SAMPLE_IMAGES_DIR / "glacier.jpg"
        if default_sample.exists():
            image_source = str(default_sample)
            source_name_hint = "glacier.jpg"

# CHOICE 2: URL Input Mode
else:
    st.markdown("##### 🔗 Provide Image URL:")
    col_u1, col_u2 = st.columns([4, 1])
    with col_u1:
        url_text = st.text_input(
            label="Enter Image URL to Classify..",
            value=st.session_state.get("last_entered_url", "https://storage.googleapis.com/image_classification_2021/Glacier-Argentina-South-America"),
            placeholder="https://example.com/location_photo.jpg",
            key="main_url_input",
            label_visibility="collapsed",
        )
    with col_u2:
        fetch_url_btn = st.button("🚀 Classify", use_container_width=True)

    if url_text.strip():
        image_source = url_text.strip()
        source_name_hint = url_text.strip()
        st.session_state["last_entered_url"] = url_text.strip()

# Run Inference & Display Results
if image_source:
    try:
        current_geo_hint = st.session_state.get("active_geo_query", "")
        # If user hasn't explicitly searched, use the source filename/URL hint
        effective_hint = current_geo_hint if current_geo_hint else source_name_hint

        with st.spinner("Analyzing location features & coordinates..."):
            result = predict_location(
                image_source,
                model_name=model_name,
                location_hint=effective_hint,
            )

        pred_class = result["predicted_class"]
        confidence = result["confidence"]
        probs = result["probabilities"]
        display_img = result["image"]
        is_anim = result.get("is_animation", False)
        anim_reason = result.get("animation_reason", "")
        geo = result.get("geo_location", {})

        # Animation / Cartoon Detection Alert
        if is_anim:
            st.markdown(
                f"""
                <div class="animation-warning">
                    <strong>⚠️ Animation / Graphic Image Detected!</strong><br>
                    {anim_reason}. This model is trained on real geographical landscapes. For accurate results, please upload a real camera photograph.
                </div>
                """,
                unsafe_allow_html=True,
            )

        meta = CLASS_METADATA.get(pred_class, {
            "icon": "📍",
            "title": pred_class.capitalize(),
            "color": "#38bdf8",
            "desc": "Scene location recognized.",
        })

        # 1. Main Predicted Class Card
        st.markdown(
            f"""
            <div class="result-card" style="border-left: 6px solid {meta['color']};">
                <div class="result-label">Predicted Class :</div>
                <div class="result-name">
                    <span>{meta['icon']} {meta['title']}</span>
                    <span class="confidence-tag">{confidence:.1%} Match</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.95rem;">{meta['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Geographical Location Card (Coordinates, Place & Google Map)
        if geo:
            lat = float(geo.get("latitude", 0.0))
            lon = float(geo.get("longitude", 0.0))
            lat_str = f"{abs(lat):.4f}° {'N' if lat >= 0 else 'S'}"
            lon_str = f"{abs(lon):.4f}° {'E' if lon >= 0 else 'W'}"
            location_name = geo.get("location_name", "Global Landscape")
            source_type = geo.get("source_type", "AI Geo Analysis")
            region_details = geo.get("region_details", "")
            candidates = geo.get("candidates", [])

            st.markdown(
                f"""
                <div class="geo-box">
                    <div class="geo-tag">📍 Estimated Real-World Location & Map</div>
                    <div class="geo-title">🌍 {location_name}</div>
                    <div class="geo-coords">Coordinates: {lat_str}, {lon_str} &bull; Source: {source_type}</div>
                    <div class="geo-desc">{region_details}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Embedded Real Google Map (Interactive iframe with pin & zoom)
            google_map_embed_url = f"https://maps.google.com/maps?q={lat},{lon}&hl=en&z=12&output=embed"
            st.markdown(
                f"""
                <div style="margin: 14px 0 10px 0; border-radius: 14px; overflow: hidden; border: 1px solid #1e293b; box-shadow: 0 10px 30px rgba(0,0,0,0.7);">
                    <iframe width="100%" height="380" frameborder="0" style="border:0;" src="{google_map_embed_url}" allowfullscreen></iframe>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Direct Google Maps Action Buttons
            col_map_btn1, col_map_btn2 = st.columns(2)
            with col_map_btn1:
                st.link_button(
                    "🗺️ Open in Google Maps (Full View)",
                    f"https://www.google.com/maps/search/?api=1&query={lat},{lon}",
                    use_container_width=True,
                )
            with col_map_btn2:
                st.link_button(
                    "🧭 Get Directions on Google Maps",
                    f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}",
                    use_container_width=True,
                )

            # Interactive Location Search / Update using Live Geocoding API
            st.markdown("---")
            st.markdown("##### 📍 Search or Change Location (Powered by Live Geocoding API):")
            st.caption("Type any city, mountain, valley, beach, or landmark in the world to update the Google Map live:")

            col_s1, col_s2 = st.columns([3, 1])
            with col_s1:
                user_loc_query = st.text_input(
                    "Search location:",
                    value=st.session_state.get("active_geo_query", ""),
                    placeholder="e.g. Siachen Glacier, Ladakh, Mount Fuji, Swiss Alps, Goa, Paris...",
                    key="loc_search_input_box",
                    label_visibility="collapsed",
                )
            with col_s2:
                search_btn = st.button("🔍 Update Map", use_container_width=True)

            if search_btn and user_loc_query.strip():
                if st.session_state.get("active_geo_query") != user_loc_query.strip():
                    st.session_state["active_geo_query"] = user_loc_query.strip()
                    st.rerun()

            # Alternative Famous Landmarks for this scene
            if candidates:
                cand_options = ["Explore alternative famous landmarks for this scene..."] + candidates
                chosen_cand = st.selectbox(
                    "Or pick a world landmark from this scene:",
                    options=cand_options,
                    index=0,
                    key="cand_select_box",
                )
                if chosen_cand != "Explore alternative famous landmarks for this scene...":
                    if st.session_state.get("active_geo_query") != chosen_cand:
                        st.session_state["active_geo_query"] = chosen_cand
                        st.rerun()

            # Optional Vector Pin Map
            with st.expander("🌐 View Terrain Mapbox Pin", expanded=False):
                map_df = pd.DataFrame([{"lat": lat, "lon": lon}])
                st.map(map_df, zoom=10 if geo.get("has_hardware_gps") else 4)


        # 3. Classified Image Preview
        st.markdown("### 🖼️ Classified Image Preview:")
        st.image(display_img, use_container_width=True, caption=f"Classified Scene: {pred_class.upper()}")

        # 4. Footer Interactive Option for Real Data Percentages
        st.markdown("---")
        with st.expander("📊 Tap to View Real Data Breakdown (All 6 Categories)", expanded=False):
            st.markdown("##### 📈 Real Data Probability Distribution:")
            st.caption("Quantitative probability scores calculated directly from the deep learning softmax layer:")

            for cls_name, prob_val in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                cmeta = CLASS_METADATA.get(cls_name, {"icon": "📍", "color": "#38bdf8"})
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; font-weight: 600; margin-top: 8px;">
                        <span style="color: #e2e8f0;">{cmeta['icon']} {cls_name.capitalize()}</span>
                        <span style="color: #38bdf8; font-family: monospace;">{prob_val * 100:.2f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(float(prob_val))

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Tip: Please check the image URL or upload a valid photo file.")
else:
    st.info("Please upload a photo or enter an image URL to identify its location.")


# Vercel Serverless Function & WSGI entrypoint compatibility
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        html_path = Path(__file__).resolve().parent / "index.html"
        if html_path.exists():
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.wfile.write(b"<h1>Location Image Classifier - MLOps System</h1>")

app = handler
application = handler


