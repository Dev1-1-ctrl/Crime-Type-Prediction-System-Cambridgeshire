import pandas as pd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
from pathlib import Path

MODE = "predicted_2026"
INPUT_ACTUAL = "crime_type_features.csv"
INPUT_PRED = "predicted_crime_2025_per_lsoa.csv"   # from Step 5
YEAR = 2025
MAX_HEAT_POINTS = 8000
CENTER = [52.2, 0.12]
ZOOM = 10

# helpers


def group_crime(c):
    c = str(c).lower()
    if any(k in c for k in ["theft", "burglary", "shoplifting", "robbery", "vehicle"]):
        return "Theft"
    if any(k in c for k in ["violence", "weapons", "arson", "criminal damage"]):
        return "Violence"
    if "anti-social" in c:
        return "Anti-social"
    if "drug" in c:
        return "Drugs"
    return "Other"


icon_by_group = {
    "Violence":    ("darkpurple", "shield"),
    "Theft":       ("orange",     "shopping-cart"),
    "Anti-social": ("green",      "users"),
    "Drugs":       ("black",      "flask"),
    "Other":       ("gray",       "question")
}
colors_for_bar = {
    "Violence": "#6f42c1",
    "Theft": "#fd7e14",
    "Anti-social": "#198754",
    "Drugs": "#000000",
    "Other": "gray"
}


def pin_size(n):  # not making the sizes too big or too small.
    try:
        n = float(n)
    except:
        return 6
    return max(4, min(11, int((n**0.5)/2 + 4)))


# load and prepare
if MODE == "actual_2025":
    df = pd.read_csv(INPUT_ACTUAL)
    for c in ["Latitude", "Longitude", "Crime type", "LSOA name"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column in {INPUT_ACTUAL}: {c}")

    # basic cleaning and filtering the year
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    if YEAR is not None and "Year" in df.columns:
        df = df[df["Year"] == YEAR].copy()

    # group to 5
    df["Crime_group"] = df["Crime type"].apply(group_crime)

    # per-LSOA centroid, modal group and counts (for pins)
    centroids = df.groupby("LSOA name")[["Latitude", "Longitude"]].mean()
    mode_grp = df.groupby("LSOA name")["Crime_group"].agg(
        lambda s: s.mode().iat[0])
    counts = df.groupby("LSOA name").size()
    agg = (centroids.join(mode_grp.rename("top_group"))
           .join(counts.rename("count"))
           .reset_index()
           .dropna(subset=["Latitude", "Longitude"]))

    # heatmap points = raw incidents
    heat_points = df[["Latitude", "Longitude"]].copy()
    out_html = f"crime_map_actual_{YEAR}.html"
    out_png = f"chart_actual_{YEAR}_distribution.png"
    out_csv = f"summary_actual_{YEAR}_counts.csv"

elif MODE == "predicted_2026":
    # expected columns: LSOA name, Latitude, Longitude, top_group
    df = pd.read_csv(INPUT_PRED)
    for c in ["Latitude", "Longitude", "LSOA name", "top_group"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column in {INPUT_PRED}: {c}")

    # pins is one per LSOA, color by predicted group
    agg = df.rename(columns={"top_group": "top_group"})[
        ["LSOA name", "Latitude", "Longitude", "top_group"]].copy()
    agg["count"] = 1

    # heatmap: we only have centroids
    heat_points = df[["Latitude", "Longitude"]].copy()
    out_html = "crime_map_predicted_2026.html"
    out_png = "chart_predicted_2026_distribution.png"
    out_csv = "summary_predicted_2026_counts.csv"

else:
    raise ValueError("MODE must be 'actual_2025' or 'predicted_2026'.")

#  make map
m = folium.Map(location=CENTER, zoom_start=ZOOM, control_scale=True)

# basemaps (Google-ish)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Esri.WorldStreetMap", overlay=False, control=True
).add_to(m)
folium.TileLayer("CartoDB Voyager", name="CartoDB.Voyager",
                 control=True).add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Esri.WorldImagery (satellite)", overlay=False, control=True
).add_to(m)

# Heatmap layer
heat_layer = folium.FeatureGroup(name="Heatmap", show=True)
hp = heat_points.dropna().copy()
if len(hp) > MAX_HEAT_POINTS:
    hp = hp.sample(MAX_HEAT_POINTS, random_state=42)
HeatMap(hp[["Latitude", "Longitude"]].values.tolist(),
        radius=8, blur=12).add_to(heat_layer)
heat_layer.add_to(m)

# Pins (centroids)
areas_layer = folium.FeatureGroup(
    name="LSOA area pins" if MODE == "actual_2025" else "Predicted LSOA pins",
    show=True
).add_to(m)

for _, r in agg.iterrows():
    grp = r.get("top_group", "Other")
    color, icon = icon_by_group.get(grp, ("lightgray", "map-marker"))
    folium.Marker(
        [r["Latitude"], r["Longitude"]],
        icon=folium.Icon(color=color, icon=icon, prefix="fa"),
        popup=folium.Popup(
            (
                f"<b>LSOA:</b> {r['LSOA name']}<br>"
                f"<b>{'Top group' if MODE == 'predicted_2026' else 'Top group'}:</b> {grp}<br>"
                f"{'' if MODE == 'predicted_2026' else f'<b>Crimes ({YEAR}):</b> {int(r.get('count', 1))}'}"
            ),
            max_width=320
        )
    ).add_to(areas_layer)

# Legend
legend = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
background: white; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; font-size: 13px;">
<b>Crime Group</b><br>
<span style="color:#6f42c1;">◆</span> Violence (darkpurple pin)<br>
<span style="color:#fd7e14;">◆</span> Theft (orange pin)<br>
<span style="color:#198754;">◆</span> Anti-social (green pin)<br>
<span style="color:#000000;">◆</span> Drugs (black pin)<br>
<span style="color:#6c757d;">◆</span> Other (gray pin)
</div>
"""
m.get_root().html.add_child(folium.Element(legend))

folium.LayerControl(collapsed=False).add_to(m)
m.save(out_html)
print(f" Saved map → {out_html}")

#  charts and  summary
if MODE == "actual_2025":
    # distribution by top group per LSOA
    dist = agg["top_group"].value_counts().reindex(
        ["Violence", "Theft", "Anti-social", "Drugs", "Other"]).fillna(0)
else:
    # predicted per-LSOA distribution
    dist = df["top_group"].value_counts().reindex(
        ["Violence", "Theft", "Anti-social", "Drugs", "Other"]).fillna(0)

# bar chart
plt.figure(figsize=(7, 4))
dist.plot(kind="bar", color=[colors_for_bar[g] for g in dist.index])
plt.title("Crime Group Distribution" + (" (Actual 2025)" if MODE ==
          "actual_2025" else " (Predicted 2026)"))
plt.ylabel("Number of LSOAs")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(out_png, dpi=150)
plt.close()
print(f" Saved chart → {out_png}")

# CSV summary
dist.to_csv(out_csv, header=["count"])
print(f" Saved summary → {out_csv}")

print("Done.")
