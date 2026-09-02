"""
Config: bounding boxes and constants for the Kalimantan karhutla (forest & land fire)
prediction project.

Province bounding boxes are APPROXIMATE rectangles (not precise administrative
boundaries). They're good enough to tag a hotspot with a likely province for
exploratory analysis. For anything that needs precise boundaries (e.g. area
statistics per kabupaten), swap this out for a real shapefile/GeoJSON and do a
proper spatial join with geopandas.
"""

# Whole-Kalimantan bounding box: (west, south, east, north)
KALIMANTAN_BBOX = (108.0, -4.5, 119.5, 4.5)

# Rough per-province bounding boxes (west, south, east, north).
# Real province borders are irregular polygons, not rectangles, so these
# WILL misclassify some points near borders. Good enough for exploratory
# analysis; for anything that needs to be precise, replace this with a
# real GeoJSON/shapefile (e.g. from Indonesia's Badan Informasi Geospasial)
# and do a proper point-in-polygon spatial join with geopandas instead.
#
# Dict order matters: more distinctive/smaller boxes are listed first and
# checked first, since a point can fall inside more than one rectangle.
#
# NOTE: because the fetch bbox (KALIMANTAN_BBOX) covers the whole island of
# Borneo geographically, it also picks up real hotspots in Malaysian Borneo
# (Sarawak/Sabah) and Brunei, which are NOT Indonesian provinces. Those are
# deliberately NOT force-fit into an Indonesian province box below - they
# get tagged "Malaysia/Brunei (Borneo)" by tag_province() instead of being
# silently misclassified.
PROVINCE_BBOXES = {
    "Kalimantan Selatan": (114.3, -4.4, 116.3, -1.2),
    "Kalimantan Utara":   (115.5, 1.7, 119.5, 4.5),
    "Kalimantan Timur":   (115.0, -2.7, 119.5, 1.7),
    "Kalimantan Tengah":  (110.7, -3.6, 115.0, 0.9),
    "Kalimantan Barat":   (108.0, -3.2, 111.5, 2.1),
}

# Points north of this latitude, within the overall bbox but outside every
# Indonesian province box above, are most likely in Malaysia (Sarawak/Sabah)
# or Brunei rather than actually "unclassified Indonesian territory."
MALAYSIA_BRUNEI_LAT_THRESHOLD = 1.5

# FIRMS sensors:
# - "*_NRT" sources only cover roughly the last ~60 days (near real-time).
# - "*_SP"  sources are the standard, quality-controlled *archive* product,
#           used for anything older than ~2 months.
SENSORS_NRT = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"]
SENSORS_ARCHIVE = ["VIIRS_SNPP_SP", "MODIS_SP"]  # NOAA-20/21 archive not always available

# FIRMS Area API only allows up to 10 days of data per request in theory,
# but has been observed to reject (400 Bad Request) requests above 5 days
# for some sensors/keys. 5 is the safe, documented value - use it.
MAX_DAY_RANGE = 5

FIRMS_AREA_URL_TEMPLATE = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "{map_key}/{sensor}/{bbox}/{day_range}/{date}"
)
