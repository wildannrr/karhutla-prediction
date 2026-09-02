# Whole-Kalimantan bounding box: (west, south, east, north)
KALIMANTAN_BBOX = (108.0, -4.5, 119.5, 4.5)

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
