# this file serves as general experiment meta data including:
# strain_exp_dict that maps imaging experiments to the FOVS we use 
# as well as the strain associated with that strain 
# Minutes interval is a dict that keeps track of which experiment uses 10 or 5 minute image intervals
# Exposure_MS is a dict that tracks [phase, yfp] exposure rates in ms
# as well as a list of failed experiments. Right now lacZ induced with lactose is included to just 
# ommit it in the basal conditions other strains were tested in

# Gene -> experiment -> list of zero-padded FOV ids.  Add new experiments here.
STRAIN_EXP_DICT = {
    "alkA": {
        "DUMM_alkA_chpS_111325": ["017", "018", "022", "023"],
        "DUMM_giTG066_063_120925": ["018", "019"],
        "DUMM_giTG068_063_061725": ["000", "006"],
        "DUMM_giTG068_063_061725_v2": ["000"],
        "DUMM_giTG069_063_121125": ["001", "005", "006", "008"],
        "DUMM_giTG069_063_121225": ["000"],
        "DUMM_giTG63_giTG67_Glucose_121724": ["005"],
        "DUMM_giTG63_giTG67_Glucose_121724_1_v2": ["005"],
    },
    "araB": {
        "CL008_giTG068_072925": ["003", "007", "009"],
        "DUMM_CL008_080825": ["002", "005", "006", "007"],
        "DUMM_CL008_giTG061_081225": ["015", "016", "017", "018", "019", "020"],
    },
    "baeS": {
        "DUMM_baeS_gfcE_112025": ["005"],
        "DUMM_giTG066_063_120925": ["002", "007"],
        "DUMM_giTG66_Glucose_012325": ["003"],
        "DUMM_gitg068_baeS_100225": ["013", "017", "018", "019", "020", "021", "022", "023"],
    },
    "chpS": {
        "DUMM_alkA_chpS_111325": ["003", "005", "006", "007", "009", "010", "011"],
        "DUMM_giTG062_064_121325": ["005", "010", "012"],
        "DUMM_giTG62_Glucose_012925": ["005"],
    },
    "gfcE": {
        "DUMM_baeS_gfcE_112025": ["011", "012", "016"],
        "DUMM_giTG060_064_121425": ["013", "014", "020", "022"],
        "DUMM_giTG062_064_121325": ["015", "019"],
        "DUMM_giTG064_Glucose_022625": ["001", "015"],
    },
    "gldA": {
        "DUMM_giTG069_063_121125": ["013", "016", "022", "025"],
        "DUMM_giTG069_063_121225": ["013", "018", "022"],
        "DUMM_giTG69_Glucose_013025": ["007"],
    },
    "hupA": {
        "DUMM_giTG068_052925": ["005"],
        "DUMM_giTG068_059__lactose_062025": ["009"],
        "DUMM_giTG068_063_061725": ["012"],
    },
    "lacZ": {
        "DUMM_giTG059_060_061125": ["008"],
        "DUMM_giTG059_068_SC_093025": ["015", "020"],
        "DUMM_giTG059_noKan_Glucose_031125": ["004"],
        "DUMM_giTG068_059__lactose_062025": ["000", "001", "002"],
    },
    "mazF": {
        "DUMM_giTG059_060_061125": ["015"],
        "DUMM_giTG060_064_121425": ["000", "005"],
        "DUMM_mazF_murQ_121625": ["005", "006", "012"],
    },
}

# time interval (min)
MINUTES_INTERVAL = {
    "CL008_giTG068_072925": 10,
    "DUMM_CL008_080825": 10,
    "DUMM_CL008_giTG061_081225": 10,
    "DUMM_alkA_chpS_111325": 10,
    "DUMM_baeS_gfcE_112025": 10,
    "DUMM_giTG059_060_061125": 10,
    "DUMM_giTG059_068_SC_093025": 10,
    "DUMM_giTG059_noKan_Glucose_031125": 5,
    "DUMM_giTG060_064_121425": 10,
    "DUMM_giTG062_064_121325": 10,
    "DUMM_giTG064_Glucose_022625": 5,
    "DUMM_giTG066_063_120925": 10,
    "DUMM_giTG068_052925": 5,
    "DUMM_giTG068_063_061725": 10,
    "DUMM_giTG068_063_061725_v2": 10,
    "DUMM_giTG069_063_121125": 10,
    "DUMM_giTG069_063_121225": 10,
    "DUMM_giTG62_Glucose_012925": 5,
    "DUMM_giTG63_giTG67_Glucose_121724": 5,
    "DUMM_giTG63_giTG67_Glucose_121724_1_v2": 5,
    "DUMM_giTG66_Glucose_012325": 5,
    "DUMM_giTG69_Glucose_013025": 5,
    "DUMM_gitg068_baeS_100225": 10,
    "DUMM_mazF_murQ_121625": 10,
    "DUMM_giTG068_059__lactose_062025": 10,
}
#[phase, yfp] exposure (ms)
EXPOSURE_MS = {
    "CL008_giTG068_072925": [45, 150],
    "DUMM_CL008_080825": [120, 450],
    "DUMM_CL008_giTG061_081225": [120, 150],
    "DUMM_alkA_chpS_111325": [120, 150],
    "DUMM_baeS_gfcE_112025": [120, 150],
    "DUMM_giTG059_060_061125": [250, 250],
    "DUMM_giTG059_068_SC_093025": [120, 150],
    "DUMM_giTG059_noKan_Glucose_031125": [250, 450],
    "DUMM_giTG060_064_121425": [120, 150],
    "DUMM_giTG062_064_121325": [120, 150],
    "DUMM_giTG064_Glucose_022625": [250, 500],
    "DUMM_giTG066_063_120925": [120, 150],
    "DUMM_giTG068_052925": [250, 450],
    "DUMM_giTG068_063_061725": [45, 150],
    "DUMM_giTG068_063_061725_v2": [45, 150],
    "DUMM_giTG069_063_121125": [120, 150],
    "DUMM_giTG069_063_121225": [120, 150],
    "DUMM_giTG62_Glucose_012925": [250, 500],
    "DUMM_giTG63_giTG67_Glucose_121724": [250, 500],
    "DUMM_giTG63_giTG67_Glucose_121724_1_v2": [250, 500],
    "DUMM_giTG66_Glucose_012325": [250, 500],
    "DUMM_giTG69_Glucose_013025": [250, 500],
    "DUMM_gitg068_baeS_100225": [120, 150],
    "DUMM_mazF_murQ_121625": [120, 150],
    "DUMM_giTG068_059__lactose_062025": [90, 150],
}

FAILED_EXPERIMENTS = (
    "DUMM_alkA_chpS_111325",             # alkA (9 trenches) + chpS (15)
    "DUMM_baeS_gfcE_112025",             # gfcE (7) only -- no baeS data despite the name
    "DUMM_giTG068_059__lactose_062025",  # lacZ (4) + hupA (1)
)

FAILED_KYMOGRAPHS = (
    # duplicate trench 
    ("DUMM_CL008_080825", "002", "1355"),
    # need to finish segmenting for more data 
    ("DUMM_giTG064_Glucose_022625", "001", "2501"),
    # missing fluorescence data -- 001/114 is already absent from every table (cells, fiji,
    # all_data; it survives only in time_dict_082026.json), and an entry that matches no
    # trench makes drop_excluded_kymographs raise, so it is recorded here but not listed.
    # ("DUMM_giTG064_Glucose_022625", "001", "114"),


)


def exposure_for(experiment_name):
    phase, fluor = EXPOSURE_MS.get(experiment_name, (None, None))
    return phase, fluor
