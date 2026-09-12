"""Class taxonomy and disease knowledge base.

Maps the 38 PlantVillage classes onto the backend `pests_diseases` contract:

    predicted_pest_disease_id: int   (stable, deterministic per class)

Also holds disease knowledge priors used ONLY for the progression estimator and
explanation notes (standard agronomy knowledge, documented as priors — not
learned model output). These are association-level statements, not causal
claims established by this system.

The id scheme is deterministic: crop_id * 100 + index. It matches what the
backend seed script will use for the same rows; crop_id values here mirror the
backend `crops` table ordering.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Backend crops table (id, name) — subset relevant to PlantVillage classes.
CROPS: dict[int, str] = {
    1: "tomato",
    2: "potato",
    3: "grape",
    4: "maize",
    5: "soybean",
    6: "bell_pepper",
    7: "apple",
    8: "cherry",
    9: "peach",
    10: "strawberry",
    11: "blueberry",
    12: "orange",
    13: "squash",
    14: "raspberry",
}

# Maharashtra districts this system is scoped for (context encoder vocabulary).
REGIONS: list[str] = [
    "nashik",
    "pune",
    "aurangabad",
    "nagpur",
    "amravati",
    "solapur",
    "kolhapur",
    "latur",
    "akola",
    "wardha",
    "jalgaon",
    "sangli",
]

GROWTH_STAGES: list[str] = ["seedling", "vegetative", "flowering", "fruiting", "maturity"]


@dataclass(frozen=True)
class DiseaseInfo:
    name: str
    disease_type: str  # disease | pest | nutrient_disorder | healthy
    crop_id: int
    favorable_conditions: str = ""
    progression_days: tuple[int, int] = (5, 7)  # symptom-doubling horizon
    weather_sensitivity: str = "low"  # low | medium | high
    note: str = ""


_DISEASES: dict[str, DiseaseInfo] = {
    "Apple___Apple_scab": DiseaseInfo("Apple Scab", "disease", 7, "Cool, wet spring conditions favor scab development.", (5, 7), "high", "Visual: characteristic olive-black lesions on leaves."),
    "Apple___Black_rot": DiseaseInfo("Apple Black Rot", "disease", 7, "Warm, humid weather favors black rot infections.", (5, 7), "medium", ""),
    "Apple___Cedar_apple_rust": DiseaseInfo("Cedar Apple Rust", "disease", 7, "Wet spring weather and nearby juniper hosts favor rust.", (5, 7), "medium", ""),
    "Apple___healthy": DiseaseInfo("Healthy", "healthy", 7, "", (5, 7), "low", ""),
    "Blueberry___healthy": DiseaseInfo("Healthy", "healthy", 11, "", (5, 7), "low", ""),
    "Cherry_(including_sour)___healthy": DiseaseInfo("Healthy", "healthy", 8, "", (5, 7), "low", ""),
    "Cherry_(including_sour)___Powdery_mildew": DiseaseInfo("Cherry Powdery Mildew", "disease", 8, "Warm days and cool nights with high humidity favor mildew.", (5, 7), "high", ""),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": DiseaseInfo("Maize Gray Leaf Spot", "disease", 4, "Prolonged leaf wetness and warm temperatures favor GLS.", (5, 7), "high", ""),
    "Corn_(maize)___Common_rust_": DiseaseInfo("Maize Common Rust", "disease", 4, "Cool, humid nights with dew favor rust sporulation.", (5, 7), "high", ""),
    "Corn_(maize)___healthy": DiseaseInfo("Healthy", "healthy", 4, "", (5, 7), "low", ""),
    "Corn_(maize)___Northern_Leaf_Blight": DiseaseInfo("Maize Northern Leaf Blight", "disease", 4, "Extended wetness (>14h) and moderate temperatures favor NL B.", (5, 7), "high", ""),
    "Grape___Black_rot": DiseaseInfo("Grape Black Rot", "disease", 3, "Warm, wet weather during fruit set favors black rot.", (5, 7), "high", "Major concern for Nashik vineyards."),
    "Grape___Esca_(Black_Measles)": DiseaseInfo("Grape Esca", "disease", 3, "Often associated with vine stress and pruning wounds; slow progression.", (5, 7), "medium", "A wood disease — symptom spread is slow."),
    "Grape___healthy": DiseaseInfo("Healthy", "healthy", 3, "", (5, 7), "low", ""),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": DiseaseInfo("Grape Leaf Blight", "disease", 3, "Humid, warm vineyard conditions favor leaf blight.", (5, 7), "high", ""),
    "Orange___Haunglongbing_(Citrus_greening)": DiseaseInfo("Citrus Greening (HLB)", "disease", 12, "Spread by psyllid vectors; no cure — detection is critical.", (5, 7), "medium", "Vector-borne disease; isolation is the only response."),
    "Peach___Bacterial_spot": DiseaseInfo("Peach Bacterial Spot", "disease", 9, "Warm, wet weather favors bacterial spot epidemics.", (5, 7), "high", ""),
    "Peach___healthy": DiseaseInfo("Healthy", "healthy", 9, "", (5, 7), "low", ""),
    "Pepper,_bell___Bacterial_spot": DiseaseInfo("Bell Pepper Bacterial Spot", "disease", 6, "Warm, wet, windy weather spreads bacterial spot.", (5, 7), "high", ""),
    "Pepper,_bell___healthy": DiseaseInfo("Healthy", "healthy", 6, "", (5, 7), "low", ""),
    "Potato___Early_blight": DiseaseInfo("Potato Early Blight", "disease", 2, "Alternating wet-dry cycles and warm temperatures favor early blight.", (5, 7), "medium", ""),
    "Potato___healthy": DiseaseInfo("Healthy", "healthy", 2, "", (5, 7), "low", ""),
    "Potato___Late_blight": DiseaseInfo("Potato Late Blight", "disease", 2, "Cool (15–22C), wet weather with >10h leaf wetness favors late blight.", (5, 7), "high", "Epidemic potential is high in monsoon months."),
    "Raspberry___healthy": DiseaseInfo("Healthy", "healthy", 14, "", (5, 7), "low", ""),
    "Soybean___healthy": DiseaseInfo("Healthy", "healthy", 5, "", (5, 7), "low", ""),
    "Squash___Powdery_mildew": DiseaseInfo("Squash Powdery Mildew", "disease", 13, "High humidity, moderate temperatures favor powdery mildew.", (5, 7), "medium", ""),
    "Strawberry___healthy": DiseaseInfo("Healthy", "healthy", 10, "", (5, 7), "low", ""),
    "Strawberry___Leaf_scorch": DiseaseInfo("Strawberry Leaf Scorch", "disease", 10, "Warm, wet weather favors leaf scorch.", (5, 7), "medium", ""),
    "Tomato___Bacterial_spot": DiseaseInfo("Tomato Bacterial Spot", "disease", 1, "Warm, wet, windy conditions spread bacterial spot.", (5, 7), "high", ""),
    "Tomato___Early_blight": DiseaseInfo("Tomato Early Blight", "disease", 1, "Warm, humid weather with leaf wetness favors early blight.", (5, 7), "medium", ""),
    "Tomato___healthy": DiseaseInfo("Healthy", "healthy", 1, "", (5, 7), "low", ""),
    "Tomato___Late_blight": DiseaseInfo("Tomato Late Blight", "disease", 1, "Cool, wet monsoon weather strongly favors late blight.", (5, 7), "high", "Epidemic potential is high in monsoon months."),
    "Tomato___Leaf_Mold": DiseaseInfo("Tomato Leaf Mold", "disease", 1, "High humidity (85%+) and poor ventilation favor leaf mold.", (5, 7), "high", ""),
    "Tomato___Septoria_leaf_spot": DiseaseInfo("Tomato Septoria Leaf Spot", "disease", 1, "Warm, wet weather favors Septoria sporulation.", (5, 7), "high", ""),
    "Tomato___Spider_mites Two-spotted_spider_mite": DiseaseInfo("Two-Spotted Spider Mite", "pest", 1, "Hot, dry conditions favor mite population explosion.", (5, 7), "medium", "Pest, not a fungal disease — intervention differs."),
    "Tomato___Target_Spot": DiseaseInfo("Tomato Target Spot", "disease", 1, "Warm, wet weather favors target spot.", (5, 7), "high", ""),
    "Tomato___Tomato_mosaic_virus": DiseaseInfo("Tomato Mosaic Virus", "disease", 1, "Mechanical transmission; no cure — roguing is the response.", (5, 7), "low", "Viral disease; visual cues are often subtle."),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": DiseaseInfo("Tomato Yellow Leaf Curl Virus", "disease", 1, "Whitefly-vectored; high temperature favors vector buildup.", (5, 7), "medium", "Vector-borne viral disease."),
}

CLASSES: list[str] = sorted(_DISEASES.keys())


@dataclass(frozen=True)
class Taxonomy:
    """Deterministic mapping between class names and backend ids."""

    classes: list[str] = field(default_factory=lambda: CLASSES)

    def pest_disease_id(self, class_name: str) -> int:
        info = _DISEASES[class_name]
        if info.crop_id is None:
            raise KeyError(class_name)
        idx = _class_index_within_crop(class_name, info.crop_id)
        return info.crop_id * 100 + idx

    def info(self, class_name: str) -> DiseaseInfo:
        return _DISEASES[class_name]

    def crop_id(self, class_name: str) -> int:
        return _DISEASES[class_name].crop_id

    def crop_name(self, crop_id: int) -> str:
        return CROPS[crop_id]

    def is_healthy(self, class_name: str) -> bool:
        # Unknown classes (e.g. pest-detector names like 'spot_damage') are by
        # definition not a known healthy label.
        info = _DISEASES.get(class_name)
        return info is not None and info.disease_type == "healthy"

    def class_for_id(self, pest_disease_id: int) -> str:
        for name in self.classes:
            if self.pest_disease_id(name) == pest_disease_id:
                return name
        raise KeyError(pest_disease_id)


def _class_index_within_crop(class_name: str, crop_id: int) -> int:
    same_crop = sorted(
        n for n in CLASSES if _DISEASES[n].crop_id == crop_id
    )
    return same_crop.index(class_name) + 1


TAXONOMY = Taxonomy()