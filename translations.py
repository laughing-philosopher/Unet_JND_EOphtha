# translations.py (Make sure this is saved in the exact same folder as app.py!)

# translations.py

LANGUAGES = {
    "English": "en",
    "हिन्दी (Hindi)": "hi",
    "ଓଡ଼ିଆ (Odia)": "or",
    "বাংলা (Bengali)": "bn"
}

UI_TEXT = {
    "en": {
        "app_title": "AAKHI",
        "app_subtitle": "AI Assisted Kit for Human Intraocular Imaging",
        "app_motto": "Simple. Offline. Retina Screening.",
        "patient_details": "🧑‍⚕️ Patient Details",
        "patient_name": "Patient Name",
        "patient_age": "Patient Age",
        "gender": "Gender",
        "confirm": "Confirm Patient",
        "change_patient": "Change Patient",
        "upload_prompt": "Upload fundus image",
        "analyze_btn": "▶ Analyze",
        "export_btn": "📥 Export Report",
        "active_model": "Active model:",
        "ODOC": "Optic Disc / Cup (OD-OC)",
        "DR_GRADING": "Diabetic Retinopathy Grading",
        "LESIONS": "Multi-Lesion Detection",
        "MA": "Microaneurysm Detection",
        "analysis_results": "Analysis Results"
    },
    "hi": {
        "app_title": "आँखी (AAKHI)",
        "app_subtitle": "मानव अंतःनेत्रीय इमेजिंग के लिए एआई सहायक किट",
        "app_motto": "सरल। ऑफ़लाइन। रेटिना स्क्रीनिंग।",
        "patient_details": "🧑‍⚕️ मरीज का विवरण",
        "patient_name": "मरीज का नाम",
        "patient_age": "उम्र",
        "gender": "लिंग",
        "confirm": "मरीज की पुष्टि करें",
        "change_patient": "मरीज बदलें",
        "upload_prompt": "फंडस छवि अपलोड करें",
        "analyze_btn": "▶ विश्लेषण करें",
        "export_btn": "📥 रिपोर्ट डाउनलोड करें",
        "active_model": "सक्रिय मॉडल:"
    },
    "or": {
        "app_title": "ଆଖି (AAKHI)",
        "app_subtitle": "ମାନବ ଇଣ୍ଟ୍ରାଓକୁଲାର୍ ଇମେଜିଙ୍ଗ୍ ପାଇଁ AI ସହାୟକ କିଟ୍",
        "app_motto": "ସରଳ। ଅଫଲାଇନ୍। ରେଟିନାଲ୍ ସ୍କ୍ରିନିଂ।",
        "patient_details": "🧑‍⚕️ ରୋଗୀର ବିବରଣୀ",
        "patient_name": "ରୋଗୀର ନାମ",
        "patient_age": "ବୟସ",
        "gender": "ଲିଙ୍ଗ",
        "confirm": "ନିଶ୍ଚିତ କରନ୍ତୁ",
        "change_patient": "ରୋଗୀ ପରିବର୍ତ୍ତନ କରନ୍ତୁ",
        "upload_prompt": "ଫଣ୍ଡସ୍ ଇମେଜ୍ ଅପଲୋଡ୍ କରନ୍ତୁ",
        "analyze_btn": "▶ ବିଶ୍ଳେଷଣ କରନ୍ତୁ",
        "export_btn": "📥 ରିପୋର୍ଟ ରପ୍ତାନି କରନ୍ତୁ",
        "active_model": "ସକ୍ରିୟ ମଡେଲ୍:"
    },
    "bn": {
        "app_title": "আঁখি (AAKHI)",
        "app_subtitle": "মানব ইন্ট্রাওকুলার ইমেজিংয়ের জন্য এআই সহায়তা কিট",
        "app_motto": "সহজ। অফলাইন। রেটিনাল স্ক্রীনিং।",
        "patient_details": "🧑‍⚕️ রোগীর বিবরণ",
        "patient_name": "রোগীর নাম",
        "patient_age": "বয়স",
        "gender": "লিঙ্গ",
        "confirm": "নিশ্চিত করুন",
        "change_patient": "রোগী পরিবর্তন করুন",
        "upload_prompt": "ফান্ডাস ছবি আপলোড করুন",
        "analyze_btn": "▶ বিশ্লেষণ করুন",
        "export_btn": "📥 রিপোর্ট ডাউনলোড করুন",
        "active_model": "সক্রিয় মডেল:"
    }
}

def get_text(lang_code, key):
    return UI_TEXT.get(lang_code, UI_TEXT["en"]).get(key, UI_TEXT["en"].get(key, key))