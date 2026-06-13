# ============================================================
# FairGraph-CV: Recruitment Model (XGBoost Regressor)
# Built on: resume_dataset_augmented.csv
# Version : 2.0 — Fixed scoring, uses hiring_score as target
# ============================================================
#
# FIXES in v2.0:
#   FIX 1 → Switched from binary classifier to XGBoost Regressor
#            Target is now hiring_score (2.58–8.4) not binary selected
#            This gives meaningful score range for bias comparison
#
#   FIX 2 → employment_gap was all zeros in dataset (no real data)
#            Replaced with a computed gap from graduation_year
#            Female candidates realistically get larger gaps
#
#   FIX 3 → Added score normalization (0.0–1.0) for easy comparison
#            across all intern modules
#
# Bias factors detected:
#   Gender         → Male avg 5.04 vs Female avg 4.50
#   University Tier→ Tier1 avg 6.98 vs Tier3 avg 4.41
#   University Type→ Govt avg 6.30 vs Private avg 4.63
#   Region         → Minor signal
# ============================================================

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)


# ============================================================
# SECTION 1: CONSTANTS & LOOKUP MAPS
# ============================================================

DATASET_PATH = "/content/resume_dataset_augmented.csv"

TIER_MAP = {
    "Tier 1": 1,
    "Tier 2": 2,
    "Tier 3": 3
}

DEGREE_MAP = {
    "PhD"    : 6,
    "M.Tech" : 5,
    "MBA"    : 4,
    "B.Tech" : 4,
    "B.E."   : 4,
    "MSc"    : 3,
    "MCA"    : 3,
    "BSc"    : 2,
    "BCA"    : 2,
}

REGION_MAP = {
    "Urban"      : 3,
    "Semi-Urban" : 2,
    "Rural"      : 1,
}

UTYPE_MAP = {
    "Government" : 2,
    "Private"    : 1,
}

FEATURE_NAMES = [
    # ── Bias factors ──────────────────────────────
    "gender_enc",            # 0=Female, 1=Male
    "tier_enc",              # 1=Tier1, 2=Tier2, 3=Tier3
    "utype_enc",             # 1=Private, 2=Government
    "degree_enc",            # degree prestige rank
    "region_enc",            # 1=Rural, 2=SemiUrban, 3=Urban
    "employment_gap_months", # computed gap (months)
    # ── Merit factors ─────────────────────────────
    "cgpa",
    "years_experience",
    "skill_count",
    "has_certification",
    "internship_count",
    "projects_count",
    "communication_score",
    "technical_score",
    "aptitude_score",
]


# ============================================================
# SECTION 2: LOAD & CLEAN DATASET
# ============================================================

def compute_employment_gap(row):
    """
    FIX 2: employment_gap_months in the dataset is all zeros.
    We compute a realistic gap from graduation_year.

    Logic:
    - Base gap = (2024 - graduation_year) - years_experience
    - Female candidates get an additional realistic gap
      (research shows women take longer to enter workforce)
    - Clamp to 0–36 months
    """
    try:
        grad_year   = int(row["graduation_year"])
        exp_years   = float(row["years_experience"]) if pd.notna(row["years_experience"]) else 0
        gender      = row["gender"]

        raw_gap_years = (2024 - grad_year) - exp_years
        gap_months    = max(0, raw_gap_years * 12)

        # Female bias: add realistic additional gap
        if gender == "Female":
            gap_months += np.random.choice([0, 3, 6, 12],
                                           p=[0.50, 0.25, 0.15, 0.10])
        else:
            gap_months += np.random.choice([0, 1, 3],
                                           p=[0.70, 0.20, 0.10])

        return float(np.clip(gap_months, 0, 36))
    except:
        return 0.0


def load_and_clean(path=DATASET_PATH):
    """
    Load real CSV, fix all issues, return clean DataFrame.
    """
    df = pd.read_csv(path)
    print(f"  Raw dataset      : {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Drop rows where hiring_score is missing ───────────
    df = df.dropna(subset=["hiring_score"])
    print(f"  After score clean: {len(df)} rows")

    # ── FIX 2: Compute real employment gap ────────────────
    df["employment_gap_months"] = df.apply(compute_employment_gap, axis=1)

    # ── FIX 2b: Inject gap penalty into hiring_score ───────
    # The original hiring_score was computed BEFORE employment_gap
    # existed, so the model never saw any relationship between
    # gap and score (importance was ~0.0044, basically noise).
    #
    # To make this a real, learnable bias signal, we apply a small
    # penalty to hiring_score proportional to the gap:
    #   -0.02 points per month of gap, capped at -0.6 points
    #
    # This reflects the real-world pattern researched earlier:
    # longer employment gaps (often affecting women more) are
    # penalized by recruitment systems.
    gap_penalty = np.clip(df["employment_gap_months"] * 0.02, 0, 0.6)
    df["hiring_score"] = (df["hiring_score"] - gap_penalty).clip(lower=0)

    # ── Encode bias features ──────────────────────────────
    df["gender_enc"]  = df["gender"].map({"Male": 1, "Female": 0}).fillna(0.5)
    df["tier_enc"]    = df["university_tier"].map(TIER_MAP).fillna(3).astype(int)
    df["utype_enc"]   = df["university_type"].map(UTYPE_MAP).fillna(1).astype(float)
    df["degree_enc"]  = df["degree"].map(DEGREE_MAP).fillna(3).astype(float)
    df["region_enc"]  = df["candidate_region"].map(REGION_MAP).fillna(2).astype(float)

    # ── Encode merit features ─────────────────────────────
    df["cgpa"]              = df["cgpa"].fillna(df["cgpa"].median())
    df["years_experience"]  = df["years_experience"].fillna(0)
    df["skill_count"]       = df["skills"].fillna("").apply(
        lambda x: len([s for s in x.split("|") if s.strip()]) if x else 0)
    df["has_certification"] = df["certifications"].notna().astype(int)
    df["internship_count"]  = df["internship_count"].fillna(0)
    df["projects_count"]    = df["projects_count"].fillna(0)
    df["communication_score"] = df["communication_score"].fillna(
        df["communication_score"].median())
    df["technical_score"]   = df["technical_score"].fillna(
        df["technical_score"].median())
    df["aptitude_score"]    = df["aptitude_score"].fillna(
        df["aptitude_score"].median())

    # ── Print bias stats ──────────────────────────────────
    print(f"\n  ── Bias visible in hiring_score ──────────────")
    print(f"  Male avg score   : {df[df['gender']=='Male']['hiring_score'].mean():.2f}")
    print(f"  Female avg score : {df[df['gender']=='Female']['hiring_score'].mean():.2f}")
    print(f"  Tier-1 avg score : {df[df['tier_enc']==1]['hiring_score'].mean():.2f}")
    print(f"  Tier-2 avg score : {df[df['tier_enc']==2]['hiring_score'].mean():.2f}")
    print(f"  Tier-3 avg score : {df[df['tier_enc']==3]['hiring_score'].mean():.2f}")
    print(f"  Govt   avg score : {df[df['utype_enc']==2]['hiring_score'].mean():.2f}")
    print(f"  Private avg score: {df[df['utype_enc']==1]['hiring_score'].mean():.2f}")
    print(f"  Emp gap (Female) : {df[df['gender']=='Female']['employment_gap_months'].mean():.1f} months")
    print(f"  Emp gap (Male)   : {df[df['gender']=='Male']['employment_gap_months'].mean():.1f} months")
    print(f"  Avg gap penalty applied: {gap_penalty.mean():.3f} points")
    print(f"  hiring_score range after penalty: {df['hiring_score'].min():.2f} - {df['hiring_score'].max():.2f}")

    return df


# ============================================================
# SECTION 3: FEATURE MATRIX
# ============================================================

def get_feature_matrix(df):
    """Returns X (features) and y (hiring_score target)."""
    X = df[FEATURE_NAMES].values.astype(float)
    y = df["hiring_score"].values.astype(float)
    return X, y


# ============================================================
# SECTION 4: TRAIN MODEL
# ============================================================

def train_model(X_train, y_train):
    """
    FIX 1: XGBoost REGRESSOR — predicts hiring_score (2.58–8.4)
    instead of binary classification.
    Gives a full continuous score range for meaningful comparison.
    """
    model = XGBRegressor(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        eval_metric      = "rmse",
        random_state     = 42,
        verbosity        = 0
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
# SECTION 5: SCORE A SINGLE CV
# ============================================================

_GENDER_MAP = {"Male": 1, "Female": 0, "male": 1, "female": 0}
_TIER_MAP   = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3, 1: 1, 2: 2, 3: 3}
_UTYPE_MAP  = {"Government": 2, "Private": 1, "government": 2, "private": 1}
_REGION_MAP = {"Urban": 3, "Semi-Urban": 2, "Rural": 1,
               "urban": 3, "semi-urban": 2, "rural": 1}

# Score bounds from dataset (used for normalization)
SCORE_MIN = 2.58
SCORE_MAX = 8.40


def score_single_cv(cv_dict, model):
    """
    Score ONE candidate CV.

    Args:
        cv_dict : dict — CV fields (see contract below)
        model   : trained XGBoost model from load_model()

    Returns:
        dict:
            raw_score        : float (2.58–8.40 scale, same as dataset)
            normalized_score : float (0.0–1.0 for easy comparison)
            decision         : "SHORTLISTED" or "REJECTED"
            features         : dict of all encoded values used

    ── CV DICT CONTRACT ──────────────────────────────────────
    {
        "gender"               : "Female",
        "university_tier"      : "Tier 3",      # "Tier 1/2/3" or 1/2/3
        "university_type"      : "Private",     # "Government" / "Private"
        "degree"               : "B.Tech",
        "candidate_region"     : "Urban",       # "Urban"/"Semi-Urban"/"Rural"
        "employment_gap_months": 6,             # int/float in months
        "cgpa"                 : 7.5,
        "years_experience"     : 2,
        "skills"               : "Python|SQL|ML",
        "certifications"       : "AWS",         # None if no certification
        "internship_count"     : 1,
        "projects_count"       : 2,
        "communication_score"  : 7.0,
        "technical_score"      : 6.5,
        "aptitude_score"       : 6.0,
    }
    ──────────────────────────────────────────────────────────
    """
    skills_raw  = cv_dict.get("skills", "") or ""
    skill_count = len([s for s in skills_raw.split("|") if s.strip()])

    features = {
        "gender_enc"           : _GENDER_MAP.get(cv_dict.get("gender", "Male"), 1),
        "tier_enc"             : _TIER_MAP.get(cv_dict.get("university_tier", "Tier 3"), 3),
        "utype_enc"            : _UTYPE_MAP.get(cv_dict.get("university_type", "Private"), 1),
        "degree_enc"           : DEGREE_MAP.get(cv_dict.get("degree", "B.Tech"), 3),
        "region_enc"           : _REGION_MAP.get(cv_dict.get("candidate_region", "Urban"), 2),
        "employment_gap_months": float(cv_dict.get("employment_gap_months", 0)),
        "cgpa"                 : float(cv_dict.get("cgpa", 7.0)),
        "years_experience"     : float(cv_dict.get("years_experience", 0)),
        "skill_count"          : skill_count,
        "has_certification"    : 1 if cv_dict.get("certifications") else 0,
        "internship_count"     : float(cv_dict.get("internship_count", 0)),
        "projects_count"       : float(cv_dict.get("projects_count", 0)),
        "communication_score"  : float(cv_dict.get("communication_score", 6.0)),
        "technical_score"      : float(cv_dict.get("technical_score", 6.0)),
        "aptitude_score"       : float(cv_dict.get("aptitude_score", 6.0)),
    }

    X         = np.array([[features[f] for f in FEATURE_NAMES]])
    raw_score = float(model.predict(X)[0])
    raw_score = float(np.clip(raw_score, SCORE_MIN, SCORE_MAX))

    # FIX 3: Normalize to 0.0–1.0
    norm_score = (raw_score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)
    norm_score = round(float(np.clip(norm_score, 0.0, 1.0)), 4)

    # Threshold: above dataset mean (4.92) = shortlisted
    decision = "SHORTLISTED" if raw_score >= 4.92 else "REJECTED"

    return {
        "raw_score"        : round(raw_score, 2),
        "normalized_score" : norm_score,
        "decision"         : decision,
        "features"         : features,
    }


# ============================================================
# SECTION 6: SCORE A FULL BATCH
# Used by Intern 2 (counterfactual) and Intern 3 (graph)
# ============================================================

def score_batch(df_clean, model):
    """
    Score all rows in a cleaned DataFrame at once.
    Adds 3 columns: raw_score, normalized_score, decision
    """
    X, _ = get_feature_matrix(df_clean)
    raw_preds  = model.predict(X)
    raw_preds  = np.clip(raw_preds, SCORE_MIN, SCORE_MAX)
    norm_preds = (raw_preds - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)

    df_out = df_clean.copy()
    df_out["raw_score"]         = raw_preds.round(2)
    df_out["normalized_score"]  = norm_preds.round(4)
    df_out["decision"]          = np.where(raw_preds >= 4.92,
                                            "SHORTLISTED", "REJECTED")
    return df_out


# ============================================================
# SECTION 7: SAVE & LOAD
# ============================================================

def save_model(model, path="model/recruitment_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved → {path}")


def load_model(path="model/recruitment_model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"  Model loaded ← {path}")
    return model


# ============================================================
# SECTION 8: MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  FairGraph-CV: Recruitment Model v2.0 (XGBoost Regressor)")
    print("  Target: hiring_score (continuous 2.58–8.40)")
    print("=" * 60)

    # Step 1: Load & clean
    print("\n[1/5] Loading & cleaning dataset...")
    df = load_and_clean()

    # Step 2: Feature matrix
    print("\n[2/5] Building feature matrix...")
    X, y = get_feature_matrix(df)
    print(f"  Features : {FEATURE_NAMES}")
    print(f"  X shape  : {X.shape}")
    print(f"  Target   : hiring_score — min:{y.min():.2f} mean:{y.mean():.2f} max:{y.max():.2f}")

    # Step 3: Split
    print("\n[3/5] Splitting 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Step 4: Train
    print("\n[4/5] Training XGBoost Regressor...")
    model = train_model(X_train, y_train)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    print(f"\n  MAE (avg error) : {mae:.4f} points")
    print(f"  R² Score        : {r2:.4f}")
    print(f"  Prediction range: {preds.min():.2f} – {preds.max():.2f}")
    print(f"  Actual range    : {y_test.min():.2f} – {y_test.max():.2f}")

    # Feature importance
    importances = pd.Series(
        model.feature_importances_, index=FEATURE_NAMES
    ).sort_values(ascending=False)

    print("\n  Feature Importance (what drives hiring decisions):")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 60)
        tag = " ← BIAS" if feat in [
            "gender_enc", "tier_enc", "utype_enc",
            "degree_enc", "region_enc", "employment_gap_months"
        ] else " ← merit"
        print(f"    {feat:<28} {imp:.4f}  {bar}{tag}")

    # Step 5: Save
    print("\n[5/5] Saving model...")
    save_model(model)

    # ── DEMO ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DEMO: Same Skills — Different Identity")
    print("  Every candidate has identical skills, CGPA, experience")
    print("=" * 60)

    base = {
        "skills"               : "Python|SQL|Machine Learning|React|Cloud",
        "certifications"       : "AWS Certified",
        "cgpa"                 : 7.5,
        "years_experience"     : 3,
        "internship_count"     : 2,
        "projects_count"       : 4,
        "communication_score"  : 7.0,
        "technical_score"      : 7.0,
        "aptitude_score"       : 7.0,
        "employment_gap_months": 0,
    }

    test_cases = [
        {**base,
         "label"           : "Male   | Tier 1 | Govt   | Urban   | 0mo gap",
         "gender"          : "Male",
         "university_tier" : "Tier 1",
         "university_type" : "Government",
         "candidate_region": "Urban",
         "degree"          : "B.Tech"},
        {**base,
         "label"           : "Female | Tier 1 | Govt   | Urban   | 0mo gap",
         "gender"          : "Female",
         "university_tier" : "Tier 1",
         "university_type" : "Government",
         "candidate_region": "Urban",
         "degree"          : "B.Tech"},
        {**base,
         "label"           : "Male   | Tier 3 | Private| Rural   | 0mo gap",
         "gender"          : "Male",
         "university_tier" : "Tier 3",
         "university_type" : "Private",
         "candidate_region": "Rural",
         "degree"          : "BCA"},
        {**base,
         "label"           : "Female | Tier 3 | Private| Rural   | 0mo gap",
         "gender"          : "Female",
         "university_tier" : "Tier 3",
         "university_type" : "Private",
         "candidate_region": "Rural",
         "degree"          : "BCA"},
        {**base,
         "label"           : "Female | Tier 1 | Govt   | Urban   | 6mo gap",
         "gender"          : "Female",
         "university_tier" : "Tier 1",
         "university_type" : "Government",
         "candidate_region": "Urban",
         "degree"          : "B.Tech",
         "employment_gap_months": 6},
        {**base,
         "label"           : "Female | Tier 1 | Govt   | Urban   |12mo gap",
         "gender"          : "Female",
         "university_tier" : "Tier 1",
         "university_type" : "Government",
         "candidate_region": "Urban",
         "degree"          : "B.Tech",
         "employment_gap_months": 12},
    ]

    print(f"\n{'Candidate Profile':<52} {'Raw':>5}  {'0-1':>5}  {'Decision'}")
    print("-" * 75)

    for case in test_cases:
        label  = case.pop("label")
        result = score_single_cv(case, model)
        raw    = result["raw_score"]
        norm   = result["normalized_score"]
        dec    = result["decision"]
        status = "✅" if dec == "SHORTLISTED" else "❌"

        filled = "█" * int(norm * 30)
        empty  = "░" * (30 - int(norm * 30))

        print(f"{label:<52} {raw:>5.2f}  {norm:>5.2f}  {status} {dec}")
        print(f"  [{filled}{empty}]")
        print()

    print("=" * 60)
    print("  HOW INTERNS USE THIS MODULE:")
    print()
    print("  from recruitment_model import load_model, score_single_cv")
    print("  model  = load_model()")
    print("  result = score_single_cv(cv_dict, model)")
    print()
    print("  result['raw_score']         # e.g. 6.84  (2.58–8.40 scale)")
    print("  result['normalized_score']  # e.g. 0.74  (0.0–1.0 scale)")
    print("  result['decision']          # SHORTLISTED or REJECTED")
    print("=" * 60)
