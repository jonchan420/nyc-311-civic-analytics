"""Who files fewer housing complaints than their housing conditions predict?

Unit: NYC Health's Modified ZCTAs (177 neighborhoods). Outcome: HPD
housing-maintenance complaints (heat, plumbing, mold, leaks, ...) per
1,000 renter households per year, 2022-2024. Housing complaints are used
because tenants file them about their own building, so the complaint
lands in the ZIP where the reporter lives — unlike street or noise
complaints, which commuters file wherever they happen to be.

Two models, both OLS on log(rate) with HC3 robust standard errors:
  1. Need model: housing risk only — share of renter units built before
     1950, poverty rate, overcrowding, share of renters in 1-4 unit
     buildings, borough. Its prediction is the "expected" rate; the gap
     is how far each neighborhood's actual rate falls from it.
  2. Full model: need + reporting barriers — limited-English households
     by language group, residents 65+, households with no internet.
     Barrier coefficients show how complaint rates differ between
     neighborhoods with similar housing risk.

Everything here is association, not causation: a neighborhood can file
fewer complaints because conditions really are better in ways the
Census can't see, not only because residents face barriers.

Run (after fetch_reporting_gap_data.py):
    python src/reporting_gap_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import DATA_DIR, VISUAL_DIR

EQUITY_DIR = DATA_DIR / "equity"
RESULTS_DIR = EQUITY_DIR / "results"
MAP_PATH = VISUAL_DIR / "reporting_gap_map.png"
YEARS = 3
MIN_RENTER_HOUSEHOLDS = 1_000
ROBUSTNESS_MIN_RENTER_HOUSEHOLDS = 2_500

NEED = ["pct_pre1950", "pct_poverty", "pct_overcrowded", "pct_small_building"]
BARRIERS = ["pct_lep_spanish", "pct_lep_chinese", "pct_lep_other_asian", "pct_lep_other",
            "pct_age_65_plus", "pct_no_internet"]
LABELS = {
    "pct_pre1950": "Renter units built before 1950",
    "pct_poverty": "Residents below poverty line",
    "pct_overcrowded": "Renter units with >1 person per room",
    "pct_small_building": "Renters in 1-4 unit buildings",
    "pct_lep_spanish": "Limited-English households: Spanish",
    "pct_lep_chinese": "Limited-English households: Chinese",
    "pct_lep_other_asian": "Limited-English households: other Asian languages",
    "pct_lep_other": "Limited-English households: all other languages",
    "pct_age_65_plus": "Residents 65+",
    "pct_no_internet": "Households with no internet access",
    "pct_income_150k": "Households earning $150k+",
}
# Added after the first results, because several of the largest shortfalls
# were affluent areas (Financial District, Battery Park City): checks
# whether the barrier findings are just an income effect in disguise.
INCOME_CHECK = ["pct_income_150k"]


def build_neighborhoods() -> pd.DataFrame:
    crosswalk = pd.read_csv(EQUITY_DIR / "modzcta_crosswalk.csv", dtype=str)
    acs = pd.read_csv(EQUITY_DIR / "acs_2024_nyc_zcta.csv", dtype={"zcta": str})
    complaints = pd.read_csv(EQUITY_DIR / "hpd_complaints_2022_2024_by_zip.csv",
                             dtype={"incident_zip": str})

    names = crosswalk.drop_duplicates("modzcta").set_index("modzcta")["neighborhood"]
    census = acs.merge(crosswalk[["zcta", "modzcta"]], on="zcta").groupby("modzcta").sum(numeric_only=True)

    complaints = complaints.merge(crosswalk[["zcta", "modzcta"]], left_on="incident_zip", right_on="zcta")
    totals = complaints.groupby("modzcta")["complaints"].sum()
    phone = complaints[complaints["channel"] == "PHONE"].groupby("modzcta")["complaints"].sum()
    borough = (complaints[complaints["borough"].isin(
        ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"])]
        .groupby(["modzcta", "borough"])["complaints"].sum()
        .reset_index().sort_values("complaints").drop_duplicates("modzcta", keep="last")
        .set_index("modzcta")["borough"])

    df = census.copy()
    df["neighborhood"] = names
    df["borough"] = borough
    df["complaints_2022_2024"] = totals.reindex(df.index).fillna(0).astype(int)
    df["pct_phone"] = 100 * phone.reindex(df.index).fillna(0) / df["complaints_2022_2024"].where(
        df["complaints_2022_2024"] > 0)

    def pct(numerator: str, denominator: str) -> pd.Series:
        return 100 * df[numerator] / df[denominator].where(df[denominator] > 0)

    df["pct_pre1950"] = pct("renter_units_pre1950", "renter_units_year_built_universe")
    df["pct_poverty"] = pct("below_poverty", "poverty_universe")
    df["pct_overcrowded"] = pct("renter_units_overcrowded", "renter_units_occupancy_universe")
    df["pct_small_building"] = pct("renter_units_1_to_4", "renter_units_structure_universe")
    for group in ["spanish", "chinese", "other_asian", "other"]:
        df[f"pct_lep_{group}"] = pct(f"lep_{group}", "households")
    df["pct_lep_any"] = (df["pct_lep_spanish"] + df["pct_lep_chinese"]
                         + df["pct_lep_other_asian"] + df["pct_lep_other"])
    df["pct_age_65_plus"] = pct("age_65_plus", "age_universe")
    df["pct_no_internet"] = pct("no_internet", "internet_universe")
    df["pct_income_150k"] = pct("income_150k_plus", "income_universe")
    # several MODZCTAs share a neighborhood name (three "Financial District"s)
    df["label"] = df["neighborhood"] + " (" + df.index + ")"
    df["complaints_per_1000_renters"] = (
        1000 * df["complaints_2022_2024"] / YEARS / df["renter_households"].where(df["renter_households"] > 0))
    return df.reset_index()


def design(df: pd.DataFrame, terms: list[str], borough_effects: bool) -> pd.DataFrame:
    X = df[terms].copy()
    if borough_effects:
        # reference = first borough alphabetically present in the sample
        X = X.join(pd.get_dummies(df["borough"], prefix="borough", dtype=float, drop_first=True))
    X.insert(0, "const", 1.0)
    return X


def ols_hc3(y: pd.Series, X: pd.DataFrame) -> dict:
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    n, k = Xv.shape
    xtx_inv = np.linalg.inv(Xv.T @ Xv)
    beta = xtx_inv @ Xv.T @ yv
    resid = yv - Xv @ beta
    leverage = np.einsum("ij,jk,ik->i", Xv, xtx_inv, Xv)
    meat = Xv.T @ (Xv * (resid ** 2 / (1 - leverage) ** 2)[:, None])
    se = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
    df_resid = n - k
    t = beta / se
    crit = stats.t.ppf(0.975, df_resid)
    r2 = 1 - (resid ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    table = pd.DataFrame({
        "term": X.columns, "coef": beta, "se_hc3": se,
        "ci_low": beta - crit * se, "ci_high": beta + crit * se,
        "p_value": 2 * stats.t.sf(np.abs(t), df_resid),
    })
    return {"table": table, "fitted": Xv @ beta, "r2": r2, "n": n}


def vif(X: pd.DataFrame, terms: list[str]) -> dict[str, float]:
    out = {}
    for term in terms:
        others = X.drop(columns=term).to_numpy(float)
        target = X[term].to_numpy(float)
        coef, *_ = np.linalg.lstsq(others, target, rcond=None)
        resid = target - others @ coef
        r2 = 1 - (resid ** 2).sum() / ((target - target.mean()) ** 2).sum()
        out[term] = 1 / (1 - r2)
    return out


def fit(df: pd.DataFrame, name: str, terms: list[str], outcome: str = "log_rate",
        borough_effects: bool = True) -> tuple[pd.DataFrame, dict]:
    X = design(df, terms, borough_effects)
    result = ols_hc3(df[outcome], X)
    table = result["table"]
    table.insert(0, "model", name)
    table["n"] = result["n"]
    table["r2"] = result["r2"]
    vifs = vif(X, terms)
    table["vif"] = table["term"].map(vifs)
    if outcome == "log_rate":
        # percent change in complaint rate for +10 percentage points
        table["pct_change_per_10pts"] = 100 * (np.exp(10 * table["coef"]) - 1)
    else:
        table["pct_change_per_10pts"] = np.nan
    table["label"] = table["term"].map(LABELS)
    return table, result


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    neighborhoods = build_neighborhoods()
    complete = neighborhoods.dropna(subset=NEED + BARRIERS + ["complaints_per_1000_renters", "borough"])
    included = complete[(complete["renter_households"] >= MIN_RENTER_HOUSEHOLDS)
                        & (complete["complaints_2022_2024"] > 0)].copy()
    excluded = neighborhoods[~neighborhoods["modzcta"].isin(included["modzcta"])]
    print(f"{len(included)} of {len(neighborhoods)} neighborhoods analyzed; excluded "
          f"{len(excluded)} with < {MIN_RENTER_HOUSEHOLDS:,} renter households or missing data: "
          f"{', '.join(excluded['neighborhood'].fillna(excluded['modzcta']).head(12))}"
          f"{' ...' if len(excluded) > 12 else ''}")
    included["log_rate"] = np.log(included["complaints_per_1000_renters"])

    need_table, need_fit = fit(included, "need_only", NEED)
    income_need_table, income_need_fit = fit(included, "need_plus_income", NEED + INCOME_CHECK)
    full_table, _ = fit(included, "need_plus_barriers", NEED + BARRIERS)
    phone_sample = included.dropna(subset=["pct_phone"])
    phone_table, _ = fit(phone_sample, "phone_share", NEED + BARRIERS, outcome="pct_phone")
    coefficients = pd.concat([need_table, income_need_table, full_table, phone_table], ignore_index=True)

    # robustness: re-fit the full model under alternative samples/specifications
    checks = [full_table.assign(check="main", dropped="")]
    robust = included[included["renter_households"] >= ROBUSTNESS_MIN_RENTER_HOUSEHOLDS]
    checks.append(fit(robust, f"min_{ROBUSTNESS_MIN_RENTER_HOUSEHOLDS}_renters", NEED + BARRIERS)[0]
                  .assign(dropped=""))
    checks.append(fit(included, "no_borough_effects", NEED + BARRIERS, borough_effects=False)[0]
                  .assign(dropped=""))
    checks.append(fit(included, "plus_income_150k", NEED + INCOME_CHECK + BARRIERS)[0].assign(dropped=""))
    for borough in sorted(included["borough"].unique()):
        checks.append(fit(included[included["borough"] != borough], "drop_one_borough",
                          NEED + BARRIERS)[0].assign(dropped=borough))
    for _, row in included.nlargest(10, "pct_lep_chinese").iterrows():
        checks.append(fit(included[included["modzcta"] != row["modzcta"]],
                          "drop_one_high_chinese_lep_neighborhood", NEED + BARRIERS)[0]
                      .assign(dropped=row["label"]))
    robustness = pd.concat(checks, ignore_index=True)
    robustness["check"] = robustness["check"].fillna(robustness["model"])
    robustness = robustness[robustness["term"].isin(BARRIERS + INCOME_CHECK)].assign(
        ci_low_pct=lambda d: 100 * (np.exp(10 * d["ci_low"]) - 1),
        ci_high_pct=lambda d: 100 * (np.exp(10 * d["ci_high"]) - 1),
    )[["check", "dropped", "term", "label", "pct_change_per_10pts", "ci_low_pct", "ci_high_pct",
       "p_value", "n"]]

    for name, fitted in [("", need_fit), ("_income_adjusted", income_need_fit)]:
        included[f"expected_per_1000_renters{name}"] = np.exp(fitted["fitted"])
        included[f"gap_pct{name}"] = 100 * (included["complaints_per_1000_renters"]
                                            / included[f"expected_per_1000_renters{name}"] - 1)
    gap_cols = ["expected_per_1000_renters", "gap_pct",
                "expected_per_1000_renters_income_adjusted", "gap_pct_income_adjusted"]
    out = neighborhoods.merge(included[["modzcta"] + gap_cols], on="modzcta", how="left")
    out["analyzed"] = out["modzcta"].isin(included["modzcta"])
    keep = (["modzcta", "neighborhood", "label", "borough", "analyzed", "renter_households",
             "complaints_2022_2024", "complaints_per_1000_renters"] + gap_cols
            + ["pct_phone", "pct_lep_any"] + NEED + BARRIERS + INCOME_CHECK)
    out = out[keep].sort_values("gap_pct")
    out.round(2).to_csv(RESULTS_DIR / "neighborhood_reporting_gap.csv", index=False)
    coefficients.round(6).to_csv(RESULTS_DIR / "model_coefficients.csv", index=False)
    robustness.round(4).to_csv(RESULTS_DIR / "robustness_checks.csv", index=False)

    pd.set_option("display.width", 220)
    for model in ["need_only", "need_plus_income", "need_plus_barriers"]:
        t = coefficients[(coefficients["model"] == model) & coefficients["label"].notna()]
        print(f"\n=== {model} (n={t['n'].iloc[0]}, R^2={t['r2'].iloc[0]:.2f})")
        print(t[["label", "pct_change_per_10pts", "ci_low", "ci_high", "p_value", "vif"]]
              .assign(ci_low=lambda d: 100 * (np.exp(10 * d["ci_low"]) - 1),
                      ci_high=lambda d: 100 * (np.exp(10 * d["ci_high"]) - 1))
              .round(2).to_string(index=False))

    print("\n=== robustness of each barrier (% change per +10 pts) across all re-fits")
    summary = robustness[robustness["term"].isin(BARRIERS)].groupby("label").agg(
        main=("pct_change_per_10pts", "first"),
        lowest=("pct_change_per_10pts", "min"), highest=("pct_change_per_10pts", "max"),
        fits=("pct_change_per_10pts", "size"),
        negative=("pct_change_per_10pts", lambda s: int((s < 0).sum())),
        significant=("p_value", lambda s: int((s < 0.05).sum())),
    )
    print(summary.round(1).to_string())
    print("\nfits where the effect is not significant (p >= 0.05):")
    weak = robustness[robustness["term"].isin(["pct_lep_chinese", "pct_lep_other", "pct_age_65_plus"])
                      & (robustness["p_value"] >= 0.05)]
    print(weak[["label", "check", "dropped", "pct_change_per_10pts", "p_value"]].round(3).to_string(index=False))

    print("\n=== phone share of complaints (percentage-point change per +10 pts)")
    phone = coefficients[(coefficients["model"] == "phone_share") & coefficients["term"].isin(BARRIERS)]
    print(phone.assign(change_per_10=10 * phone["coef"])[["label", "change_per_10", "p_value"]]
          .round(3).to_string(index=False))

    cols = ["label", "complaints_per_1000_renters", "expected_per_1000_renters", "gap_pct",
            "gap_pct_income_adjusted", "pct_lep_spanish", "pct_lep_chinese", "pct_lep_other",
            "pct_income_150k"]
    shown = out[out["analyzed"]]
    print("\n=== largest shortfalls vs. housing-risk expectation")
    print(shown[cols].head(12).round(1).to_string(index=False))
    print("\n=== largest shortfalls once income is also accounted for")
    print(shown.sort_values("gap_pct_income_adjusted")[cols].head(12).round(1).to_string(index=False))
    print("\n=== neighborhoods with the most Chinese limited-English households")
    print(shown.sort_values("pct_lep_chinese", ascending=False)[cols].head(10).round(1).to_string(index=False))
    save_map(out)
    print(f"\nSaved to {RESULTS_DIR} and {MAP_PATH}")


def save_map(out: pd.DataFrame) -> None:
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Polygon

    boundaries = json.loads((EQUITY_DIR / "modzcta_boundaries.geojson").read_text())
    gap = out.set_index("modzcta")["gap_pct"]
    cmap, norm = plt.get_cmap("RdBu"), Normalize(-100, 100)
    fig, ax = plt.subplots(figsize=(8, 8.6))
    for feature in boundaries["features"]:
        value = gap.get(feature["properties"]["modzcta"])
        color = "#d9d9d9" if pd.isna(value) else cmap(norm(max(-100, min(100, value))))
        for polygon in feature["geometry"]["coordinates"]:
            ax.add_patch(Polygon(polygon[0], closed=True, facecolor=color,
                                 edgecolor="white", linewidth=0.3))
    labels = {  # name: (point inside the neighborhood, where the text goes)
        "Chinatown": ((-73.988, 40.715), (-74.20, 40.735)),
        "Sunset Park": ((-74.010, 40.645), (-74.24, 40.665)),
        "Borough Park": ((-73.990, 40.634), (-74.24, 40.640)),
        "Bensonhurst": ((-73.996, 40.611), (-74.22, 40.600)),
        "Flushing": ((-73.828, 40.760), (-73.80, 40.830)),
    }
    for name, (point, text) in labels.items():
        ax.annotate(name, point, xytext=text, fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "#333333", "lw": 0.6})
    ax.set_xlim(-74.27, -73.69)
    ax.set_ylim(40.49, 40.92)
    ax.set_aspect(1 / np.cos(np.radians(40.7)))
    ax.axis("off")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.01)
    bar.set_label("Housing complaints vs. what housing conditions predict (%)")
    ax.set_title("Who files fewer housing complaints than expected?\n"
                 "HPD complaints per renter household, 2022–2024 (red = fewer than expected)",
                 fontsize=11)
    fig.savefig(MAP_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
