from __future__ import annotations
import os, json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
RANK_CSV = PROCESSED_DIR / "risk_ranking.csv"
REPORT_JSON = PROCESSED_DIR / "risk_report.json"

# Load environment variables
load_dotenv(BASE_DIR / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Determine risk level and investment advice based on thresholds
def classify_risk(var_1d_95_pct, vol_annual_pct, beta, score):
    if score >= 0.8 or var_1d_95_pct > 0.03 or vol_annual_pct > 0.4:
        return "High", "Not Best: High volatility and risk"
    elif score >= 0.5 or var_1d_95_pct > 0.02 or vol_annual_pct > 0.25:
        return "Medium", "Best for moderate-risk portfolios"
    else:
        return "Low", "Best: Low volatility and risk"

# Build advanced prompt for AI explanation
def build_prompt(df: pd.DataFrame) -> str:
    header = (
        "You are a senior risk analyst. Using the risk metrics below, provide a 1-2 sentence explanation "
        "for why each stock has its assigned RiskLevel and InvestmentAdvice. Follow these rules strictly:\n\n"
        "1. Return strict JSON only, as a list of objects, one object per stock:\n"
        "[{\"Symbol\": \"TSLA\", \"Explanation\": \"Reason here\"}, ...]\n"
        "2. Do not add any text outside JSON.\n"
        "3. Match symbols exactly as provided (case-sensitive).\n"
        "4. Explanation should be practical, concise, and mention VaR, Volatility, Beta, and RiskScore.\n"
        "5. Mention extreme metrics if present.\n"
        "6. Ensure every stock has an explanation.\n"
        "7. Do not use markdown, bullet points, or lists. Only JSON.\n\n"
        "Data:"
    )
    lines = [header]
    for _, r in df.iterrows():
        risk_level, advice = r.RiskLevel, r.InvestmentAdvice
        lines.append(
            f"{r.symbol}: VaR1d95={r.var_1d_95_pct*100:.2f}%, Vol={r.vol_annual_pct*100:.1f}%, "
            f"Beta={r.beta:.2f}, Score={r.risk_score:.2f}, "
            f"RiskLevel={risk_level}, InvestmentAdvice={advice}"
        )
    return "\n".join(lines)

# Main function
def main():
    if not RANK_CSV.exists():
        print("risk_ranking.csv not found. Run: python backend/models/risk.py")
        return

    df = pd.read_csv(RANK_CSV)

    if not OPENAI_API_KEY:
        REPORT_JSON.write_text(
            json.dumps({"summary": "⚠ No OPENAI_API_KEY set. Skipping AI risk summary."}, indent=2)
        )
        print("No OPENAI_API_KEY. Wrote placeholder.")
        return

    # Add deterministic risk classification
    df["RiskLevel"], df["InvestmentAdvice"] = zip(*df.apply(
        lambda r: classify_risk(r.var_1d_95_pct, r.vol_annual_pct, r.beta, r.risk_score), axis=1
    ))

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = build_prompt(df)

    explanations = {}
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,  # deterministic output for JSON reliability
            messages=[
                {"role": "system", "content": "You are concise, accurate, and practical."},
                {"role": "user", "content": prompt},
            ],
        )
        explanation_text = resp.choices[0].message.content.strip()
        
        # Parse AI JSON list
        ai_list = json.loads(explanation_text)
        for item in ai_list:
            symbol = item.get("Symbol")
            explanation = item.get("Explanation", "No explanation")
            if symbol:
                explanations[symbol] = explanation

    except json.JSONDecodeError:
        print("⚠ AI did not return valid JSON. Please check the prompt and model output.")
    except Exception as e:
        print(f"⚠ ChatGPT error: {e}")

    # Build final output
    output = {}
    for _, r in df.iterrows():
        output[r.symbol] = {
            "VaR1d95": round(r.var_1d_95_pct*100, 2),
            "Volatility": round(r.vol_annual_pct*100, 1),
            "Beta": round(r.beta, 2),
            "RiskScore": round(r.risk_score, 2),
            "RiskLevel": r.RiskLevel,
            "InvestmentAdvice": r.InvestmentAdvice,
            "Explanation": explanations.get(r.symbol, "No explanation")
        }

    REPORT_JSON.write_text(json.dumps(output, indent=2))
    print(f"Saved AI risk summary → {REPORT_JSON}")

if __name__ == "__main__":
    main()
