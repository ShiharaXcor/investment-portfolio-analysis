from typing import Tuple

BANNED_KEYWORDS = ["hack","password","exploit","bomb","terror","kill","suicide","sex","porn","child","malware","phish","crime"]
SECURITY_KEYWORDS = ["private key","secret key","stolen","password","ssh key","credentials","api key","access token"]
FINANCIAL_ADVICE_KEYWORDS = ["guarantee","sure","surefire","no risk","get rich","investment guarantee"]

def is_allowed_query(user_query: str) -> Tuple[bool,str]:
    q = user_query.lower()
    for word in BANNED_KEYWORDS:
        if word in q: return False,"This request appears harmful or illegal."
    for word in SECURITY_KEYWORDS:
        if word in q: return False,"Cannot provide security-sensitive information."
    return True,""

def post_process_answer(user_query:str, ai_answer:str) -> str:
    q = user_query.lower()
    if any(term in q for term in ["buy","sell","should i","invest","investment","portfolio recommendation"]):
        disclaimer = "\n\n⚠️ Disclaimer: This assistant provides data-driven analysis for educational purposes. It is not financial advice."
        return ai_answer+disclaimer
    for term in FINANCIAL_ADVICE_KEYWORDS:
        if term in q:
            return "I cannot guarantee investment outcomes. I provide analysis and context, not guarantees."
    return ai_answer
