"""Agent 4 - Customer Success Agent: LLM drafts, deterministic guardrails validate.
The LLM only receives *facts already decided* by the upstream agents. It cannot invent dates or offers.
Output is validated; if the LLM fails validation twice we fall back to a template (never send junk)."""
import json, re
from . import config as C

PT_MONTHS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
BANNED = ["guarantee", "100%", "garantimos", "legal action", "your fault", "ação judicial"]


def fmt_date(ts, lang):
    return f"{ts.day} {PT_MONTHS[ts.month-1]} {ts.year}" if lang == "pt" else ts.strftime("%d %b %Y").lstrip("0")


def build_facts(r, d, lang):
    return {"order_ref": r.order_ref, "product": " & ".join(c.replace("_", " ") for c in r.categories[:2]),
            "order_date": fmt_date(r.order_purchase_timestamp, lang), "original_eta": fmt_date(r.order_estimated_delivery_date, lang),
            "new_eta": fmt_date(r.new_eta, lang), "action": r.action, "residual_delay_days": int(r.residual_breach),
            "voucher_brl": float(r.voucher_brl), "can_refund": r.action == "CANCEL_OFFER",
            "rerouted": r.action == "REROUTE", "cause": d.cause, "origin": d.region_label(),
            "already_late": bool(r.already_late),}


TPL = {
 "en": {"subj": "Update on your order #{order_ref}: new delivery date {new_eta}",
        "hi": "Hello,\n\n",
        "sit": "We're writing about your {product} order #{order_ref}, placed on {order_date}. Because of {cause} affecting shipments from {origin}, we're sorry to say it is running late. We know how frustrating that is.\n\n",
        "REROUTE": "To limit the delay, we're sending it through an alternate route at no extra cost to you. Your new estimated delivery date is {new_eta} (originally {original_eta}).\n\n",
        "DELAY": "Your new estimated delivery date is {new_eta} (originally {original_eta}). Your order is safe and will move as soon as routes reopen.\n\n",
        "CANCEL_OFFER": "Your new estimated delivery date is {new_eta} (originally {original_eta}). If that no longer works for you, just reply to this email and we'll issue a full refund - no questions asked.\n\n",
        "V": "As an apology, we've added a R$ {voucher_brl:.2f} voucher to your account for your next purchase.\n\n",
        "bye": "Thank you for your patience - we'll keep you updated if anything changes.\n\nWarm regards,\nCustomer Care Team"},
 "pt": {"subj": "Atualização do seu pedido #{order_ref}: nova data de entrega {new_eta}",
        "hi": "Olá,\n\n",
        "sit": "Escrevemos sobre o seu pedido #{order_ref} ({product}), feito em {order_date}. Devido a {cause} que afetam os envios de {origin}, infelizmente ele sofrerá atraso. Sabemos como isso é frustrante.\n\n",
        "REROUTE": "Para reduzir o atraso, estamos enviando por uma rota alternativa, sem custo adicional para você. A nova data estimada é {new_eta} (antes: {original_eta}).\n\n",
        "DELAY": "A nova data estimada de entrega é {new_eta} (antes: {original_eta}). Seu pedido está seguro e seguirá assim que as rotas forem liberadas.\n\n",
        "CANCEL_OFFER": "A nova data estimada de entrega é {new_eta} (antes: {original_eta}). Se não for mais conveniente, responda este e-mail e faremos o reembolso integral.\n\n",
        "V": "Como pedido de desculpas, adicionamos um voucher de R$ {voucher_brl:.2f} à sua conta para a próxima compra.\n\n",
        "bye": "Obrigado pela paciência - avisaremos se algo mudar.\n\nAtenciosamente,\nEquipe de Atendimento"}}

SIT_LATE = {
 "en": "We're writing about your {product} order #{order_ref}, placed on {order_date}. It was already running behind schedule, and {cause} affecting shipments from {origin} will delay it further. We're truly sorry - we know how frustrating that is.\n\n",
 "pt": "Escrevemos sobre o seu pedido #{order_ref} ({product}), feito em {order_date}. Ele já estava atrasado e {cause} que afetam os envios de {origin} causarão atraso adicional. Sentimos muito - sabemos como isso é frustrante.\n\n"}


def template_email(f, lang):
    t = TPL[lang]
    sit = SIT_LATE[lang] if f["already_late"] else t["sit"]
    body = t["hi"] + sit + t[f["action"]] + (t["V"] if f["voucher_brl"] > 0 else "") + t["bye"]    
    return {"subject": t["subj"].format(**f), "body": body.format(**f)}


def validate(email, f):
    """Deterministic guardrails. Returns a list of problems (empty = pass)."""
    b, errs = email["body"], []
    if f["order_ref"] not in b + email["subject"]: errs.append("missing order reference")
    if f["new_eta"] not in b: errs.append("missing new delivery date")
    if f["voucher_brl"] > 0 and f"{f['voucher_brl']:.2f}".replace(".", ",") not in b.replace(".", ",") : errs.append("voucher amount missing/wrong")
    if f["voucher_brl"] == 0 and re.search(r"voucher|R\$", b): errs.append("offers something not approved")
    if f["can_refund"] and not re.search(r"refund|reembolso", b, re.I): errs.append("missing refund option")
    if not f["can_refund"] and re.search(r"refund|reembolso", b, re.I): errs.append("offers refund not approved")
    if any(w in b.lower() for w in BANNED): errs.append("banned phrase")
    if not 50 <= len(b.split()) <= 230: errs.append("length out of range")
    if re.search(r"\[[^\]]+\]|\{[^}]+\}", b): errs.append("unfilled placeholder")
    if f["already_late"] and not re.search(r"already|já estava", b, re.I): errs.append("does not acknowledge pre-existing lateness")
    if any(re.search(r"voucher|vale", s, re.I) and re.search(r"reply|respond|responda", s, re.I)
           for s in re.split(r"(?<=[.!?])\s+", b)):
        errs.append("links the voucher to replying (voucher is already applied; reply is only for a refund)")
    if not re.search(r"sorry|apolog|desculp|lamentamos|sentimos|infelizmente", b, re.I): errs.append("no apology")
    if not f["already_late"] and re.search(r"already (running )?(late|behind)|já estava", b, re.I): errs.append("claims order was already late")
    if re.search(r"only way", b, re.I): errs.append("overpromises on how to get a refund")
    return errs


SYSTEM = ("You are a senior customer-care writer for a Brazilian e-commerce marketplace. Write a short, warm, "
          "empathetic email (90-150 words). Rules: use ONLY the facts provided; mention the order reference, the new "
          "delivery date and the reason; apologise once, sincerely, without blaming anyone; state exactly the "
          "compensation given (voucher amount in R$) and nothing else; if can_refund is true, say they can reply to "
          "get a full refund; never promise anything beyond the facts; no marketing language. "
          "if already_late is true, acknowledge the order was already running late and that the disruption adds further delay; do not blame the disruption for the whole delay. "
          "Sign off exactly as 'Customer Care Team'. Never use placeholders like [Your Name]. "
          'Return ONLY JSON: {"subject": "...", "body": "..."}')


def draft(r, d, llm, lang="en"):
    f = build_facts(r, d, lang)
    result = {"order_ref": r.order_ref, "action": r.action, "facts": f, "source": "template", "attempts": 0, "guardrail_issues": []}
    if llm and llm.available:
        prompt = f"Language: {'Brazilian Portuguese' if lang == 'pt' else 'English'}\nFacts:\n{json.dumps(f, ensure_ascii=False)}"
        must = [f"the order reference #{f['order_ref']}", f"the new delivery date {f['new_eta']}"]
        must.append(f"a voucher of exactly R$ {f['voucher_brl']:.2f}" if f["voucher_brl"] > 0
                    else "NO voucher or compensation of any kind")
        must.append("a full-refund option (reply to this email)" if f["can_refund"]
                    else "NO mention of refunds or cancellation")
        if f["voucher_brl"] > 0 and f["can_refund"]:
            must.append("two separate points: the voucher is already applied, and replying to the email is ONLY for requesting a refund")
        must.append("paragraphs separated by a blank line")
        if f["voucher_brl"] > 0:
            must.append("that the voucher has ALREADY been added to their account (no action needed to claim it)")
        if f["already_late"]:
            must.append("an acknowledgement that the order was already running late before the disruption")
        prompt += "\nThe email MUST include: " + "; ".join(must)
        for attempt in (1, 2):
            result["attempts"] = attempt
            out = llm.complete(SYSTEM, prompt + (f"\nFix these problems from your last attempt: {errs}" if attempt > 1 else ""), max_tokens=500)
            try:
                email = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
                errs = validate(email, f)
            except Exception:
                errs = ["unparseable output"]
            if not errs:
                return {**result, **email, "source": f"llm:{llm.name}", "guardrail_issues": []}
            result["guardrail_issues"] = errs
    email = template_email(f, lang)
    assert not validate(email, f), validate(email, f)      # templates must always pass our own guardrails
    return {**result, **email}


def pick_samples(df, n=2):
    """Show variety: highest-severity order of each action type first, then fill by severity."""
    df = df[df.action != "NO_ACTION"].sort_values("severity", ascending=False)
    picks = [g.index[0] for _, g in df.groupby("action", sort=False)]
    picks = sorted(picks, key=lambda i: -df.loc[i, "severity"])[:n]
    picks += [i for i in df.index if i not in picks][:max(0, n - len(picks))]
    return df.loc[picks]
