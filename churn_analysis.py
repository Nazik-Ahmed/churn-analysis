#!/usr/bin/env python3
"""
تحليل انسحاب المشتركين (churn) - بيانات غير متوازنة (~9.3% ملغون)

الاستخدام:
    python3 churn_analysis.py --data subscriptions-churn.csv --out results/

المخرجات: results.json + جداول CSV. كل الأرقام في التقرير تُقرأ من هذه المخرجات.

المنهج باختصار:
  1) وصف إحصائي: نسب الإلغاء لكل فئة مع فواصل ثقة (Wilson / Newcombe) + اختبارات مع تصحيح Holm.
  2) نموذج لوجستي متعدد المتغيرات (أخطاء معيارية من مصفوفة هسّيان) + أثر هامشي متوسط (AME) بفاصل bootstrap.
  3) ثبات الترتيب: كم مرة يقع كل عامل ضمن الأقوى ثلاثة عبر عيّنات bootstrap.
  4) تقييم تنبؤي بتحقق متقاطع متكرر (5 طيّات x 10 مرات) بمقاييس تناسب عدم التوازن:
     PR-AUC و ROC-AUC و Brier و recall/precision عند نسبة تنبيه ثابتة و lift.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

SEED = 42
RNG = np.random.default_rng(SEED)

COLS = ["id", "signup", "plan", "price", "cycle", "tenure", "pay",
        "tickets", "sessions", "disc", "onb", "churn"]


# ----------------------------------------------------------------- مساعدات
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe_rd(k1, n1, k2, n2, z=1.96):
    """فرق نسبتين p1-p2 بفاصل Newcombe (Wilson hybrid)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d, d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def logit_fit(X, y, iters=100):
    """انحدار لوجستي بلا عقوبة (Newton-Raphson). يعيد المعاملات والتغاير وlog-likelihood."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ b)
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        g = X.T @ (y - p)
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    p = np.clip(sigmoid(X @ b), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return b, np.linalg.inv(H), ll


def ame(X, b, j):
    """أثر هامشي متوسط لمتغير ثنائي: متوسط P(y|x_j=1) - P(y|x_j=0)."""
    Xa, Xb = X.copy(), X.copy()
    Xa[:, j], Xb[:, j] = 1.0, 0.0
    return float(np.mean(sigmoid(Xa @ b) - sigmoid(Xb @ b)))


# ----------------------------------------------------------------- تحميل
def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = COLS
    df["signup"] = pd.to_datetime(df["signup"])
    df["y"] = (df["churn"] == "نعم").astype(int)
    # الميزات الثنائية (العتبات اختيرت استكشافيًا؛ انظر النموذج ذي العتبات المتداخلة في التحقق)
    df["low_use"] = (df["sessions"] <= 12).astype(int)
    df["many_tix"] = (df["tickets"] >= 5).astype(int)
    df["no_onb"] = (df["onb"] == "لا").astype(int)
    df["monthly"] = (df["cycle"] == "شهري").astype(int)
    df["plan_basic"] = (df["plan"] == "أساسي").astype(int)
    df["pay_transfer"] = (df["pay"] == "تحويل").astype(int)
    df["pay_wallet"] = (df["pay"] == "محفظة").astype(int)
    return df


FEATS = ["low_use", "many_tix", "no_onb", "monthly", "plan_basic"]
LABELS = {
    "low_use": "استخدام منخفض (≤12 جلسة/شهر)",
    "many_tix": "تذاكر دعم كثيرة (≥5)",
    "no_onb": "لم يحضر التهيئة",
    "monthly": "فوترة شهرية",
    "plan_basic": "باقة «أساسي»",
}


# ----------------------------------------------------------------- 1) الوصف
def describe(df):
    out = {}
    n, k = len(df), int(df.y.sum())
    out["n"], out["churned"], out["rate"] = n, k, k / n
    out["rate_ci"] = wilson(k, n)
    out["majority_accuracy"] = 1 - k / n

    rows, pvals = [], []
    cat_specs = [("الباقة", "plan"), ("دورة الفوترة", "cycle"), ("وسيلة الدفع", "pay"),
                 ("حضر التهيئة", "onb")]
    for title, c in cat_specs:
        tab = pd.crosstab(df[c], df.y)
        chi2, p, dof, _ = stats.chi2_contingency(tab, correction=False)
        pvals.append(p)
        for lvl, r in tab.iterrows():
            kk, nn = int(r.get(1, 0)), int(r.sum())
            lo, hi = wilson(kk, nn)
            rows.append(dict(variable=title, level=lvl, n=nn, churned=kk, rate=kk / nn,
                             ci_lo=lo, ci_hi=hi, test="chi2", p=p))
    # متغيرات رقمية
    num_rows = []
    for title, c in [("تذاكر الدعم", "tickets"), ("متوسط الجلسات", "sessions"),
                     ("أشهر البقاء", "tenure"), ("نسبة الخصم", "disc"), ("السعر الشهري", "price")]:
        a, b = df.loc[df.y == 1, c].values, df.loc[df.y == 0, c].values
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc = mw.statistic / (len(a) * len(b))
        # فاصل bootstrap لـ AUC الأحادي ولفرق المتوسطات
        aucs, dms = [], []
        for _ in range(1000):
            ai = RNG.choice(a, len(a)); bi = RNG.choice(b, len(b))
            dms.append(ai.mean() - bi.mean())
            aucs.append(stats.mannwhitneyu(ai, bi).statistic / (len(ai) * len(bi)))
        pvals.append(mw.pvalue)
        num_rows.append(dict(variable=title, mean_churn=a.mean(), mean_stay=b.mean(),
                             median_churn=float(np.median(a)), median_stay=float(np.median(b)),
                             diff=a.mean() - b.mean(),
                             diff_lo=float(np.percentile(dms, 2.5)), diff_hi=float(np.percentile(dms, 97.5)),
                             auc=auc, auc_lo=float(np.percentile(aucs, 2.5)), auc_hi=float(np.percentile(aucs, 97.5)),
                             p=mw.pvalue))
    adj = holm(pvals)
    out["cat_table"] = rows
    out["num_table"] = num_rows
    # تعيين p المصحّحة (الفئوية أولاً ثم الرقمية بنفس ترتيب الإدخال)
    for i, title_c in enumerate([s[0] for s in cat_specs]):
        for r in rows:
            if r["variable"] == title_c:
                r["p_holm"] = adj[i]
    for j, r in enumerate(num_rows):
        r["p_holm"] = adj[len(cat_specs) + j]

    # المعدل عبر عتبات التذاكر والجلسات
    def band(col, bins, labels):
        g = df.groupby(pd.cut(df[col], bins, labels=labels, right=True), observed=True).y.agg(["sum", "count"])
        res = []
        for lvl, r in g.iterrows():
            lo, hi = wilson(r["sum"], r["count"])
            res.append(dict(level=str(lvl), n=int(r["count"]), churned=int(r["sum"]),
                            rate=r["sum"] / r["count"], ci_lo=lo, ci_hi=hi))
        return res
    out["tickets_band"] = band("tickets", [-1, 0, 1, 2, 4, 6, 8, 11],
                               ["0", "1", "2", "3–4", "5–6", "7–8", "9–11"])
    out["sessions_band"] = band("sessions", [0, 5, 10, 12, 15, 20, 30, 40, 50, 60],
                                ["≤5", "6–10", "11–12", "13–15", "16–20", "21–30", "31–40", "41–50", "51–60"])
    # فرق خام لكل ميزة ثنائية
    raw = []
    for f in FEATS:
        a = df[df[f] == 1]; b = df[df[f] == 0]
        d, lo, hi = newcombe_rd(a.y.sum(), len(a), b.y.sum(), len(b))
        ra, rb = a.y.mean(), b.y.mean()
        raw.append(dict(feature=f, label=LABELS[f], n_exposed=len(a), rate_exposed=ra,
                        rate_other=rb, rd=d, rd_lo=lo, rd_hi=hi, rr=ra / rb))
    out["raw_effects"] = raw

    # اتساق التواريخ: نهاية الاشتراك المتوقعة مقابل تاريخ التسليم
    today = pd.Timestamp("2026-10-26")
    end = df.signup + pd.to_timedelta(df.tenure * 30.4, unit="D")
    out["tenure_future_end"] = int((end > today).sum())
    out["signup_range"] = [str(df.signup.min().date()), str(df.signup.max().date())]
    out["corr_tenure_y"] = float(df[["tenure", "y"]].corr().iloc[0, 1])
    out["price_plan_perfect"] = bool(df.groupby("plan").price.nunique().max() == 1)

    # شرائح الإشارات المركّبة
    df = df.assign(nflags=df.low_use + df.many_tix)
    seg = []
    for f_, g in df.groupby("nflags"):
        lo, hi = wilson(g.y.sum(), len(g))
        seg.append(dict(flags=int(f_), n=len(g), churned=int(g.y.sum()), rate=g.y.mean(), ci_lo=lo, ci_hi=hi))
    out["flag_table"] = seg
    flagged = df[df.nflags >= 1]
    unflag = df[df.nflags == 0]
    d, lo, hi = newcombe_rd(flagged.y.sum(), len(flagged), unflag.y.sum(), len(unflag))
    out["segment"] = dict(
        n=len(flagged), share=len(flagged) / len(df), churned=int(flagged.y.sum()),
        rate=flagged.y.mean(), rate_ci=wilson(flagged.y.sum(), len(flagged)),
        captured=flagged.y.sum() / df.y.sum(),
        captured_ci=wilson(flagged.y.sum(), df.y.sum()),
        rate_rest=unflag.y.mean(), rd=d, rd_lo=lo, rd_hi=hi,
        lift=flagged.y.mean() / df.y.mean(),
        mrr_churned=float(df.loc[df.y == 1, "price"].sum()),
        mrr_churned_flagged=float(flagged.loc[flagged.y == 1, "price"].sum()))
    return out


# ----------------------------------------------------------------- 2) النموذج
def design(df, feats):
    return np.column_stack([np.ones(len(df))] + [df[f].values.astype(float) for f in feats])


def main_model(df, B=2000):
    X, y = design(df, FEATS), df.y.values.astype(float)
    b, cov, ll = logit_fit(X, y)
    se = np.sqrt(np.diag(cov))
    ll0 = logit_fit(X[:, :1], y)[2]
    res = dict(loglik=ll, loglik_null=ll0, mcfadden_r2=1 - ll / ll0,
               lr_chi2=2 * (ll - ll0), lr_df=len(FEATS), lr_p=float(stats.chi2.sf(2 * (ll - ll0), len(FEATS))))
    # ملاحظة: LR لكل عامل (حذف واحد)
    coefs = []
    for j, f in enumerate(FEATS, start=1):
        keep = [i for i in range(X.shape[1]) if i != j]
        ll_r = logit_fit(X[:, keep], y)[2]
        lr = 2 * (ll - ll_r)
        coefs.append(dict(feature=f, label=LABELS[f], beta=b[j], se=se[j],
                          OR=float(np.exp(b[j])), OR_lo=float(np.exp(b[j] - 1.96 * se[j])),
                          OR_hi=float(np.exp(b[j] + 1.96 * se[j])),
                          z=b[j] / se[j], p=float(2 * stats.norm.sf(abs(b[j] / se[j]))),
                          lr_chi2=lr, lr_p=float(stats.chi2.sf(lr, 1)),
                          ame=ame(X, b, j)))
    # bootstrap لـ AME وثبات الترتيب
    n = len(df)
    ames = np.zeros((B, len(FEATS)))
    for t in range(B):
        idx = RNG.integers(0, n, n)
        bb, _, _ = logit_fit(X[idx], y[idx])
        for j in range(len(FEATS)):
            ames[t, j] = ame(X[idx], bb, j + 1)
    ranks = (-ames).argsort(axis=1).argsort(axis=1) + 1   # 1 = الأقوى
    for j, c in enumerate(coefs):
        c["ame_lo"], c["ame_hi"] = float(np.percentile(ames[:, j], 2.5)), float(np.percentile(ames[:, j], 97.5))
        c["p_top3"] = float((ranks[:, j] <= 3).mean())
        c["p_rank1"] = float((ranks[:, j] == 1).mean())
        c["median_rank"] = float(np.median(ranks[:, j]))
    res["coefs"] = coefs
    res["intercept"] = float(b[0])

    # عوامل أخرى: هل تضيف شيئًا؟ (دفع، خصم، أشهر البقاء)
    ext = FEATS + ["pay_transfer", "pay_wallet"]
    Xe = np.column_stack([design(df, ext), df.disc.values / 10.0, df.tenure.values / 12.0])
    be, ce, lle = logit_fit(Xe, y)
    lr = 2 * (lle - ll)
    res["extras_test"] = dict(df=4, chi2=lr, p=float(stats.chi2.sf(lr, 4)),
                              note="دفع (2) + خصم + أشهر البقاء")
    se_e = np.sqrt(np.diag(ce))
    names = ["pay_transfer", "pay_wallet", "disc_per10pct", "tenure_per12m"]
    res["extras"] = [dict(name=nm, OR=float(np.exp(be[len(FEATS) + 1 + i])),
                          lo=float(np.exp(be[len(FEATS) + 1 + i] - 1.96 * se_e[len(FEATS) + 1 + i])),
                          hi=float(np.exp(be[len(FEATS) + 1 + i] + 1.96 * se_e[len(FEATS) + 1 + i])))
                     for i, nm in enumerate(names)]
    return res


# ----------------------------------------------------------------- 3) التحقق
def fit_logit_pred(Xtr, ytr, Xte):
    b, _, _ = logit_fit(Xtr, ytr)
    return sigmoid(Xte @ b)


def best_cut(tr, col, cands, direction):
    best, best_ll = None, -np.inf
    y = tr.y.values.astype(float)
    for c in cands:
        z = (tr[col] >= c) if direction == ">=" else (tr[col] <= c)
        Xc = np.column_stack([np.ones(len(tr)), z.astype(float)])
        if z.sum() < 20 or (~z).sum() < 20:
            continue
        ll = logit_fit(Xc, y)[2]
        if ll > best_ll:
            best, best_ll = c, ll
    return best


def model_prevalence(tr, te):
    return np.full(len(te), tr.y.mean())


def model_fixed(tr, te):
    return fit_logit_pred(design(tr, FEATS), tr.y.values.astype(float), design(te, FEATS))


CUTS_CHOSEN = {"tix": [], "ses": []}


def model_nested(tr, te):
    ct = best_cut(tr, "tickets", range(2, 11), ">=")
    cs = best_cut(tr, "sessions", range(5, 41), "<=")
    CUTS_CHOSEN["tix"].append(ct); CUTS_CHOSEN["ses"].append(cs)

    def mk(d):
        return np.column_stack([np.ones(len(d)), (d.tickets >= ct).astype(float), (d.sessions <= cs).astype(float),
                                d.no_onb, d.monthly, d.plan_basic])
    return fit_logit_pred(mk(tr), tr.y.values.astype(float), mk(te))


def model_linear(tr, te):
    mu_t, sd_t = tr.tickets.mean(), tr.tickets.std()
    mu_s, sd_s = tr.sessions.mean(), tr.sessions.std()

    def mk(d):
        return np.column_stack([np.ones(len(d)), (d.tickets - mu_t) / sd_t, (d.sessions - mu_s) / sd_s,
                                d.no_onb, d.monthly, d.plan_basic])
    return fit_logit_pred(mk(tr), tr.y.values.astype(float), mk(te))


RAW = ["tickets", "sessions", "tenure", "disc", "price", "no_onb", "monthly", "pay_transfer", "pay_wallet"]


def model_gbm(tr, te):
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                       min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
    m.fit(tr[RAW], tr.y)
    return m.predict_proba(te[RAW])[:, 1]


def drop_variants(drop):
    feats = [f for f in FEATS if f != drop]

    def m(tr, te):
        return fit_logit_pred(design(tr, feats), tr.y.values.astype(float), design(te, feats))
    return m


def cv_oof(df, fn, n_rep=10, k=5):
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_rep, random_state=SEED)
    oof = np.zeros((n_rep, len(df)))
    for i, (tr, te) in enumerate(rskf.split(df, df.y)):
        rep = i // k
        oof[rep, te] = fn(df.iloc[tr], df.iloc[te])
    return oof


def metrics_from(y, p, share=0.20, seed=0):
    out = dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
               brier=brier_score_loss(y, p), logloss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    prev = y.mean()
    out["brier_skill"] = 1 - out["brier"] / (prev * (1 - prev))
    # قرار عند 0.5 (يُظهر لماذا الدقة وحدها مضللة)
    pred = (p >= 0.5).astype(int)
    out["accuracy_at_0.5"] = float((pred == y).mean())
    out["recall_at_0.5"] = float(pred[y == 1].mean())
    out["flagged_at_0.5"] = int(pred.sum())
    # تنبيه أعلى share من المخاطر مع كسر تعادل عشوائي ثابت
    r = np.random.default_rng(seed)
    order = np.lexsort((r.random(len(p)), -p))
    kf = int(round(share * len(p)))
    flag = np.zeros(len(p), bool); flag[order[:kf]] = True
    out[f"precision_top{int(share*100)}"] = float(y[flag].mean())
    out[f"recall_top{int(share*100)}"] = float(y[flag].sum() / y.sum())
    out[f"lift_top{int(share*100)}"] = float(y[flag].mean() / prev)
    return out


def evaluate(df):
    y = df.y.values
    models = {"prevalence": model_prevalence, "logit_fixed_cuts": model_fixed,
              "logit_nested_cuts": model_nested, "logit_linear": model_linear, "gbm": model_gbm}
    res, oofs = {}, {}
    B = 1000
    for name, fn in models.items():
        oof = cv_oof(df, fn)
        oofs[name] = oof
        pbar = oof.mean(axis=0)
        point = metrics_from(y, pbar)
        per_rep = [metrics_from(y, oof[r]) for r in range(oof.shape[0])]
        boots = {k: [] for k in point}
        n = len(y)
        for _ in range(B):
            idx = RNG.integers(0, n, n)
            if y[idx].sum() == 0:
                continue
            m = metrics_from(y[idx], pbar[idx])
            for k in point:
                boots[k].append(m[k])
        res[name] = {k: dict(est=point[k], lo=float(np.percentile(boots[k], 2.5)),
                             hi=float(np.percentile(boots[k], 97.5)),
                             rep_mean=float(np.mean([m[k] for m in per_rep])),
                             rep_sd=float(np.std([m[k] for m in per_rep]))) for k in point}
    # معايرة النموذج الأساسي (خماسيات المخاطر)
    p = oofs["logit_fixed_cuts"].mean(axis=0)
    q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False)
    cal = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    res["calibration"] = cal.reset_index().to_dict("records")
    res["nested_cuts_chosen"] = dict(
        tickets=pd.Series(CUTS_CHOSEN["tix"]).value_counts().to_dict(),
        sessions=pd.Series(CUTS_CHOSEN["ses"]).value_counts().to_dict())
    # أثر حذف كل عامل على الأداء المتحقق منه
    full_pr = res["logit_fixed_cuts"]["pr_auc"]["est"]
    full_ll = res["logit_fixed_cuts"]["logloss"]["est"]
    drops = []
    for f in FEATS:
        o = cv_oof(df, drop_variants(f)).mean(axis=0)
        m = metrics_from(y, o)
        drops.append(dict(feature=f, label=LABELS[f], pr_auc=m["pr_auc"], d_pr_auc=full_pr - m["pr_auc"],
                          logloss=m["logloss"], d_logloss=m["logloss"] - full_ll,
                          roc_auc=m["roc_auc"]))
    res["drop_one"] = drops
    # منحنى recall مقابل نسبة التنبيه للنموذج الأساسي
    curve = []
    for s in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        mm = metrics_from(y, p, share=s)
        curve.append(dict(share=s, precision=mm[f"precision_top{int(s*100)}"],
                          recall=mm[f"recall_top{int(s*100)}"], lift=mm[f"lift_top{int(s*100)}"]))
    res["curve"] = curve
    return res


# ----------------------------------------------------------------- تشغيل
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.data)
    out = dict(describe=describe(df), model=main_model(df), validation=evaluate(df))

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(clean(out), f, ensure_ascii=False, indent=1)
    print("تم. النتائج في", os.path.join(a.out, "results.json"))


if __name__ == "__main__":
    main()
