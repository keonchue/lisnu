# -*- coding: utf-8 -*-
"""
07_analyze_results.py
=====================
실험 결과 분석 및 시각화

실행: python 07_analyze_results.py
필요 패키지: pandas, matplotlib, scipy, numpy
  설치: pip install matplotlib scipy numpy
"""

import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import wasserstein_distance
from pathlib import Path
from collections import Counter

# ============================================================
# 경로
# ============================================================

PROJECT_DIR = Path(r"C:\Users\Keon\Desktop\SNU\2학년\1학기\Basic computing\project")
OUTPUT_DIR  = PROJECT_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
GPT_PATH    = OUTPUT_DIR / "responses_gpt.json"
CLAUDE_PATH = OUTPUT_DIR / "responses_claude.json"
SUMMARY_PATH = OUTPUT_DIR / "analysis_summary.md"

# ============================================================
# 한국어 폰트 설정 (Windows)
# ============================================================

def setup_korean_font():
    candidates = ["Malgun Gothic", "맑은 고딕", "NanumGothic", "NanumBarunGothic", "AppleGothic"]
    available  = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.family"] = c
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[폰트] '{c}' 사용")
            return c
    # 경로에서 직접 탐색
    for f in fm.findSystemFonts():
        if any(k in f.lower() for k in ["malgun", "nanum", "gothic"]):
            prop = fm.FontProperties(fname=f)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[폰트] '{prop.get_name()}' 사용 ({f})")
            return prop.get_name()
    print("[폰트] 한국어 폰트를 찾지 못했습니다. 글씨가 깨질 수 있습니다.")
    plt.rcParams["axes.unicode_minus"] = False
    return None

# ============================================================
# 문항 메타데이터 + 정답지
# ============================================================

QUESTIONS = [
    {
        "id": "G1", "category": "정치",
        "text": "윤석열 대통령 직무 평가",
        "type": "choice",
        "options": {1: "잘함", 2: "잘못함", 3: "중립", 4: "모름"},
        "ground_truth": {1: 27.0, 2: 63.0, 3: 4.0, 4: 7.0},
        "source": "갤럽 593호 (2024.8)",
    },
    {
        "id": "G2", "category": "정치",
        "text": "정당 지지도 (2024년 8월)",
        "type": "choice",
        "options": {1: "민주", 2: "국힘", 3: "조국", 4: "개혁", 5: "기타", 6: "무당"},
        "ground_truth": {1: 31.0, 2: 32.0, 3: 8.0, 4: 2.0, 5: 5.0, 6: 22.0},
        "source": "갤럽 593호 (2024.8)",
    },
    {
        "id": "G3", "category": "경제",
        "text": "향후 1년 경기 전망",
        "type": "choice",
        "options": {1: "좋아짐", 2: "나빠짐", 3: "비슷", 4: "모름"},
        "ground_truth": {1: 17.0, 2: 54.0, 3: 25.0, 4: 4.0},
        "source": "갤럽 593호 (2024.8)",
    },
    {
        "id": "G4", "category": "올림픽",
        "text": "파리 올림픽 인상적 선수 [AI cutoff 테스트]",
        "type": "free",
        "options": None,
        "ground_truth": {"안세영": 39.0, "신유빈": 25.0, "김우진": 21.0, "오상욱": 11.0, "김예지": 9.3},
        "source": "갤럽 593호 (2024.8)",
    },
    {
        "id": "G5", "category": "정치",
        "text": "정당 지지도 (2024년 12월, 탄핵 직후)",
        "type": "choice",
        "options": {1: "민주", 2: "국힘", 3: "조국", 4: "개혁", 5: "기타", 6: "무당"},
        "ground_truth": {1: 48.0, 2: 24.0, 3: 4.0, 4: 2.0, 5: 1.0, 6: 21.0},
        "source": "갤럽 607호 (2024.12)",
    },
    {
        "id": "G6", "category": "정치",
        "text": "장래 대통령감 선호도",
        "type": "free",
        "options": None,
        "ground_truth": {"이재명": 37.0, "한동훈": 5.0, "홍준표": 5.0, "조국": 3.0, "오세훈": 2.0, "김문수": 2.0},
        "source": "갤럽 607호 (2024.12)",
    },
    {
        "id": "C1", "category": "소비",
        "text": "지속가능소비 실천 여부",
        "type": "choice",
        "options": {1: "실천함", 2: "비실천"},
        "ground_truth": {1: 42.4, 2: 57.6},
        "source": "한국소비자원 (2023)",
    },
    {
        "id": "C2", "category": "소비",
        "text": "지속가능소비 방해요인 (1순위)",
        "type": "choice",
        "options": {1: "비싼가격", 2: "부품부재", 3: "정보부족", 4: "시간부족", 5: "기타"},
        # 다중응답 → 1순위 비율로 단순화 (상위 3개만 유효)
        "ground_truth": {1: 56.9, 2: 50.9, 3: 49.9, 4: None, 5: None},
        "source": "한국소비자원 (2023) ※다중응답",
        "note": "원본은 다중응답. 1순위 단답 응답과 직접 비교 시 해석 주의.",
    },
    {
        "id": "C3", "category": "소비",
        "text": "소비자교육 참여 의향",
        "type": "choice",
        "options": {1: "있음", 2: "없음"},
        "ground_truth": {1: 26.4, 2: 73.6},
        "source": "한국소비자원 (2023)",
    },
]
Q_MAP = {q["id"]: q for q in QUESTIONS}

# ============================================================
# 데이터 로드
# ============================================================

def load_responses(path):
    """JSON → {qid: [answer, ...]} 딕셔너리"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    responses = {q["id"]: [] for q in QUESTIONS}
    for pid, persona in data.items():
        for qid, r in persona["responses"].items():
            if qid in responses:
                responses[qid].append({
                    "answer":      r["parsed"]["answer"] if r["parsed"] else None,
                    "demographics": persona["demographics"],
                })
    return responses

# ============================================================
# 분포 계산
# ============================================================

def compute_choice_dist(answers, options):
    """선택형 문항 → {option_key: pct} 정규화 분포"""
    counts = {k: 0 for k in options}
    valid  = 0
    for a in answers:
        v = a["answer"]
        if v is None:
            continue
        try:
            k = int(v)
        except (ValueError, TypeError):
            continue
        if k in counts:
            counts[k] += 1
            valid += 1
    if valid == 0:
        return {k: 0.0 for k in options}
    return {k: counts[k] / valid * 100 for k in options}


def compute_free_dist(answers, top_n=7):
    """자유응답 문항 → {name: pct} 상위 N개"""
    freq = Counter()
    total = 0
    for a in answers:
        v = a["answer"]
        if v is None or str(v).strip() in ("", "null", "None"):
            continue
        name = str(v).strip()
        freq[name] += 1
        total += 1
    if total == 0:
        return {}
    return {k: v / total * 100 for k, v in freq.most_common(top_n)}

# ============================================================
# 정확도 지표
# ============================================================

def mae_accuracy(ai_dist, gt_dist):
    """
    정확도 = 1 - MAE / 100
    선택형 문항에서 정답 분포와 AI 분포의 평균 절대 오차로 계산.
    None 정답은 제외.
    """
    valid_keys = [k for k, v in gt_dist.items() if v is not None]
    if not valid_keys:
        return None
    gt_total = sum(gt_dist[k] for k in valid_keys)
    ai_total = sum(ai_dist.get(k, 0) for k in valid_keys)
    if gt_total == 0 or ai_total == 0:
        return None
    # 유효 선택지 내에서 재정규화
    gt_norm = {k: gt_dist[k] / gt_total * 100 for k in valid_keys}
    ai_norm = {k: ai_dist.get(k, 0) / ai_total * 100 for k in valid_keys}
    mae = np.mean([abs(ai_norm[k] - gt_norm[k]) for k in valid_keys])
    return max(0.0, 1 - mae / 100)


def wd_score(ai_dist, gt_dist, option_keys):
    """Wasserstein distance (None 항목 제외, 숫자 옵션만)"""
    valid_keys = [k for k in option_keys if gt_dist.get(k) is not None]
    if len(valid_keys) < 2:
        return None
    gt_total = sum(gt_dist[k] for k in valid_keys)
    ai_total = sum(ai_dist.get(k, 0) for k in valid_keys)
    if gt_total == 0 or ai_total == 0:
        return None
    gt_w = np.array([gt_dist[k] / gt_total for k in valid_keys])
    ai_w = np.array([ai_dist.get(k, 0) / ai_total for k in valid_keys])
    try:
        vals = list(range(len(valid_keys)))
        return wasserstein_distance(vals, vals, u_weights=gt_w, v_weights=ai_w)
    except Exception:
        return None

# ============================================================
# 시각화 공통 스타일
# ============================================================

BLUE  = "#2563eb"
GRAY  = "#94a3b8"
GREEN = "#16a34a"
RED   = "#dc2626"
ORANGE = "#f59e0b"

def save_fig(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → 저장: {path.name}")

# ============================================================
# 그림 1–9: 문항별 분포 비교
# ============================================================

def fig_per_question(gpt_res, claude_res):
    print("\n[1/4] 문항별 분포 그래프 생성 중...")

    for q in QUESTIONS:
        qid = q["id"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(f"[{qid}] {q['text']}", fontsize=13, fontweight="bold", y=1.02)

        datasets = [
            ("GPT-4o-mini",    gpt_res[qid],    axes[0]),
            ("Claude Haiku",   claude_res[qid], axes[1]),
        ]

        for model_label, answers, ax in datasets:
            ax.set_title(model_label, fontsize=11, fontweight="bold")

            if q["type"] == "choice":
                opts  = q["options"]
                ai_d  = compute_choice_dist(answers, opts)
                gt    = q["ground_truth"]
                valid = [k for k in opts if gt.get(k) is not None]
                labels = [opts[k] for k in valid]
                ai_vals = [ai_d.get(k, 0) for k in valid]
                gt_vals = [gt[k] for k in valid]

                x = np.arange(len(valid))
                w = 0.35
                bars_ai = ax.bar(x - w/2, ai_vals, w, label="AI", color=BLUE,   alpha=0.85, zorder=3)
                bars_gt = ax.bar(x + w/2, gt_vals, w, label="실제", color=GRAY, alpha=0.85, zorder=3)

                # 값 레이블
                for bar in bars_ai:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.1f}%",
                                ha="center", va="bottom", fontsize=8, color=BLUE, fontweight="bold")
                for bar in bars_gt:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.1f}%",
                                ha="center", va="bottom", fontsize=8, color="#475569")

                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=9)
                ax.set_ylabel("응답 비율 (%)", fontsize=9)
                ax.set_ylim(0, max(max(ai_vals+gt_vals, default=0)*1.2, 10))
                ax.legend(fontsize=9)
                ax.grid(axis="y", alpha=0.3, zorder=0)
                ax.spines[["top","right"]].set_visible(False)

                # 정확도 표시
                acc = mae_accuracy(ai_d, gt)
                if acc is not None:
                    ax.text(0.97, 0.97, f"정확도: {acc:.2f}", transform=ax.transAxes,
                            ha="right", va="top", fontsize=9, color=BLUE,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE, alpha=0.8))

            else:  # free
                ai_d = compute_free_dist(answers, top_n=6)
                gt   = q["ground_truth"]
                # 두 분포의 모든 이름 합쳐서 상위 7개
                all_names = list(dict.fromkeys(list(ai_d.keys()) + list(gt.keys())))[:7]
                ai_vals = [ai_d.get(n, 0) for n in all_names]
                gt_vals = [gt.get(n, 0) for n in all_names]

                y = np.arange(len(all_names))
                w = 0.35
                ax.barh(y + w/2, ai_vals, w, label="AI",  color=BLUE, alpha=0.85, zorder=3)
                ax.barh(y - w/2, gt_vals, w, label="실제", color=GRAY, alpha=0.85, zorder=3)
                ax.set_yticks(y)
                ax.set_yticklabels(all_names, fontsize=9)
                ax.set_xlabel("응답 비율 (%)", fontsize=9)
                ax.legend(fontsize=9)
                ax.grid(axis="x", alpha=0.3, zorder=0)
                ax.spines[["top","right"]].set_visible(False)

        fig.text(0.5, -0.03, f"출처: {q['source']}", ha="center", fontsize=8, color="#94a3b8")
        plt.tight_layout()
        save_fig(fig, f"q_{qid}.png")

# ============================================================
# 그림 10: 모델별 정확도 비교
# ============================================================

def fig_model_comparison(gpt_res, claude_res):
    print("\n[2/4] 모델별 정확도 비교 그래프 생성 중...")

    qids  = []
    gpt_accs   = []
    claude_accs = []

    for q in QUESTIONS:
        if q["type"] != "choice":
            continue
        qid   = q["id"]
        opts  = q["options"]
        gt    = q["ground_truth"]

        gpt_d    = compute_choice_dist(gpt_res[qid],    opts)
        claude_d = compute_choice_dist(claude_res[qid], opts)

        gpt_acc    = mae_accuracy(gpt_d,    gt)
        claude_acc = mae_accuracy(claude_d, gt)

        if gpt_acc is None or claude_acc is None:
            continue

        qids.append(qid)
        gpt_accs.append(gpt_acc)
        claude_accs.append(claude_acc)

    x = np.arange(len(qids))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, gpt_accs,    w, label="GPT-4o-mini", color=BLUE,  alpha=0.85, zorder=3)
    bars2 = ax.bar(x + w/2, claude_accs, w, label="Claude Haiku", color=GREEN, alpha=0.85, zorder=3)

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.005, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(0.5, color=RED, linestyle="--", linewidth=1, alpha=0.5, label="기준선 (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(qids)
    ax.set_ylabel("정확도 (1 - MAE/100)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("모델별 문항 정확도 비교 (선택형 문항만)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top","right"]].set_visible(False)

    # 평균 정확도
    ax.text(0.98, 0.02, f"GPT 평균: {np.mean(gpt_accs):.3f}  /  Claude 평균: {np.mean(claude_accs):.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#475569")

    plt.tight_layout()
    save_fig(fig, "model_comparison.png")
    return dict(zip(qids, zip(gpt_accs, claude_accs)))

# ============================================================
# 그림 11: 영역별 정확도 히트맵
# ============================================================

def fig_category_heatmap(gpt_res, claude_res):
    print("\n[3/4] 영역별 정확도 히트맵 생성 중...")

    categories = ["정치", "경제", "소비", "올림픽"]
    models = ["GPT-4o-mini", "Claude Haiku"]
    heat = np.full((len(categories), len(models)), np.nan)

    for ci, cat in enumerate(categories):
        qs_in_cat = [q for q in QUESTIONS if q["category"] == cat and q["type"] == "choice"]
        for mi, (label, res) in enumerate([("GPT-4o-mini", gpt_res), ("Claude Haiku", claude_res)]):
            accs = []
            for q in qs_in_cat:
                d   = compute_choice_dist(res[q["id"]], q["options"])
                acc = mae_accuracy(d, q["ground_truth"])
                if acc is not None:
                    accs.append(acc)
            if accs:
                heat[ci, mi] = np.mean(accs)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(heat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="정확도")

    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels(models, fontsize=11)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_title("영역별 평균 정확도 히트맵", fontsize=12, fontweight="bold")

    for ci in range(len(categories)):
        for mi in range(len(models)):
            v = heat[ci, mi]
            if not np.isnan(v):
                ax.text(mi, ci, f"{v:.2f}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if v < 0.4 or v > 0.75 else "#1e293b")

    plt.tight_layout()
    save_fig(fig, "category_heatmap.png")

# ============================================================
# 그림 12: 인구통계 그룹별 분포 (G2 정당지지도)
# ============================================================

def fig_demographic_breakdown(gpt_res, claude_res):
    print("\n[4/4] 인구통계 그룹별 분포 그래프 생성 중...")

    q   = Q_MAP["G2"]
    opts = q["options"]

    def group_dist(answers, key_fn):
        groups = {}
        for a in answers:
            grp = key_fn(a["demographics"])
            if grp not in groups:
                groups[grp] = []
            groups[grp].append(a)
        return {g: compute_choice_dist(ans, opts) for g, ans in groups.items()}

    def age_group(demo):
        age = int(str(demo["나이"]).replace("세", ""))
        if age < 30: return "20대"
        if age < 40: return "30대"
        if age < 50: return "40대"
        if age < 60: return "50대"
        return "60대+"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("G2 정당 지지도 — 인구통계 그룹별 분포 (GPT)", fontsize=13, fontweight="bold")

    groups_def = [
        ("성별",   lambda d: d["성별"],       axes[0, 0]),
        ("연령대", age_group,                 axes[0, 1]),
        ("교육",   lambda d: "대졸이상" if any(k in d["교육수준"] for k in ["4년제","석사","박사"]) else "대졸미만", axes[1, 0]),
        ("소득",   lambda d: "고소득" if any(k in d["가구 월소득"] for k in ["600","700","800"]) else "저·중소득", axes[1, 1]),
    ]

    colors = [BLUE, GREEN, ORANGE, RED, "#8b5cf6", GRAY]
    opt_labels = list(opts.values())

    for title, key_fn, ax in groups_def:
        grp_dists = group_dist(gpt_res["G2"], key_fn)
        grp_names = sorted(grp_dists.keys())
        x = np.arange(len(opts))
        w = 0.8 / max(len(grp_names), 1)

        for gi, grp in enumerate(grp_names):
            vals = [grp_dists[grp].get(k, 0) for k in opts]
            offset = (gi - len(grp_names)/2 + 0.5) * w
            ax.bar(x + offset, vals, w * 0.9, label=grp,
                   color=colors[gi % len(colors)], alpha=0.82, zorder=3)

        ax.set_title(f"{title}별 정당 지지 분포", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(opt_labels, fontsize=9)
        ax.set_ylabel("%", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    save_fig(fig, "demographic_breakdown.png")

# ============================================================
# 분석 요약 MD 저장
# ============================================================

def save_summary(gpt_res, claude_res, acc_table):
    lines = []
    lines.append("# 실험 결과 요약\n")
    lines.append(f"- **날짜**: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    lines.append(f"- **페르소나 수**: 300명")
    lines.append(f"- **모델**: GPT-4o-mini, Claude Haiku 4.5\n")

    lines.append("## 문항별 정확도 (선택형)\n")
    lines.append("| 문항 | 영역 | GPT 정확도 | Claude 정확도 | Wasserstein (GPT) |")
    lines.append("|------|------|-----------|--------------|-------------------|")

    for q in QUESTIONS:
        if q["type"] != "choice":
            continue
        qid  = q["id"]
        opts = q["options"]
        gt   = q["ground_truth"]
        gpt_d    = compute_choice_dist(gpt_res[qid],    opts)
        claude_d = compute_choice_dist(claude_res[qid], opts)
        gpt_acc    = mae_accuracy(gpt_d,    gt)
        claude_acc = mae_accuracy(claude_d, gt)
        wd = wd_score(gpt_d, gt, list(opts.keys()))

        gpt_str    = f"{gpt_acc:.3f}"    if gpt_acc    is not None else "N/A"
        claude_str = f"{claude_acc:.3f}" if claude_acc is not None else "N/A"
        wd_str     = f"{wd:.3f}"         if wd         is not None else "N/A"
        note       = f" ⚠️ {q.get('note','')}" if q.get("note") else ""
        lines.append(f"| {qid} | {q['category']} | {gpt_str} | {claude_str} | {wd_str} |{note}")

    lines.append("")
    lines.append("## 자유응답 문항 (상위 5개)\n")
    for q in QUESTIONS:
        if q["type"] != "free":
            continue
        lines.append(f"### {q['id']} — {q['text']}\n")
        for label, res in [("GPT", gpt_res), ("Claude", claude_res)]:
            d = compute_free_dist(res[q["id"]], top_n=5)
            lines.append(f"**{label}** 상위 5개: " + ", ".join(f"{k}({v:.1f}%)" for k, v in d.items()))
        gt = q["ground_truth"]
        lines.append(f"**실제** 상위 5개: " + ", ".join(f"{k}({v}%)" for k, v in list(gt.items())[:5]))
        lines.append("")

    lines.append("## 해석 노트\n")
    lines.append("- **G5 탄핵 직후 정당지지**: 8월 학습 데이터로 12월 격변을 예측할 수 없음 → 낮은 정확도 예상")
    lines.append("- **G4 올림픽 선수**: AI 학습 컷오프 이전 데이터로 정확도 측정 가능 (안세영 파리 올림픽 활약)")
    lines.append("- **C2 방해요인**: 원본 다중응답 데이터와 1순위 단답 비교로 직접 비교 주의 필요")
    lines.append("- **정확도 = 1 - MAE/100**: 1에 가까울수록 실제 분포와 유사")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  → 요약 저장: {SUMMARY_PATH.name}")

# ============================================================
# 메인
# ============================================================

def main():
    print("=" * 60)
    print("실험 결과 분석 시작")
    print("=" * 60)

    setup_korean_font()

    print("\n[데이터 로드]")
    if not GPT_PATH.exists():
        raise FileNotFoundError(f"파일 없음: {GPT_PATH}\n06_run_experiment.py 먼저 실행하세요.")
    if not CLAUDE_PATH.exists():
        raise FileNotFoundError(f"파일 없음: {CLAUDE_PATH}\n06_run_experiment.py 먼저 실행하세요.")

    gpt_res    = load_responses(GPT_PATH)
    claude_res = load_responses(CLAUDE_PATH)

    n_gpt    = len(json.load(open(GPT_PATH,    encoding="utf-8")))
    n_claude = len(json.load(open(CLAUDE_PATH, encoding="utf-8")))
    print(f"  GPT: {n_gpt}명 / Claude: {n_claude}명 로드 완료")

    # 그림 생성
    fig_per_question(gpt_res, claude_res)
    acc_table = fig_model_comparison(gpt_res, claude_res)
    fig_category_heatmap(gpt_res, claude_res)
    fig_demographic_breakdown(gpt_res, claude_res)

    # 요약 저장
    save_summary(gpt_res, claude_res, acc_table)

    print("\n" + "=" * 60)
    print("분석 완료!")
    print(f"  그래프: {FIGURES_DIR}")
    print(f"  요약:   {SUMMARY_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
