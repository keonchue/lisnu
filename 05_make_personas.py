# -*- coding: utf-8 -*-
"""
05_make_personas.py
===================
사회조사 2023 마이크로데이터에서 페르소나를 추출하고
LLM system prompt 형태로 변환해 JSON으로 저장합니다.

실행: Spyder 또는 python 05_make_personas.py
필요 패키지: pandas
"""

import pandas as pd
import json
from pathlib import Path

# ============================================================
# 경로 설정
# ============================================================

PROJECT_DIR = Path(r"C:\Users\Keon\Desktop\SNU\2학년\1학기\Basic computing\project")
OUTPUT_DIR = PROJECT_DIR / "output"

# CSV 파일을 이름으로 직접 찾음 (한글 파일명 인코딩 문제 우회)
def _find_csv():
    candidates = list(PROJECT_DIR.glob("2023*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"프로젝트 폴더에서 2023*.csv 파일을 찾을 수 없어요.\n"
            f"확인 폴더: {PROJECT_DIR}"
        )
    return candidates[0]


# ============================================================
# 코드값 → 한국어 매핑
# ============================================================

SEX_MAP = {1: "남자", 2: "여자"}

EDU_MAP = {
    0: "학교를 다니지 않음",
    1: "초등학교 졸업",
    2: "중학교 졸업",
    3: "고등학교 졸업",
    4: "2~3년제 대학 졸업",
    5: "4년제 대학 졸업",
    6: "대학원 석사 졸업",
    7: "대학원 박사 졸업",
}

MARITAL_MAP = {
    1: "미혼",
    2: "배우자 있음 (기혼)",
    3: "사별",
    4: "이혼",
    5: "별거",
}

INCOME_MAP = {
    1: "월 100만원 미만",
    2: "월 100~200만원",
    3: "월 200~300만원",
    4: "월 300~400만원",
    5: "월 400~500만원",
    6: "월 500~600만원",
    7: "월 600~700만원",
    8: "월 700~800만원",
    9: "월 800만원 이상",
}

REGION_MAP = {
    11: "서울", 21: "부산", 22: "대구", 23: "인천", 24: "광주",
    25: "대전", 26: "울산", 29: "세종", 31: "경기", 32: "강원",
    33: "충북", 34: "충남", 35: "전북", 36: "전남", 37: "경북",
    38: "경남", 39: "제주",
}

SUBJ_INC_MAP = {
    1: "매우 여유 있음",
    2: "약간 여유 있음",
    3: "적정함",
    4: "약간 부족함",
    5: "매우 부족함",
}


# ============================================================
# 핵심 함수
# ============================================================

def build_demographics(row):
    """응답자 row → 인구통계 dict (한국어 값)"""
    return {
        "성별": SEX_MAP.get(row["성별코드"], "미응답"),
        "나이": f"{int(row['만연령'])}세",
        "거주지역": REGION_MAP.get(row["행정구역시도코드"], "미응답"),
        "교육수준": EDU_MAP.get(row["교육정도코드"], "미응답"),
        "혼인상태": MARITAL_MAP.get(row["혼인상태코드"], "미응답"),
        "가구 월소득": INCOME_MAP.get(row["가구소득코드"], "응답 안 함")
            if pd.notna(row["가구소득코드"]) else "응답 안 함",
        "주관적 경제 상태": SUBJ_INC_MAP.get(row["주관적소득수준코드"], "응답 안 함")
            if pd.notna(row["주관적소득수준코드"]) else "응답 안 함",
    }


def build_system_prompt(demographics):
    """인구통계 dict → LLM system prompt 문자열"""
    lines = "\n".join(f"- {k}: {v}" for k, v in demographics.items())
    return (
        "당신은 다음 인구통계 특성을 가진 대한민국 국민입니다.\n\n"
        f"{lines}\n\n"
        "지침:\n"
        "(1) 위 특성을 가진 실제 한국인으로서 솔직하게 답하세요.\n"
        "(2) AI가 아닌 일상 경험에서 답하세요.\n"
        "(3) 모든 응답은 반드시 다음 JSON 형식으로만 출력하세요. "
        "다른 텍스트는 절대 포함하지 마세요.\n"
        '   {"answer": <선택지 번호 또는 텍스트>, "reasoning": "<한 문장 이유>"}'
    )


def make_personas(
    n=300,
    filter_age=None,        # 예: (19, 29)  → 19~29세만
    filter_region=None,     # 예: [11]       → 서울만
    filter_education=None,  # 예: [5, 6, 7]  → 4년제 이상
    random_seed=42,
):
    """
    사회조사 CSV에서 페르소나를 추출합니다.

    Parameters
    ----------
    n : int
        추출할 페르소나 수
    filter_age : tuple (min, max) or None
        연령 범위. None이면 전체 성인(19세 이상)
    filter_region : list of int or None
        지역 코드 목록. None이면 전국
    filter_education : list of int or None
        교육수준 코드 목록. None이면 전체
    random_seed : int
        재현성을 위한 난수 시드

    Returns
    -------
    list of dict
        [{"persona_id": ..., "demographics": {...}, "system_prompt": ...}, ...]
    """
    csv_path = _find_csv()

    print("=" * 60)
    print("페르소나 생성")
    print("=" * 60)
    print(f"\n[1/4] CSV 로딩 중: {csv_path.name}")
    df = pd.read_csv(csv_path, encoding="cp949", low_memory=False)
    print(f"  → 전체 응답자: {len(df):,}명")

    # --- 필터링 ---
    print(f"\n[2/4] 필터 적용 중...")

    # 기본: 성인(19세 이상)만
    mask = df["만연령"] >= 19

    if filter_age is not None:
        age_min, age_max = filter_age
        mask = mask & (df["만연령"] >= age_min) & (df["만연령"] <= age_max)
        print(f"  연령 필터: {age_min}~{age_max}세")
    else:
        print(f"  연령 필터: 19세 이상 전체")

    if filter_region is not None:
        mask = mask & (df["행정구역시도코드"].isin(filter_region))
        region_names = [REGION_MAP.get(r, str(r)) for r in filter_region]
        print(f"  지역 필터: {', '.join(region_names)}")
    else:
        print(f"  지역 필터: 전국")

    if filter_education is not None:
        mask = mask & (df["교육정도코드"].isin(filter_education))
        edu_names = [EDU_MAP.get(e, str(e)) for e in filter_education]
        print(f"  교육 필터: {', '.join(edu_names)}")
    else:
        print(f"  교육 필터: 전체")

    df_filtered = df[mask].copy()
    print(f"  → 필터 후: {len(df_filtered):,}명")

    if len(df_filtered) < n:
        raise ValueError(
            f"필터 조건을 충족하는 응답자({len(df_filtered)}명)가 "
            f"요청 수({n}명)보다 적습니다. 필터를 완화하거나 n을 줄여주세요."
        )

    # --- 추출 ---
    print(f"\n[3/4] {n}명 무작위 추출 (seed={random_seed})...")
    sampled = df_filtered.sample(n=n, random_state=random_seed)

    # --- 페르소나 변환 ---
    print(f"\n[4/4] 페르소나 변환 중...")
    personas = []
    for _, row in sampled.iterrows():
        demo = build_demographics(row)
        personas.append({
            "persona_id": f"{row['가구일련번호']}_{row['가구원번호']}",
            "demographics": demo,
            "system_prompt": build_system_prompt(demo),
        })

    # --- 저장 ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "personas.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    print(f"  → 저장: {out_path}")

    # --- 요약 통계 ---
    _print_summary(sampled, personas)

    return personas


def _print_summary(sampled, personas):
    print(f"\n{'='*60}")
    print("요약 통계")
    print(f"{'='*60}")
    n = len(sampled)

    print(f"\n[성별 분포]")
    for sex, cnt in sampled["성별코드"].map(SEX_MAP).value_counts().items():
        print(f"  {sex}: {cnt}명 ({cnt/n*100:.1f}%)")

    print(f"\n[연령대 분포]")
    age_bins = pd.cut(
        sampled["만연령"],
        bins=[18, 29, 39, 49, 59, 69, 100],
        labels=["20대", "30대", "40대", "50대", "60대", "70대+"],
    )
    for label, cnt in age_bins.value_counts().sort_index().items():
        print(f"  {label}: {cnt}명 ({cnt/n*100:.1f}%)")

    print(f"\n[지역 분포 — 상위 5개]")
    for region, cnt in sampled["행정구역시도코드"].map(REGION_MAP).value_counts().head(5).items():
        print(f"  {region}: {cnt}명 ({cnt/n*100:.1f}%)")

    print(f"\n{'='*60}")
    print("샘플 페르소나 (1번째)")
    print(f"{'='*60}")
    p = personas[0]
    print(f"ID: {p['persona_id']}")
    print("인구통계:")
    for k, v in p["demographics"].items():
        print(f"  - {k}: {v}")
    print("\nSystem prompt 미리보기:")
    print(p["system_prompt"])


# ============================================================
# 실행 진입점
# ============================================================

if __name__ == "__main__":
    # ── 여기서 필터 조건을 바꾸면 됩니다 ──
    personas = make_personas(
        n=300,
        filter_age=None,        # None = 전국 성인 전체
        filter_region=None,     # 예: [11] = 서울만
        filter_education=None,  # 예: [5,6,7] = 4년제 이상
        random_seed=42,
    )
    print(f"\n완료! 총 {len(personas)}명의 페르소나가 output/personas.json에 저장됐습니다.")
