# -*- coding: utf-8 -*-
"""
06_run_experiment.py
====================
페르소나 JSON을 로드하고 9개 검증 문항에 대해
GPT-4o-mini와 Claude Haiku 4.5의 응답을 수집합니다.

실행: python 06_run_experiment.py
필요 패키지: openai, anthropic, python-dotenv, tqdm
  설치: pip install openai anthropic python-dotenv tqdm
"""

import json
import time
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import openai
import anthropic

# ============================================================
# 경로 설정
# ============================================================

PROJECT_DIR = Path(r"C:\Users\Keon\Desktop\SNU\2학년\1학기\Basic computing\project")
OUTPUT_DIR  = PROJECT_DIR / "output"
PERSONAS_PATH = OUTPUT_DIR / "personas.json"
GPT_OUT    = OUTPUT_DIR / "responses_gpt.json"
CLAUDE_OUT = OUTPUT_DIR / "responses_claude.json"
ENV_PATH   = PROJECT_DIR / ".env"

# ============================================================
# 모델 및 비용 설정
# ============================================================

GPT_MODEL    = "gpt-4o-mini"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# 달러 / 토큰
GPT_COST    = {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000}
CLAUDE_COST = {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000}

BUDGET_LIMIT = 5.0  # 달러 — 초과 시 자동 중단

# ============================================================
# 9개 검증 문항
# ============================================================

QUESTIONS = [
    {
        "id": "G1",
        "category": "정치",
        "question": (
            "귀하는 윤석열 대통령이 대통령으로서의 직무를 잘 수행하고 있다고 보십니까, "
            "아니면 잘못 수행하고 있다고 보십니까?"
        ),
        "options": {
            1: "잘하고 있다",
            2: "잘못하고 있다",
            3: "어느 쪽도 아니다",
            4: "모름/응답거절",
        },
        "answer_type": "choice",
        "ground_truth": {1: 27, 2: 63, 3: 4, 4: 7},
        "source": "갤럽 593호 (2024.8.20-22)",
    },
    {
        "id": "G2",
        "category": "정치",
        "question": "귀하는 다음 중 어느 정당을 지지하십니까?",
        "options": {
            1: "더불어민주당",
            2: "국민의힘",
            3: "조국혁신당",
            4: "개혁신당",
            5: "기타정당",
            6: "무당층 (지지 정당 없음)",
        },
        "answer_type": "choice",
        "ground_truth": {1: 31, 2: 32, 3: 8, 4: 2, 5: 5, 6: 22},
        "source": "갤럽 593호 (2024년 8월)",
    },
    {
        "id": "G3",
        "category": "경제",
        "question": (
            "귀하는 앞으로 1년간 우리나라 경제가 현재에 비해 "
            "다음 중 어떠할 것이라고 보십니까?"
        ),
        "options": {
            1: "좋아질 것",
            2: "나빠질 것",
            3: "비슷할 것",
            4: "모름/응답거절",
        },
        "answer_type": "choice",
        "ground_truth": {1: 17, 2: 54, 3: 25, 4: 4},
        "source": "갤럽 593호 (2024년 8월)",
    },
    {
        "id": "G4",
        "category": "올림픽",
        "question": (
            "이번 파리 올림픽에 출전한 우리 대표팀에서 가장 인상적인 활약을 한 선수는 "
            "누구입니까? (선수 이름을 직접 써주세요)"
        ),
        "options": None,
        "answer_type": "free",
        "ground_truth": {"안세영": 39, "신유빈": 25, "김우진": 21, "오상욱": 11, "김예지": 9.3},
        "source": "갤럽 593호 (2024년 8월)",
    },
    {
        "id": "G5",
        "category": "정치",
        "question": (
            "귀하는 다음 중 어느 정당을 지지하십니까? "
            "(2024년 12월 탄핵 정국 상황을 기준으로 답해주세요)"
        ),
        "options": {
            1: "더불어민주당",
            2: "국민의힘",
            3: "조국혁신당",
            4: "개혁신당",
            5: "기타정당",
            6: "무당층 (지지 정당 없음)",
        },
        "answer_type": "choice",
        "ground_truth": {1: 48, 2: 24, 3: 4, 4: 2, 5: 1, 6: 21},
        "source": "갤럽 607호 (2024년 12월, 탄핵 직후)",
    },
    {
        "id": "G6",
        "category": "정치",
        "question": (
            "귀하는 앞으로 우리나라를 이끌어갈 정치 지도자, 즉 장래 대통령감으로 "
            "누가 좋다고 생각하십니까? (이름을 직접 써주세요)"
        ),
        "options": None,
        "answer_type": "free",
        "ground_truth": {"이재명": 37, "한동훈": 5, "홍준표": 5, "조국": 3, "오세훈": 2, "김문수": 2},
        "source": "갤럽 607호 (2024년 12월)",
    },
    {
        "id": "C1",
        "category": "소비",
        "question": (
            "귀하는 지속가능한 소비(친환경 제품 구매, 자원 절약 등)를 실천하고 계십니까?"
        ),
        "options": {1: "실천한다", 2: "실천하지 않는다"},
        "answer_type": "choice",
        "ground_truth": {1: 42.4, 2: 57.6},
        "source": "한국소비자원 (2023)",
    },
    {
        "id": "C2",
        "category": "소비",
        "question": (
            "지속가능한 소비를 실천하는 데 있어 가장 큰 방해요인은 무엇입니까? "
            "(1순위 하나만 선택)"
        ),
        "options": {
            1: "비싼 수리비용/제품가격",
            2: "부품 부재",
            3: "정보 부족",
            4: "시간 부족",
            5: "기타",
        },
        "answer_type": "choice",
        "ground_truth": {1: 56.9, 2: 50.9, 3: 49.9, 4: None, 5: None},
        "source": "한국소비자원 (2023)",
    },
    {
        "id": "C3",
        "category": "소비",
        "question": "귀하는 소비자교육 프로그램에 참여하실 의향이 있으십니까?",
        "options": {1: "있다", 2: "없다"},
        "answer_type": "choice",
        "ground_truth": {1: 26.4, 2: 73.6},
        "source": "한국소비자원 (2023)",
    },
]

# ============================================================
# 유틸리티
# ============================================================

def build_user_prompt(question):
    """문항 dict → user 메시지 문자열"""
    q = question["question"]
    if question["options"]:
        opts = "\n".join(f"  {k}. {v}" for k, v in question["options"].items())
        return (
            f"다음 질문에 답해주세요.\n\n"
            f"질문: {q}\n\n"
            f"선택지:\n{opts}\n\n"
            f'반드시 JSON 형식으로만 답하세요: {{"answer": <번호>, "reasoning": "<한 문장>"}}'
        )
    else:
        return (
            f"다음 질문에 답해주세요.\n\n"
            f"질문: {q}\n\n"
            f'반드시 JSON 형식으로만 답하세요: {{"answer": "<이름>", "reasoning": "<한 문장>"}}'
        )


def parse_response(raw_text):
    """LLM 응답 텍스트 → dict 또는 None"""
    text = raw_text.strip()
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def calc_cost(tokens, rate):
    return tokens["input"] * rate["input"] + tokens["output"] * rate["output"]


def load_checkpoint(path):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# API 호출
# ============================================================

def call_gpt(client, system_prompt, user_prompt, retries=3):
    """(parsed, raw, tokens) 반환"""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=150,
                temperature=0.7,
            )
            raw = resp.choices[0].message.content
            tokens = {
                "input":  resp.usage.prompt_tokens,
                "output": resp.usage.completion_tokens,
            }
            return parse_response(raw), raw, tokens
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, f"ERROR: {e}", {"input": 0, "output": 0}


def call_claude(client, system_prompt, user_prompt, retries=3):
    """(parsed, raw, tokens) 반환"""
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=150,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = resp.content[0].text
            tokens = {
                "input":  resp.usage.input_tokens,
                "output": resp.usage.output_tokens,
            }
            return parse_response(raw), raw, tokens
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, f"ERROR: {e}", {"input": 0, "output": 0}


# ============================================================
# 비용 예측
# ============================================================

def estimate_cost(n_personas, n_questions):
    # 평균 추정: input ~350 토큰, output ~60 토큰
    calls = n_personas * n_questions
    in_tok  = calls * 350
    out_tok = calls * 60
    gpt_usd    = in_tok * GPT_COST["input"]    + out_tok * GPT_COST["output"]
    claude_usd = in_tok * CLAUDE_COST["input"] + out_tok * CLAUDE_COST["output"]
    return {
        "calls_per_model": calls,
        "gpt_usd":    gpt_usd,
        "claude_usd": claude_usd,
        "total_usd":  gpt_usd + claude_usd,
    }


# ============================================================
# 메인 실험 함수
# ============================================================

def run_experiment(personas, model="both", dry_run=False):
    """
    Parameters
    ----------
    personas : list  — make_personas() 결과
    model    : str   — "gpt" | "claude" | "both"
    dry_run  : bool  — True이면 3명 × 2문항만 (API 연결 테스트용)
    """
    # API 키 로드
    load_dotenv(ENV_PATH)
    openai_key    = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    run_gpt    = model in ("gpt",    "both")
    run_claude = model in ("claude", "both")

    if run_gpt and not openai_key:
        raise ValueError(".env 파일에 OPENAI_API_KEY가 없습니다.")
    if run_claude and not anthropic_key:
        raise ValueError(".env 파일에 ANTHROPIC_API_KEY가 없습니다.")

    # dry_run 제한
    if dry_run:
        test_personas  = personas[:3]
        test_questions = QUESTIONS[:2]
        print("\n[DRY RUN] 3명 × 2문항만 실행합니다.")
    else:
        test_personas  = personas
        test_questions = QUESTIONS

    n   = len(test_personas)
    n_q = len(test_questions)

    # 비용 예측 출력
    est = estimate_cost(n, n_q)
    print("\n" + "=" * 60)
    print("예상 비용")
    print("=" * 60)
    print(f"  페르소나 수  : {n}명")
    print(f"  문항 수      : {n_q}개")
    print(f"  모델당 호출  : {est['calls_per_model']}회")
    print(f"  GPT-4o-mini  : ${est['gpt_usd']:.4f}")
    print(f"  Claude Haiku : ${est['claude_usd']:.4f}")
    print(f"  합계         : ${est['total_usd']:.4f}")
    print(f"  예산 한도    : ${BUDGET_LIMIT:.2f}")

    if not dry_run:
        ans = input("\n계속 진행할까요? (y/n): ").strip().lower()
        if ans != "y":
            print("실험을 취소했습니다.")
            return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gpt_client    = openai.OpenAI(api_key=openai_key)       if run_gpt    else None
    claude_client = anthropic.Anthropic(api_key=anthropic_key) if run_claude else None

    gpt_data    = load_checkpoint(GPT_OUT)    if run_gpt    else {}
    claude_data = load_checkpoint(CLAUDE_OUT) if run_claude else {}

    gpt_spent    = 0.0
    claude_spent = 0.0

    # ── GPT 실행 ──────────────────────────────────────────────
    if run_gpt:
        print(f"\n{'='*60}\nGPT-4o-mini 실행 중...\n{'='*60}")
        stopped = False

        for i, persona in enumerate(tqdm(test_personas, desc="GPT")):
            pid = persona["persona_id"]
            if pid not in gpt_data:
                gpt_data[pid] = {
                    "demographics": persona["demographics"],
                    "responses": {},
                }

            for q in test_questions:
                qid = q["id"]
                if qid in gpt_data[pid]["responses"]:
                    continue  # 이미 완료 → 건너뜀

                user_prompt = build_user_prompt(q)
                parsed, raw, tokens = call_gpt(
                    gpt_client, persona["system_prompt"], user_prompt
                )
                cost = calc_cost(tokens, GPT_COST)
                gpt_spent += cost

                gpt_data[pid]["responses"][qid] = {
                    "parsed":   parsed,
                    "raw":      raw,
                    "tokens":   tokens,
                    "cost_usd": cost,
                }

            if (i + 1) % 10 == 0:
                save_checkpoint(GPT_OUT, gpt_data)
                tqdm.write(f"  [저장] {i+1}명 완료 | GPT 누적 비용: ${gpt_spent:.4f}")

            if gpt_spent > BUDGET_LIMIT:
                print(f"\n예산 한도(${BUDGET_LIMIT})를 초과했습니다. GPT 중단.")
                stopped = True
                break

        save_checkpoint(GPT_OUT, gpt_data)
        print(f"\nGPT {'(조기 중단)' if stopped else '완료'}! 총 비용: ${gpt_spent:.4f}")
        print(f"저장: {GPT_OUT}")

    # ── Claude 실행 ───────────────────────────────────────────
    if run_claude:
        print(f"\n{'='*60}\nClaude Haiku 4.5 실행 중...\n{'='*60}")
        stopped = False

        for i, persona in enumerate(tqdm(test_personas, desc="Claude")):
            pid = persona["persona_id"]
            if pid not in claude_data:
                claude_data[pid] = {
                    "demographics": persona["demographics"],
                    "responses": {},
                }

            for q in test_questions:
                qid = q["id"]
                if qid in claude_data[pid]["responses"]:
                    continue

                user_prompt = build_user_prompt(q)
                parsed, raw, tokens = call_claude(
                    claude_client, persona["system_prompt"], user_prompt
                )
                cost = calc_cost(tokens, CLAUDE_COST)
                claude_spent += cost

                claude_data[pid]["responses"][qid] = {
                    "parsed":   parsed,
                    "raw":      raw,
                    "tokens":   tokens,
                    "cost_usd": cost,
                }

            if (i + 1) % 10 == 0:
                save_checkpoint(CLAUDE_OUT, claude_data)
                tqdm.write(f"  [저장] {i+1}명 완료 | Claude 누적 비용: ${claude_spent:.4f}")

            if claude_spent > BUDGET_LIMIT:
                print(f"\n예산 한도(${BUDGET_LIMIT})를 초과했습니다. Claude 중단.")
                stopped = True
                break

        save_checkpoint(CLAUDE_OUT, claude_data)
        print(f"\nClaude {'(조기 중단)' if stopped else '완료'}! 총 비용: ${claude_spent:.4f}")
        print(f"저장: {CLAUDE_OUT}")

    # ── 최종 요약 ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"실험 완료! 총 비용: ${gpt_spent + claude_spent:.4f}")
    print(f"{'='*60}")


# ============================================================
# 실행 진입점
# ============================================================

if __name__ == "__main__":
    if not PERSONAS_PATH.exists():
        print("personas.json이 없습니다. 먼저 05_make_personas.py를 실행하세요.")
    else:
        with open(PERSONAS_PATH, encoding="utf-8") as f:
            personas = json.load(f)
        print(f"페르소나 {len(personas)}명 로드 완료.")

        # ── 여기서 설정 변경 ──────────────────────────────────
        # dry_run=True  → 3명×2문항 테스트 (API 키 확인용)
        # dry_run=False → 전체 실행 (시작 전 비용 확인 물어봄)
        run_experiment(personas, model="both", dry_run=False)
