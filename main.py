"""Mini NPU Simulator

MAC(Multiply-Accumulate) 연산으로 입력 패턴이 어떤 필터(Cross / X)와
더 닮았는지 판별하는 콘솔 시뮬레이터.

- 외부 라이브러리 없이 표준 라이브러리(json, time 등)만 사용한다.
- 실행: python main.py [data.json 경로]
- 보너스: 1차원 배열 MAC(mac_flat) 최적화 비교, Cross/X 패턴 생성기(generate_pattern)
"""

import gc
import json
import math
import os
import re
import sys
import time
import unicodedata

# ---------------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------------
EPSILON = 1e-9          # 점수 비교 허용오차: |a - b| < EPSILON 이면 동점
EPSILON_TEXT = "1e-9"   # 출력용 표기
REPEAT = 10             # 성능 측정 반복 횟수 (평균을 낸다)
USER_INPUT_SIZE = 3     # 모드 1 입력 크기
GEN_MIN_SIZE = 3        # 패턴 생성기 최소 크기 (N<3 이면 Cross와 X가 구별되지 않음)
GEN_MAX_SIZE = 200      # 패턴 생성기 최대 크기 (콘솔 대기 시간 제한)
GEN_DEFAULT_SIZES = (3, 5, 13, 25, 50, 100)
PREVIEW_MAX_SIZE = 9    # 생성 패턴 미리보기를 출력할 최대 크기
OPT_TARGET_MS = 2.0     # 최적화 비교: 1회 측정 구간의 목표 길이(ms)
OPT_MAX_BATCH = 4096    # 최적화 비교: 1회 측정당 최대 연속 호출 수

LABEL_CROSS = "Cross"
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"

# 원본 표기 -> 표준 라벨 (소문자로 비교)
#   expected 값: '+' -> Cross, 'x' -> X
#   filter 키  : 'cross' -> Cross, 'x' -> X
LABEL_ALIASES = {
    "+": LABEL_CROSS,
    "cross": LABEL_CROSS,
    "x": LABEL_X,
}

FILTER_KEY_RE = re.compile(r"^size_(\d+)$")
PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(\d+)$")

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data.json"
)

# 실패 원인 분류
CAT_SCHEMA = "데이터/스키마"
CAT_LABEL = "라벨 정규화"
CAT_TIE = "수치 비교(동점)"
CAT_MISMATCH = "판정 불일치"


class ValidationError(ValueError):
    """입력/스키마 검증 실패."""


# ---------------------------------------------------------------------------
# 1. 데이터 구조: n×n 행렬
# ---------------------------------------------------------------------------
class Matrix:
    """n×n 크기의 2차원 패턴/필터 저장소."""

    def __init__(self, n, fill=0.0):
        if not isinstance(n, int) or n <= 0:
            raise ValidationError(f"행렬 크기는 1 이상의 정수여야 합니다 (받은 값: {n!r})")
        self.n = n
        self.data = [[float(fill)] * n for _ in range(n)]

    def get(self, row, col):
        """(row, col) 위치의 값을 읽는다."""
        self._check_index(row, col)
        return self.data[row][col]

    def set(self, row, col, value):
        """(row, col) 위치에 값을 저장한다."""
        self._check_index(row, col)
        self.data[row][col] = float(value)

    def _check_index(self, row, col):
        if not (0 <= row < self.n and 0 <= col < self.n):
            raise IndexError(f"위치 ({row}, {col})는 {self.n}×{self.n} 범위를 벗어납니다")

    @classmethod
    def from_rows(cls, rows):
        """2차원 리스트를 검증해 Matrix로 만든다. (정사각형, 유한한 숫자만 허용)"""
        if not isinstance(rows, list) or len(rows) == 0:
            raise ValidationError("2차원 배열(리스트의 리스트)이 아니거나 비어 있습니다")
        n = len(rows)
        matrix = cls(n)
        for r, row in enumerate(rows):
            if not isinstance(row, list):
                raise ValidationError(f"{r + 1}행이 배열이 아닙니다")
            if len(row) != n:
                raise ValidationError(
                    f"{r + 1}행의 열 수({len(row)})가 행 수({n})와 달라 정사각 행렬이 아닙니다"
                )
            for c, value in enumerate(row):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValidationError(f"({r + 1}, {c + 1}) 값 {value!r}은(는) 숫자가 아닙니다")
                if not math.isfinite(value):
                    raise ValidationError(f"({r + 1}, {c + 1}) 값 {value!r}은(는) 유한한 숫자가 아닙니다")
                matrix.set(r, c, value)
        return matrix

    def size_text(self):
        return f"{self.n}×{self.n}"

    def flatten(self):
        """[보너스 1] 행 우선(row-major) 순서의 1차원 리스트(길이 N²)로 변환한다.

        (row, col) 위치는 1차원에서 k = row * n + col 이 된다.
        """
        flat = []
        for row in self.data:
            flat.extend(row)
        return flat


# ---------------------------------------------------------------------------
# [보너스 2] 패턴 생성기
# ---------------------------------------------------------------------------
def center_indices(n):
    """가운데 줄 번호. 홀수 N은 1줄, 짝수 N은 정확한 중앙이 없으므로 가운데 2줄."""
    if n % 2 == 1:
        return {n // 2}
    return {n // 2 - 1, n // 2}


def generate_pattern(label, n):
    """크기 N의 Cross(십자가) 또는 X 패턴을 1.0/0.0 값으로 만든다."""
    if not isinstance(n, int) or n < 1:
        raise ValidationError(f"패턴 크기는 1 이상의 정수여야 합니다 (받은 값: {n!r})")
    if label not in (LABEL_CROSS, LABEL_X):
        raise ValidationError(f"생성할 수 없는 라벨입니다: {label!r}")
    matrix = Matrix(n)
    centers = center_indices(n)
    for i in range(n):
        for j in range(n):
            if label == LABEL_CROSS:
                on = i in centers or j in centers      # 가운데 행 또는 가운데 열
            else:
                on = i == j or i + j == n - 1          # 주대각선 또는 반대 대각선
            if on:
                matrix.set(i, j, 1.0)
    return matrix


# ---------------------------------------------------------------------------
# 2. 라벨 정규화
# ---------------------------------------------------------------------------
def normalize_label(raw):
    """'+', 'x', 'cross' 등 다양한 표기를 표준 라벨(Cross/X)로 바꾼다. 실패 시 None."""
    if not isinstance(raw, str):
        return None
    return LABEL_ALIASES.get(raw.strip().lower())


# ---------------------------------------------------------------------------
# 3. MAC 연산 / 판정
# ---------------------------------------------------------------------------
def mac(pattern, filt):
    """같은 위치끼리 곱하고(Multiply) 모두 더한다(Accumulate). 반복문으로 직접 구현."""
    if pattern.n != filt.n:
        raise ValidationError(f"크기 불일치: 패턴 {pattern.size_text()} ≠ 필터 {filt.size_text()}")
    n = pattern.n
    p_data = pattern.data
    f_data = filt.data
    total = 0.0
    for i in range(n):
        p_row = p_data[i]
        f_row = f_data[i]
        for j in range(n):
            total += p_row[j] * f_row[j]
    return total


def mac_flat(p_flat, f_flat):
    """[보너스 1] 1차원 배열(길이 N²) 버전 MAC. 이중 반복문 대신 인덱스 k 하나로 순회한다."""
    if len(p_flat) != len(f_flat):
        raise ValidationError(f"크기 불일치: 패턴 길이 {len(p_flat)} ≠ 필터 길이 {len(f_flat)}")
    total = 0.0
    for k in range(len(p_flat)):
        total += p_flat[k] * f_flat[k]
    return total


def is_tie(score_a, score_b, epsilon=EPSILON):
    """허용오차 기반 동점 판정."""
    return abs(score_a - score_b) < epsilon


def judge(score_a, score_b, label_a, label_b):
    """두 점수를 비교해 더 큰 쪽의 라벨을 돌려준다. 동점이면 UNDECIDED."""
    if is_tie(score_a, score_b):
        return LABEL_UNDECIDED
    return label_a if score_a > score_b else label_b


def naive_judge(score_a, score_b, label_a, label_b):
    """(진단용) 허용오차 없이 == / > 로만 비교했을 때의 판정."""
    if score_a == score_b:
        return LABEL_UNDECIDED
    return label_a if score_a > score_b else label_b


# ---------------------------------------------------------------------------
# 4. 성능 측정
# ---------------------------------------------------------------------------
def measure_ms(func, repeat=REPEAT):
    """func 호출 구간만 repeat회 측정해 평균(ms)을 돌려준다. (I/O 제외)

    측정 중에는 가비지 컬렉터를 잠시 멈춘다 (timeit과 같은 방식). 우연히 GC가
    한 번의 측정에 끼어들어 평균을 튀게 만드는 것을 막기 위해서다.
    """
    func()  # 워밍업 1회 (측정에서 제외)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        total = 0.0
        for _ in range(repeat):
            start = time.perf_counter()
            func()
            total += time.perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()
    return total / repeat * 1000.0


def calibrate_batch(func, target_ms=OPT_TARGET_MS):
    """[보너스 1] 1회 측정 구간이 target_ms 이상이 되도록 연속 호출 횟수(batch)를 정한다.

    3×3 MAC 한 번은 1µs도 안 걸려서 타이머 오차·잡음이 결과를 좌우한다.
    여러 번을 묶어 한 구간으로 재면 1회당 시간을 안정적으로 얻을 수 있다.
    """
    batch = 1
    while batch < OPT_MAX_BATCH:
        start = time.perf_counter()
        for _ in range(batch):
            func()
        if (time.perf_counter() - start) * 1000.0 >= target_ms:
            break
        batch *= 2
    return batch


def measure_pair_ms(func_a, func_b, repeat=REPEAT, batch=1):
    """[보너스 1] 두 함수를 같은 횟수만큼 번갈아 측정해 (a 1회 평균 ms, b 1회 평균 ms)를 돌려준다.

    - repeat회 측정하고, 1회 측정마다 같은 함수를 batch번 연속 호출한다. (두 함수 모두 같은 횟수)
    - 한쪽을 몰아서 재면 CPU 클럭 변화나 캐시 상태가 한쪽에만 유리할 수 있어 번갈아 잰다.
    """
    func_a()  # 워밍업 (측정에서 제외)
    func_b()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        total_a = total_b = 0.0
        for _ in range(repeat):
            start = time.perf_counter()
            for _ in range(batch):
                func_a()
            total_a += time.perf_counter() - start
            start = time.perf_counter()
            for _ in range(batch):
                func_b()
            total_b += time.perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()
    calls = repeat * batch
    return total_a / calls * 1000.0, total_b / calls * 1000.0


def benchmark_mac(pattern, filt, repeat=REPEAT):
    return measure_ms(lambda: mac(pattern, filt), repeat)


def benchmark_2d_vs_1d(pattern, filt, repeat=REPEAT):
    """[보너스 1] 같은 입력·같은 반복 횟수로 2D MAC과 1D MAC을 비교한다.

    1차원 변환(flatten)은 데이터 로드 단계에서 한 번만 하는 전처리로 보고 측정 구간에서 뺀다.
    반환: {"n", "ms_2d", "ms_1d", "batch", "same"}  (same: 두 방식의 점수가 비트 단위까지 같은지)
    """
    p_flat = pattern.flatten()
    f_flat = filt.flatten()
    run_2d = lambda: mac(pattern, filt)
    run_1d = lambda: mac_flat(p_flat, f_flat)
    batch = calibrate_batch(run_2d)
    ms_2d, ms_1d = measure_pair_ms(run_2d, run_1d, repeat, batch)
    same = run_2d() == run_1d()
    return {"n": pattern.n, "ms_2d": ms_2d, "ms_1d": ms_1d, "batch": batch, "same": same}


# ---------------------------------------------------------------------------
# 콘솔 출력 도우미
# ---------------------------------------------------------------------------
def display_width(text):
    """한글 등 전각 문자를 2칸으로 계산한 표시 폭."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text, width, align="left"):
    text = str(text)
    space = max(0, width - display_width(text))
    return (" " * space + text) if align == "right" else (text + " " * space)


def section(title):
    line = "#" + "-" * 40
    print()
    print(line)
    print(f"# {title}")
    print(line)


def print_matrix(matrix, indent="  "):
    for r in range(matrix.n):
        print(indent + " ".join(f"{matrix.get(r, c):g}" for c in range(matrix.n)))


def print_perf_table(rows, repeat=REPEAT):
    """rows: [(n, avg_ms), ...]"""
    widths = (10, 16, 16, 18)
    header = (
        pad("크기", widths[0])
        + pad("평균 시간(ms)", widths[1], "right")
        + pad("연산 횟수(N²)", widths[2], "right")
        + pad("MAC 1회당(ns)", widths[3], "right")
    )
    print(header)
    print("-" * sum(widths))
    for n, avg_ms in rows:
        ops = n * n
        per_op_ns = avg_ms * 1e6 / ops
        print(
            pad(f"{n}×{n}", widths[0])
            + pad(f"{avg_ms:.4f}", widths[1], "right")
            + pad(str(ops), widths[2], "right")
            + pad(f"{per_op_ns:.1f}", widths[3], "right")
        )
    print(f"(각 크기별 MAC 함수 호출 구간만 {repeat}회 측정한 평균, 워밍업 1회 제외)")


def print_optimization_table(results, repeat=REPEAT):
    """[보너스 1] 2D vs 1D 최적화 전/후 비교표."""
    widths = (10, 14, 14, 12, 14, 12)
    print(
        pad("크기", widths[0])
        + pad("2D 전(ms)", widths[1], "right")
        + pad("1D 후(ms)", widths[2], "right")
        + pad("속도 비율", widths[3], "right")
        + pad("측정당 호출", widths[4], "right")
        + pad("점수 일치", widths[5], "right")
    )
    print("-" * sum(widths))
    for r in results:
        ratio = r["ms_2d"] / r["ms_1d"] if r["ms_1d"] > 0 else float("inf")
        print(
            pad(f"{r['n']}×{r['n']}", widths[0])
            + pad(f"{r['ms_2d']:.5f}", widths[1], "right")
            + pad(f"{r['ms_1d']:.5f}", widths[2], "right")
            + pad(f"{ratio:.2f}x", widths[3], "right")
            + pad(str(r["batch"]), widths[4], "right")
            + pad("✓" if r["same"] else "✗", widths[5], "right")
        )
    print(f"(MAC 1회당 평균 시간 · 같은 입력으로 2D/1D를 {repeat}회씩 번갈아 측정, 1회 측정 = '측정당 호출'만큼 연속 실행)")
    print("(속도 비율 = 2D 시간 ÷ 1D 시간 → 1보다 크면 1D가 빠름 · 점수 일치 = 두 방식의 MAC 결과가 완전히 같은지)")


def run_performance(cases, repeat=REPEAT):
    """cases: [(pattern, filter), ...] → 성능 표(필수)와 2D/1D 비교표(보너스)를 함께 출력."""
    print_perf_table([(p.n, benchmark_mac(p, f, repeat)) for p, f in cases], repeat)
    results = [benchmark_2d_vs_1d(p, f, repeat) for p, f in cases]
    print()
    print("[최적화 비교: 2차원 배열 → 1차원 배열(길이 N²)]")
    print_optimization_table(results, repeat)


# ---------------------------------------------------------------------------
# 입력 도우미
# ---------------------------------------------------------------------------
def read_line(prompt=""):
    line = input(prompt)
    if not sys.stdin.isatty():
        # 파이프/리다이렉트 입력일 때 입력값을 화면에 남겨 실행 기록을 읽기 쉽게 한다.
        print(line)
    return line


def parse_row(line, n):
    """공백으로 구분된 n개의 숫자 한 줄을 파싱한다."""
    tokens = line.split()
    if len(tokens) == 0:
        raise ValidationError(
            f"빈 줄이 입력되었습니다. 행 수가 부족합니다 — {n}개의 숫자를 공백으로 구분해 입력하세요."
        )
    if len(tokens) != n:
        raise ValidationError(
            f"각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요. (입력된 값 {len(tokens)}개)"
        )
    values = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError:
            raise ValidationError(f"'{token}'은(는) 숫자로 변환할 수 없습니다. 숫자만 입력하세요.")
        if not math.isfinite(value):
            raise ValidationError(f"'{token}'은(는) 유한한 숫자가 아닙니다.")
        values.append(value)
    return values


def read_matrix(title, n=USER_INPUT_SIZE):
    """n줄을 한 줄씩 입력받는다. 잘못된 줄은 안내 후 그 줄만 다시 입력받는다.

    [보너스 2] 첫 줄에 숫자 대신 cross / + / x 를 입력하면 패턴 생성기로 n×n 전체를 채운다.
    """
    print(f"{title} ({n}줄 입력, 공백 구분 · 첫 줄에 cross 또는 x 입력 시 자동 생성)")
    matrix = Matrix(n)
    row = 0
    while row < n:
        line = read_line()
        label = normalize_label(line)
        if label is not None:
            if row == 0:
                generated = generate_pattern(label, n)
                print(f"  → 패턴 생성기: {n}×{n} {label} 패턴으로 채웠습니다")
                print_matrix(generated, indent="    ")
                return generated
            print("  입력 형식 오류: 자동 생성(cross/x)은 첫 줄에서만 사용할 수 있습니다.")
            print(f"  → {row + 1}번째 줄을 다시 입력하세요.")
            continue
        try:
            values = parse_row(line, n)
        except ValidationError as err:
            print(f"  입력 형식 오류: {err}")
            print(f"  → {row + 1}번째 줄을 다시 입력하세요.")
            continue
        for col, value in enumerate(values):
            matrix.set(row, col, value)
        row += 1
    return matrix


def choose_mode():
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    print("3. 패턴 생성기 + 최적화 비교 (보너스)")
    while True:
        choice = read_line("선택: ").strip()
        if choice in ("1", "2", "3"):
            return choice
        print("  잘못된 선택입니다. 1, 2, 3 중 하나를 입력하세요.")


# ---------------------------------------------------------------------------
# 모드 1: 사용자 입력 (3×3)
# ---------------------------------------------------------------------------
def run_user_mode():
    n = USER_INPUT_SIZE

    section("[1] 필터 입력")
    filter_a = read_matrix("필터 A", n)
    print()
    filter_b = read_matrix("필터 B", n)

    print()
    print("✓ 필터 저장 완료")
    print("필터 A:")
    print_matrix(filter_a)
    print("필터 B:")
    print_matrix(filter_b)

    section("[2] 패턴 입력")
    pattern = read_matrix("패턴", n)

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    elapsed = measure_ms(lambda: (mac(pattern, filter_a), mac(pattern, filter_b)))
    verdict = judge(score_a, score_b, "A", "B")

    section("[3] MAC 결과")
    print(f"A 점수: {score_a!r}")
    print(f"B 점수: {score_b!r}")
    print(f"연산 시간(평균/{REPEAT}회): {elapsed:.4f} ms  (필터 A·B MAC 2회 합산)")
    if verdict == LABEL_UNDECIDED:
        print(f"판정: 판정 불가 (|A-B| = {abs(score_a - score_b):.3g} < {EPSILON_TEXT})")
    else:
        print(f"판정: {verdict}")

    section(f"[4] 성능 분석 (평균/{REPEAT}회)")
    run_performance([(pattern, filter_a)])


# ---------------------------------------------------------------------------
# 모드 2: data.json 분석
# ---------------------------------------------------------------------------
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp), None
    except FileNotFoundError:
        return None, f"파일을 찾을 수 없습니다: {path}"
    except json.JSONDecodeError as err:
        return None, f"JSON 문법 오류 ({path}): {err}"
    except (OSError, UnicodeDecodeError) as err:
        return None, f"파일을 읽을 수 없습니다 ({path}): {err}"


def load_filters(data):
    """filters 섹션을 읽어 {N: {"Cross": Matrix, "X": Matrix}} 와 {N: 실패 사유} 를 돌려준다."""
    filters = {}
    errors = {}
    section_data = data.get("filters") if isinstance(data, dict) else None
    if not isinstance(section_data, dict):
        print("✗ 'filters' 섹션이 없거나 객체가 아닙니다")
        return filters, errors

    for key, entry in section_data.items():
        match = FILTER_KEY_RE.match(key)
        if not match:
            print(f"✗ {key}: 필터 키 형식 오류 (size_{{N}} 형식이어야 함) — 건너뜀")
            continue
        n = int(match.group(1))
        try:
            filters[n] = parse_filter_entry(n, entry)
            print(f"✓ {pad(key, 8)} 필터 로드 완료 ({LABEL_CROSS}, {LABEL_X})")
        except ValidationError as err:
            errors[n] = f"size_{n} 필터 로드 실패: {err}"
            print(f"✗ {pad(key, 8)} 필터 로드 실패: {err}")
    return filters, errors


def parse_filter_entry(n, entry):
    if not isinstance(entry, dict):
        raise ValidationError("필터 항목이 객체가 아닙니다")
    result = {}
    for raw_key, rows in entry.items():
        label = normalize_label(raw_key)
        if label is None:
            raise ValidationError(f"알 수 없는 필터 키 '{raw_key}' (cross 또는 x 필요)")
        if label in result:
            raise ValidationError(f"필터 키 '{raw_key}'가 {label} 필터와 중복됩니다")
        try:
            matrix = Matrix.from_rows(rows)
        except ValidationError as err:
            raise ValidationError(f"{label} 필터: {err}")
        if matrix.n != n:
            raise ValidationError(f"{label} 필터 크기 {matrix.size_text()}가 키의 N({n})과 다릅니다")
        result[label] = matrix
    missing = [label for label in (LABEL_CROSS, LABEL_X) if label not in result]
    if missing:
        raise ValidationError(f"{', '.join(missing)} 필터가 없습니다")
    return result


def make_result(case_id, status, *, category=None, reason=None, scores=None,
                verdict=None, expected=None, raw_expected=None):
    return {
        "id": case_id,
        "status": status,
        "category": category,
        "reason": reason,
        "scores": scores,
        "verdict": verdict,
        "expected": expected,
        "raw_expected": raw_expected,
    }


def evaluate_case(case_id, entry, filters, filter_errors):
    """패턴 1개를 검증·연산·판정한다. 어떤 오류든 예외 대신 FAIL 결과로 돌려준다."""
    fail = lambda cat, msg, **kw: make_result(case_id, "FAIL", category=cat, reason=msg, **kw)

    match = PATTERN_KEY_RE.match(case_id)
    if not match:
        return fail(CAT_SCHEMA, "패턴 키 형식 오류 (size_{N}_{idx} 형식이어야 함)")
    n = int(match.group(1))

    if not isinstance(entry, dict):
        return fail(CAT_SCHEMA, "패턴 항목이 객체가 아닙니다")
    raw_expected = entry.get("expected")
    expected = normalize_label(raw_expected)

    if n in filter_errors:
        return fail(CAT_SCHEMA, filter_errors[n], raw_expected=raw_expected, expected=expected)
    if n not in filters:
        return fail(CAT_SCHEMA, f"키에서 추출한 N={n}에 해당하는 size_{n} 필터가 없습니다",
                    raw_expected=raw_expected, expected=expected)
    if "input" not in entry:
        return fail(CAT_SCHEMA, "'input' 필드가 없습니다", raw_expected=raw_expected, expected=expected)

    try:
        pattern = Matrix.from_rows(entry["input"])
    except ValidationError as err:
        return fail(CAT_SCHEMA, f"input 형식 오류: {err}", raw_expected=raw_expected, expected=expected)

    cross_filter = filters[n][LABEL_CROSS]
    x_filter = filters[n][LABEL_X]
    if pattern.n != cross_filter.n:
        return fail(
            CAT_SCHEMA,
            f"크기 불일치: 패턴 {pattern.size_text()} ≠ size_{n} 필터 {cross_filter.size_text()}",
            raw_expected=raw_expected, expected=expected,
        )

    score_cross = mac(pattern, cross_filter)
    score_x = mac(pattern, x_filter)
    scores = (score_cross, score_x)
    verdict = judge(score_cross, score_x, LABEL_CROSS, LABEL_X)
    common = dict(scores=scores, verdict=verdict, expected=expected, raw_expected=raw_expected)

    if expected is None:
        if "expected" not in entry:
            return fail(CAT_SCHEMA, "'expected' 필드가 없습니다", **common)
        return fail(CAT_LABEL, f"expected 값 {raw_expected!r}을(를) 표준 라벨로 정규화할 수 없습니다", **common)

    if verdict == LABEL_UNDECIDED:
        naive = naive_judge(score_cross, score_x, LABEL_CROSS, LABEL_X)
        if naive == LABEL_UNDECIDED:
            note = "두 점수가 정확히 같음"
        else:
            note = f"epsilon 없이 비교했다면 오차 때문에 '{naive}'(으)로 판정됨"
        return fail(
            CAT_TIE,
            f"동점(UNDECIDED) 규칙에 따라 FAIL — |Cross-X| = {abs(score_cross - score_x):.3g} < {EPSILON_TEXT}"
            f" ({note})",
            **common,
        )
    if verdict != expected:
        return fail(CAT_MISMATCH, f"판정 {verdict} ≠ expected {expected} (라벨 오류 또는 필터/로직 점검 필요)",
                    **common)
    return make_result(case_id, "PASS", **common)


def print_case(result):
    print(f"--- {result['id']} ---")
    if result["scores"] is not None:
        score_cross, score_x = result["scores"]
        print(f"{LABEL_CROSS} 점수: {score_cross!r}")
        print(f"{LABEL_X} 점수: {score_x!r}")
    if result["expected"] is not None:
        expected_text = f"{result['expected']} (원본 {result['raw_expected']!r})"
    elif result["raw_expected"] is not None:
        expected_text = f"? (원본 {result['raw_expected']!r})"
    else:
        expected_text = "-"
    verdict_text = result["verdict"] or "-"
    line = f"판정: {verdict_text} | expected: {expected_text} | {result['status']}"
    if result["status"] == "FAIL":
        line += f" [{result['category']}]"
    print(line)
    if result["status"] == "FAIL":
        print(f"  사유: {result['reason']}")


def run_json_mode(path):
    section("[1] 필터 로드")
    print(f"파일: {path}")
    data, error = load_json(path)
    if error:
        print(f"✗ {error}")
        data, filters, filter_errors = {}, {}, {}
    else:
        filters, filter_errors = load_filters(data)

    section("[2] 패턴 분석 (라벨 정규화 적용)")
    patterns = data.get("patterns") if isinstance(data, dict) else None
    results = []
    if not isinstance(patterns, dict):
        print("✗ 'patterns' 섹션이 없거나 객체가 아닙니다 — 분석할 케이스가 없습니다")
    else:
        for case_id, entry in patterns.items():
            try:
                result = evaluate_case(case_id, entry, filters, filter_errors)
            except Exception as err:  # 예상 못 한 오류도 케이스 단위로 격리
                result = make_result(case_id, "FAIL", category=CAT_SCHEMA,
                                     reason=f"처리 중 예외 발생: {type(err).__name__}: {err}")
            results.append(result)
            print_case(result)

    section(f"[3] 성능 분석 (평균/{REPEAT}회)")
    run_performance(benchmark_cases(filters, patterns))

    section("[4] 결과 요약")
    print_summary(results)


def benchmark_cases(filters, patterns):
    """성능 분석용 (패턴, 필터) 목록: 3×3 + 로드된 필터 크기들.

    3×3은 data.json에 없으므로 패턴 생성기로 만든 X 패턴 × Cross 필터를 쓴다. (보너스 2 재활용)
    다른 크기는 data.json의 실제 패턴을 쓰고, 없으면 생성기 패턴으로 대신한다.
    """
    cases = [(generate_pattern(LABEL_X, 3), generate_pattern(LABEL_CROSS, 3))]
    for n in sorted(filters):
        if n == 3:
            continue
        pattern = find_sample_pattern(n, patterns) or generate_pattern(LABEL_X, n)
        cases.append((pattern, filters[n][LABEL_CROSS]))
    return cases


def find_sample_pattern(n, patterns):
    if not isinstance(patterns, dict):
        return None
    for case_id, entry in patterns.items():
        match = PATTERN_KEY_RE.match(case_id)
        if match and int(match.group(1)) == n and isinstance(entry, dict):
            try:
                matrix = Matrix.from_rows(entry.get("input"))
            except ValidationError:
                continue
            if matrix.n == n:
                return matrix
    return None


def print_summary(results):
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    print(f"총 테스트: {len(results)}개")
    print(f"통과: {len(passed)}개")
    print(f"실패: {len(failed)}개")
    if failed:
        print()
        print("실패 케이스:")
        for r in failed:
            print(f"- {r['id']} [{r['category']}]: {r['reason']}")
        counts = {}
        for r in failed:
            counts[r["category"]] = counts.get(r["category"], 0) + 1
        print()
        print("실패 원인 분류: " + ", ".join(f"{cat} {cnt}건" for cat, cnt in counts.items()))


# ---------------------------------------------------------------------------
# 모드 3 (보너스): 패턴 생성기 + 최적화 비교
# ---------------------------------------------------------------------------
def parse_sizes(line):
    """'3 5 13' 같은 크기 목록을 파싱한다. 빈 줄이면 기본 크기 목록."""
    tokens = line.split()
    if not tokens:
        return list(GEN_DEFAULT_SIZES)
    sizes = []
    for token in tokens:
        try:
            n = int(token)
        except ValueError:
            raise ValidationError(f"'{token}'은(는) 정수가 아닙니다.")
        if not GEN_MIN_SIZE <= n <= GEN_MAX_SIZE:
            raise ValidationError(f"크기 {n}은(는) {GEN_MIN_SIZE}~{GEN_MAX_SIZE} 범위여야 합니다.")
        if n not in sizes:
            sizes.append(n)
    return sizes


def print_preview(cross, x):
    """Cross / X 패턴을 나란히 ■□로 출력한다."""
    width = cross.n * 2 + 4
    print(pad(LABEL_CROSS, width) + LABEL_X)
    for r in range(cross.n):
        left = "".join("■" if cross.get(r, c) else "□" for c in range(cross.n))
        right = "".join("■" if x.get(r, c) else "□" for c in range(x.n))
        print(pad(left, width) + right)


def run_generator_mode():
    section("[1] 크기 입력")
    print(f"생성할 크기 N을 공백으로 구분해 입력하세요 ({GEN_MIN_SIZE}~{GEN_MAX_SIZE}, "
          f"빈 줄 = 기본값 {' '.join(map(str, GEN_DEFAULT_SIZES))})")
    while True:
        try:
            sizes = parse_sizes(read_line("크기: "))
            break
        except ValidationError as err:
            print(f"  입력 형식 오류: {err} → 다시 입력하세요.")

    section("[2] 패턴 생성")
    generated = {}
    for n in sizes:
        cross = generate_pattern(LABEL_CROSS, n)
        x = generate_pattern(LABEL_X, n)
        generated[n] = (cross, x)
        note = " (짝수 N: 가운데 2줄)" if n % 2 == 0 else ""
        print(f"✓ {n}×{n} 생성 완료 — Cross 칸 {sum(cross.flatten()):g}개{note}, X 칸 {sum(x.flatten()):g}개")
    small = [n for n in sizes if n <= PREVIEW_MAX_SIZE]
    for n in small:
        print()
        print(f"[{n}×{n} 미리보기]")
        print_preview(*generated[n])
    if len(small) < len(sizes):
        print()
        print(f"({PREVIEW_MAX_SIZE}×{PREVIEW_MAX_SIZE}보다 큰 패턴은 미리보기를 생략합니다)")

    section("[3] 생성 패턴 판정 검증 (입력과 필터 모두 생성 패턴)")
    total = passed = 0
    for n in sizes:
        cross, x = generated[n]
        for expected, pattern in ((LABEL_CROSS, cross), (LABEL_X, x)):
            score_cross = mac(pattern, cross)
            score_x = mac(pattern, x)
            verdict = judge(score_cross, score_x, LABEL_CROSS, LABEL_X)
            ok = verdict == expected
            total += 1
            passed += ok
            print(f"{pad(f'{n}×{n} {expected} 입력', 16)} Cross 점수 {score_cross:>5g} | X 점수 {score_x:>5g}"
                  f" | 판정 {pad(verdict, 9)} | {'PASS' if ok else 'FAIL'}")
    print(f"→ {total}개 중 {passed}개 PASS")

    section(f"[4] 성능 분석 (평균/{REPEAT}회 · 입력 = 생성 X 패턴, 필터 = 생성 Cross 패턴)")
    run_performance([(generated[n][1], generated[n][0]) for n in sizes])


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def main(argv):
    # 출력 리다이렉트 시 인코딩 문제로 프로그램이 멈추지 않도록 한다.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    data_path = argv[1] if len(argv) > 1 else DEFAULT_DATA_PATH

    print("=== Mini NPU Simulator ===")
    print()
    try:
        mode = choose_mode()
        if mode == "1":
            run_user_mode()
        elif mode == "2":
            run_json_mode(data_path)
        else:
            run_generator_mode()
    except (EOFError, KeyboardInterrupt):
        print()
        print("입력이 종료되어 프로그램을 마칩니다.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
