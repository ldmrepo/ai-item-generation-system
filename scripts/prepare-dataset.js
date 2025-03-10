// scripts/prepare-dataset.js
const fs = require("fs");
const path = require("path");

// 데이터셋 JSON
const knowledgeMapDataset = {
    knowledge_map: {
        map_info: {
            subject: "수학",
            grade: "중학교 2학년",
            unit: "함수",
            version: "1.0",
            creation_date: "2025-02-25",
            description:
                "중학교 2학년 함수 영역의 핵심 개념과 관계를 구조화한 지식맵",
        },
        nodes: [
            {
                node_id: "FUN-001",
                concept_name: "함수의 개념",
                definition:
                    "한 집합의 원소가 다른 집합의 원소에 일대일 또는 다대일로 대응하는 관계",
                difficulty_level: "기초",
                learning_time: 45,
                curriculum_mapping: {
                    education_system: "국가교육과정",
                    grade: "중학교 2학년",
                    unit: "함수",
                    achievement_standard:
                        "함수의 개념을 이해하고, 함수의 값을 구할 수 있다.",
                },
                misconceptions: [
                    "함수와 방정식의 혼동",
                    "일대일 대응만 함수로 인식하는 오류",
                    "함수를 공식으로만 한정하여 이해하는 오류",
                ],
            },
            {
                node_id: "FUN-002",
                concept_name: "좌표평면과 순서쌍",
                definition:
                    "x축과 y축으로 이루어진 2차원 평면에서 점의 위치를 순서쌍 (x, y)로 표현하는 체계",
                difficulty_level: "기초",
                learning_time: 40,
                curriculum_mapping: {
                    education_system: "국가교육과정",
                    grade: "중학교 2학년",
                    unit: "함수",
                    achievement_standard:
                        "좌표평면에서 점의 위치를 이해하고 표현할 수 있다.",
                },
                misconceptions: [
                    "x좌표와 y좌표 순서 혼동",
                    "원점의 좌표 오인",
                    "좌표축 눈금 해석 오류",
                ],
            },
            {
                node_id: "FUN-003",
                concept_name: "일차함수의 개념",
                definition:
                    "y = ax + b 형태로 표현되는 함수로, x의 계수 a가 0이 아닌 1차식으로 표현됨",
                difficulty_level: "중급",
                learning_time: 50,
                curriculum_mapping: {
                    education_system: "국가교육과정",
                    grade: "중학교 2학년",
                    unit: "함수",
                    achievement_standard:
                        "일차함수의 의미를 이해하고, 그 그래프의 성질을 이해한다.",
                },
                misconceptions: [
                    "계수 a가 0인 경우도 일차함수로 오인",
                    "일차함수와 일차방정식 간의 관계 혼동",
                    "변화율 개념 이해 부족",
                ],
            },
            {
                node_id: "FUN-004",
                concept_name: "일차함수의 그래프",
                definition:
                    "일차함수 y = ax + b의 그래프는 좌표평면 위에서 직선으로 나타남",
                difficulty_level: "중급",
                learning_time: 55,
                curriculum_mapping: {
                    education_system: "국가교육과정",
                    grade: "중학교 2학년",
                    unit: "함수",
                    achievement_standard:
                        "일차함수의 그래프를 그릴 수 있고, 그 성질을 이해한다.",
                },
                misconceptions: [
                    "y절편 개념 혼동",
                    "x축, y축과의 교점 찾기 오류",
                    "그래프의 연속성 이해 부족",
                ],
            },
            {
                node_id: "FUN-005",
                concept_name: "일차함수의 기울기",
                definition:
                    "일차함수 y = ax + b에서 x가 1단위 증가할 때 y의 증가량을 나타내는 값 a",
                difficulty_level: "중급",
                learning_time: 60,
                curriculum_mapping: {
                    education_system: "국가교육과정",
                    grade: "중학교 2학년",
                    unit: "함수",
                    achievement_standard:
                        "일차함수의 기울기를 이해하고, 이를 활용하여 문제를 해결할 수 있다.",
                },
                misconceptions: [
                    "기울기와 y절편 혼동",
                    "기울기의 부호와 함수의 증감 관계 오해",
                    "두 점 사이의 기울기 계산 오류",
                ],
            },
            {
                node_id: "FUN-006",
                concept_name: "일차함수의 응용",
                definition: "실생활 상황을 일차함수로 모델링하고 해석하는 능력",
                difficulty_level: "고급",
                learning_time: 90,
                curriculum_mapping: {
                    education_system: "국가교육과정",
                    grade: "중학교 2학년",
                    unit: "함수",
                    achievement_standard:
                        "일차함수를 활용하여 실생활 문제를 해결할 수 있다.",
                },
                misconceptions: [
                    "변수 간 관계를 함수로 표현하는 어려움",
                    "문제 상황에서 기울기의 의미 해석 오류",
                    "맥락에 따른 함수식 구성 오류",
                ],
            },
            {
                node_id: "FUN-007",
                concept_name: "그래프의 해석",
                definition:
                    "일차함수 그래프로부터 다양한 정보를 추출하고 해석하는 능력",
                difficulty_level: "중급",
                learning_time: 70,
                curriculum_mapping: {
                    education_system: "국가교육과정",
                    grade: "중학교 2학년",
                    unit: "함수",
                    achievement_standard:
                        "일차함수의 그래프로부터 함수의 성질을 파악하고 문제를 해결할 수 있다.",
                },
                misconceptions: [
                    "그래프 상의 점 좌표 읽기 오류",
                    "기울기와 y절편 그래프에서 식별 오류",
                    "그래프 해석에서 맥락 고려 부족",
                ],
            },
            {
                node_id: "FUN-008",
                concept_name: "일차방정식과의 관계",
                definition:
                    "일차함수와 일차방정식의 관계 및 연립방정식의 기하학적 의미 이해",
                difficulty_level: "고급",
                learning_time: 80,
                curriculum_mapping: {
                    education_system: "국가교육과정",
                    grade: "중학교 2학년",
                    unit: "함수",
                    achievement_standard:
                        "일차함수와 일차방정식의 관계를 이해하고, 연립방정식의 해를 기하학적으로 해석할 수 있다.",
                },
                misconceptions: [
                    "방정식의 해와 함수의 그래프 관계 혼동",
                    "연립방정식의 해와 교점 관계 이해 부족",
                    "함수와 방정식의 본질적 차이 혼동",
                ],
            },
        ],
        edges: [
            {
                source: "FUN-001",
                target: "FUN-002",
                type: "prerequisite",
                strength: "strong",
            },
            {
                source: "FUN-001",
                target: "FUN-003",
                type: "prerequisite",
                strength: "strong",
            },
            {
                source: "FUN-002",
                target: "FUN-004",
                type: "prerequisite",
                strength: "strong",
            },
            {
                source: "FUN-003",
                target: "FUN-004",
                type: "prerequisite",
                strength: "strong",
            },
            {
                source: "FUN-003",
                target: "FUN-005",
                type: "prerequisite",
                strength: "medium",
            },
            {
                source: "FUN-003",
                target: "FUN-008",
                type: "prerequisite",
                strength: "medium",
            },
            {
                source: "FUN-004",
                target: "FUN-005",
                type: "prerequisite",
                strength: "strong",
            },
            {
                source: "FUN-004",
                target: "FUN-007",
                type: "prerequisite",
                strength: "strong",
            },
            {
                source: "FUN-004",
                target: "FUN-008",
                type: "related",
                strength: "medium",
            },
            {
                source: "FUN-005",
                target: "FUN-006",
                type: "prerequisite",
                strength: "strong",
            },
            {
                source: "FUN-005",
                target: "FUN-007",
                type: "prerequisite",
                strength: "medium",
            },
            {
                source: "FUN-005",
                target: "FUN-008",
                type: "related",
                strength: "weak",
            },
            {
                source: "FUN-006",
                target: "FUN-007",
                type: "related",
                strength: "medium",
            },
            {
                source: "FUN-007",
                target: "FUN-008",
                type: "related",
                strength: "medium",
            },
        ],
    },
    seed_items: {
        "FUN-001": [
            {
                item_id: "SEED-FUN-001-001",
                type: "conceptual",
                difficulty: "basic",
                content:
                    "함수의 정의를 설명하고, 함수와 함수가 아닌 관계의 예를 각각 1개씩 제시하시오.",
                answer: "함수는 첫 번째 집합의 모든 원소가 두 번째 집합의 원소에 한 개씩만 대응되는 관계이다. 함수의 예: x와 2x의 관계, 함수가 아닌 예: x와 ±√x의 관계",
                explanation:
                    "함수는 정의역의 모든 원소가 공역의 원소에 오직 하나씩만 대응되어야 한다. x와 2x의 관계는 각 x값에 대해 2x라는 하나의 값만 대응되므로 함수이다. 반면 x와 ±√x 관계에서는 x=4일 때 y=2 또는 y=-2가 대응되므로 함수가 아니다.",
                variation_points: ["definition", "examples"],
            },
            {
                item_id: "SEED-FUN-001-002",
                type: "application",
                difficulty: "basic",
                content: "함수 f(x) = 3x - 1에서 f(2)와 f(-1)의 값을 구하시오.",
                answer: "f(2) = 5, f(-1) = -4",
                explanation:
                    "f(2) = 3(2) - 1 = 6 - 1 = 5\nf(-1) = 3(-1) - 1 = -3 - 1 = -4",
                variation_points: ["function_rule", "input_values"],
            },
            {
                item_id: "SEED-FUN-001-003",
                type: "identification",
                difficulty: "intermediate",
                content:
                    "다음 중 함수를 나타내는 것을 모두 고르시오.\n(a) y = x²\n(b) x² + y² = 4\n(c) y² = x\n(d) |y| = x",
                answer: "(a), (d)",
                explanation:
                    "(a) y = x²는 각 x값에 대해 하나의 y값만 대응되므로 함수이다.\n(b) x² + y² = 4는 원의 방정식으로, 하나의 x값에 대해 두 개의 y값이 나올 수 있으므로 함수가 아니다.\n(c) y² = x에서는 x=4일 때 y=2 또는 y=-2가 대응되므로 함수가 아니다.\n(d) |y| = x는 x ≥ 0일 때만 정의되며, 이때 y = x 또는 y = -x로 나타낼 수 있다. y를 x의 함수로 보면 함수가 아니지만, x를 y의 함수로 보면 함수이다. 문제에서 y = f(x) 형태의 함수를 묻고 있으므로, 이는 함수가 아니다.",
                variation_points: ["function_types", "representation"],
            },
        ],
        "FUN-002": [
            {
                item_id: "SEED-FUN-002-001",
                type: "plotting",
                difficulty: "basic",
                content:
                    "좌표평면 위에 점 A(3, 4), B(-2, 1), C(0, -3)을 표시하시오.",
                answer: "[좌표평면에 세 점 표시한 그림]",
                explanation:
                    "점 A(3, 4)는 x축의 양의 방향으로 3칸, y축의 양의 방향으로 4칸 이동한 위치에 있다. 점 B(-2, 1)는 x축의 음의 방향으로 2칸, y축의 양의 방향으로 1칸 이동한 위치에 있다. 점 C(0, -3)는 x축 위치는 원점과 같고, y축의 음의 방향으로 3칸 이동한 위치에 있다.",
                variation_points: ["coordinates", "quadrants"],
            },
            {
                item_id: "SEED-FUN-002-002",
                type: "calculation",
                difficulty: "intermediate",
                content:
                    "좌표평면 위의 두 점 A(1, 3)과 B(5, 7) 사이의 거리를 구하시오.",
                answer: "5.66 (또는 √32)",
                explanation:
                    "두 점 A(x₁, y₁)과 B(x₂, y₂) 사이의 거리는 √[(x₂ - x₁)² + (y₂ - y₁)²]로 구할 수 있다. 따라서 A(1, 3)과 B(5, 7) 사이의 거리는 √[(5 - 1)² + (7 - 3)²] = √[16 + 16] = √32 ≈ 5.66이다.",
                variation_points: ["coordinates", "distance_formula"],
            },
            {
                item_id: "SEED-FUN-002-003",
                type: "analysis",
                difficulty: "intermediate",
                content:
                    "좌표평면 위의 점 (a, b)가 제1사분면에 있고, 원점으로부터의 거리가 5일 때, a²+b²의 값을 구하시오.",
                answer: "25",
                explanation:
                    "점 (a, b)가 원점 (0, 0)으로부터의 거리가 5라는 것은 √[(a - 0)² + (b - 0)²] = 5를 의미한다. 따라서 √(a² + b²) = 5이므로, a² + b² = 25이다. 또한 제1사분면에 있다는 것은 a > 0, b > 0을 의미한다.",
                variation_points: ["distance", "quadrant_constraints"],
            },
        ],
        "FUN-003": [
            {
                item_id: "SEED-FUN-003-001",
                type: "conceptual",
                difficulty: "basic",
                content:
                    "일차함수의 정의를 쓰고, y = 3x - 2가 일차함수인 이유를 설명하시오.",
                answer: "일차함수는 y = ax + b (a≠0) 형태로 나타낼 수 있는 함수이다. y = 3x - 2는 a=3, b=-2인 일차함수이다.",
                explanation:
                    "일차함수의 가장 큰 특징은 x의 계수 a가 0이 아닌 1차식으로 표현된다는 것이다. y = 3x - 2에서 x의 계수는 3으로 0이 아니고, x의 최고차항이 1차이므로 일차함수이다.",
                variation_points: ["definition", "example_analysis"],
            },
            {
                item_id: "SEED-FUN-003-002",
                type: "calculation",
                difficulty: "intermediate",
                content:
                    "일차함수 f(x)에서 f(2) = 7이고 f(5) = 16일 때, 함수식 f(x)를 구하시오.",
                answer: "f(x) = 3x + 1",
                explanation:
                    "일차함수를 f(x) = ax + b라고 하면, 주어진 조건에 따라\nf(2) = 2a + b = 7\nf(5) = 5a + b = 16\n두 식을 연립하여 해결하면,\n(5a + b) - (2a + b) = 16 - 7\n3a = 9\na = 3\n따라서 2a + b = 7에서 2(3) + b = 7, b = 7 - 6 = 1\n따라서 f(x) = 3x + 1이다.",
                variation_points: ["function_values", "coefficients"],
            },
            {
                item_id: "SEED-FUN-003-003",
                type: "application",
                difficulty: "advanced",
                content:
                    "일차함수 y = ax + b가 점 (1, 4)를 지나고 x축과 만나는 점의 x좌표가 3일 때, 상수 a와 b의 값을 구하시오.",
                answer: "a = -2, b = 6",
                explanation:
                    "일차함수 y = ax + b가 점 (1, 4)를 지나므로, 4 = a(1) + b, 즉 a + b = 4 ...(1)\n또한, 이 함수가 x축과 만나는 점의 x좌표가 3이라는 것은 y = 0일 때 x = 3임을 의미한다. 따라서 0 = a(3) + b, 즉 3a + b = 0 ...(2)\n(1)과 (2)를 연립하여 해결하면,\n(a + b) - (3a + b) = 4 - 0\n-2a = 4\na = -2\n따라서 (1)식에서 -2 + b = 4, b = 6\n따라서 a = -2, b = 6이다.",
                variation_points: ["constraints", "system_of_equations"],
            },
        ],
        "FUN-004": [
            {
                item_id: "SEED-FUN-004-001",
                type: "graphing",
                difficulty: "basic",
                content: "일차함수 y = 2x - 3의 그래프를 그리시오.",
                answer: "[y = 2x - 3의 그래프 그림]",
                explanation:
                    "일차함수 y = 2x - 3의 그래프를 그리기 위해 몇 개의 점을 구해보자.\nx = 0일 때, y = 2(0) - 3 = -3, 즉 점 (0, -3)이 그래프 위에 있다.\nx = 1일 때, y = 2(1) - 3 = -1, 즉 점 (1, -1)이 그래프 위에 있다.\nx = 2일 때, y = 2(2) - 3 = 1, 즉 점 (2, 1)이 그래프 위에 있다.\n이 세 점을 좌표평면 위에 표시하고 직선으로 연결하면 주어진 일차함수의 그래프가 된다.",
                variation_points: ["coefficients", "y_intercept"],
            },
            {
                item_id: "SEED-FUN-004-002",
                type: "interpretation",
                difficulty: "intermediate",
                content:
                    "일차함수 y = ax + b의 그래프가 점 (2, 3)을 지나고 y절편이 -1일 때, 이 함수의 식을 구하시오.",
                answer: "y = 2x - 1",
                explanation:
                    "y절편이 -1이라는 것은 그래프가 점 (0, -1)을 지난다는 것이다. 따라서 b = -1이므로, 함수식은 y = ax - 1 형태이다.\n이 함수가 점 (2, 3)을 지나므로, 3 = a(2) - 1, 즉 2a = 4, a = 2\n따라서 함수식은 y = 2x - 1이다.",
                variation_points: ["y_intercept", "point_constraint"],
            },
            {
                item_id: "SEED-FUN-004-003",
                type: "analysis",
                difficulty: "advanced",
                content:
                    "일차함수 y = kx + 2의 그래프가 점 (3, 5)를 지날 때, 이 그래프와 x축이 만나는 점의 x좌표를 구하시오.",
                answer: "-2",
                explanation:
                    "일차함수 y = kx + 2가 점 (3, 5)를 지나므로, 5 = k(3) + 2, 즉 3k = 3, k = 1\n따라서 함수식은 y = x + 2이다.\n이 그래프와 x축이 만나는 점에서는 y = 0이므로, 0 = x + 2, 즉 x = -2\n따라서 그래프와 x축이 만나는 점의 x좌표는 -2이다.",
                variation_points: ["intersection", "coefficient_determination"],
            },
        ],
        "FUN-005": [
            {
                item_id: "SEED-FUN-005-001",
                type: "calculation",
                difficulty: "basic",
                content: "일차함수 y = 3x - 2에서 기울기를 구하시오.",
                answer: "3",
                explanation:
                    "일차함수 y = ax + b에서 기울기는 a의 값이다. 따라서 y = 3x - 2에서 기울기는 3이다.",
                variation_points: ["coefficient", "operation"],
            },
            {
                item_id: "SEED-FUN-005-002",
                type: "graph_interpretation",
                difficulty: "intermediate",
                content:
                    "좌표평면 위의 두 점 (1, 3)과 (4, 12)를 지나는 일차함수의 기울기를 구하시오.",
                answer: "3",
                explanation:
                    "두 점 (x₁, y₁)과 (x₂, y₂)를 지나는 직선의 기울기는 (y₂ - y₁) / (x₂ - x₁)이다. 따라서 (4, 12)와 (1, 3)의 기울기는 (12 - 3) / (4 - 1) = 9 / 3 = 3이다.",
                variation_points: ["coordinates", "context"],
            },
            {
                item_id: "SEED-FUN-005-003",
                type: "conceptual",
                difficulty: "advanced",
                content:
                    "일차함수 y = ax + b에서 기울기 a가 의미하는 바를 설명하고, 기울기가 양수, 음수, 0일 때 그래프의 특징을 각각 설명하시오.",
                answer: "기울기 a는 x가 1단위 증가할 때 y의 증가량을 의미한다. 기울기가 양수이면 그래프는 오른쪽 위로 상승하고, 음수이면 오른쪽 아래로 하강한다. 기울기가 0이면 수평선이 된다.",
                explanation:
                    "기울기는 함수의 증감과 방향을 결정하는 중요한 요소이다. 기울기의 절댓값은 그래프의 가파른 정도를 나타낸다.",
                variation_points: ["concept_depth", "examples"],
            },
        ],
        "FUN-006": [
            {
                item_id: "SEED-FUN-006-001",
                type: "application",
                difficulty: "intermediate",
                content:
                    "물탱크에 물이 분당 5리터씩 채워지고 있다. 처음에 탱크에는 10리터의 물이 있었다. 시간 t(분)에 따른 물의 양 y(리터)를 일차함수로 나타내시오.",
                answer: "y = 5t + 10",
                explanation:
                    "분당 5리터씩 증가하므로 기울기는 5이고, 처음(t=0)에 10리터가 있으므로 y절편은 10이다. 따라서 y = 5t + 10이다.",
                variation_points: ["rate", "initial_value", "context"],
            },
            {
                item_id: "SEED-FUN-006-002",
                type: "problem_solving",
                difficulty: "advanced",
                content:
                    "택시 기본요금은 3,000원이며, 주행 거리 1km마다 추가로 1,000원씩 요금이 부과된다. 주행 거리 x(km)에 따른 택시 요금 y(원)를 일차함수로 나타내고, 택시 요금이 8,000원일 때 주행 거리를 구하시오.",
                answer: "y = 1000x + 3000, 5km",
                explanation:
                    "기본요금이 3,000원이므로 y절편은 3000이고, 1km마다 1,000원씩 증가하므로 기울기는 1000이다. 따라서 y = 1000x + 3000이다. 요금이 8,000원일 때, 8000 = 1000x + 3000, 따라서 x = 5이다.",
                variation_points: ["base_value", "rate", "target_value"],
            },
            {
                item_id: "SEED-FUN-006-003",
                type: "analysis",
                difficulty: "advanced",
                content:
                    "한 회사의 월별 수익 y(만원)는 생산량 x(개)에 따라 y = 2x - 100으로 나타낼 수 있다. 이 회사가 적자를 내지 않기 위한 최소 생산량은 얼마인지 구하고, 그 의미를 설명하시오.",
                answer: "50개 이상",
                explanation:
                    "수익이 0 이상이어야 적자가 아니므로, 0 ≤ 2x - 100, 따라서 x ≥ 50이다. 이는 최소 50개 이상을 생산해야 회사가 적자를 내지 않는다는 의미이다.",
                variation_points: [
                    "coefficients",
                    "constraint",
                    "interpretation",
                ],
            },
        ],
        "FUN-007": [
            {
                item_id: "SEED-FUN-007-001",
                type: "interpretation",
                difficulty: "intermediate",
                content:
                    "아래 그래프는 일차함수 y = ax + b의 그래프이다. 이 그래프로부터 상수 a와 b의 값을 구하시오.\n[y = ax + b 그래프, x축과는 (3, 0)에서, y축과는 (0, -6)에서 만나는 그래프]",
                answer: "a = 2, b = -6",
                explanation:
                    "그래프가 y축과 만나는 점이 (0, -6)이므로 y절편은 -6, 즉 b = -6이다. 그래프가 x축과 만나는 점이 (3, 0)이므로 0 = a(3) + (-6), 즉 3a = 6, a = 2이다. 따라서 함수식은 y = 2x - 6이다.",
                variation_points: ["intercepts", "coefficient_calculation"],
            },
            {
                item_id: "SEED-FUN-007-002",
                type: "analysis",
                difficulty: "advanced",
                content:
                    "두 일차함수 f(x) = 2x - 1과 g(x) = -x + 5의 그래프가 만나는 점의 좌표를 구하시오.",
                answer: "(2, 3)",
                explanation:
                    "두 함수의 그래프가 만나는 점에서는 함수값이 같다. 따라서 f(x) = g(x)를 만족하는 x값을 찾으면 된다.\n2x - 1 = -x + 5\n2x + x = 1 + 5\n3x = 6\nx = 2\n따라서 y = f(2) = 2(2) - 1 = 3\n그러므로 두 그래프가 만나는 점의 좌표는 (2, 3)이다.",
                variation_points: ["functions", "intersection"],
            },
            {
                item_id: "SEED-FUN-007-003",
                type: "real_world",
                difficulty: "advanced",
                content:
                    "아래 그래프는 어떤 물체의 시간에 따른 높이를 나타낸 것이다. 이 물체의 시간에 따른 높이 변화를 설명하고, t=5일 때의 높이를 구하시오.\n[y = -10x + 60 그래프, 축은 x: 시간(초), y: 높이(m)]",
                answer: "물체는 초당 10m씩 높이가 감소하고 있으며, t=5일 때 높이는 10m이다.",
                explanation:
                    "그래프는 y = -10x + 60 형태의 일차함수이다. 기울기가 -10이므로 시간이 1초 증가할 때마다 높이는 10m씩 감소한다. 또한 y절편이 60이므로 t=0일 때 물체의 높이는 60m이다. t=5일 때의 높이는 y = -10(5) + 60 = -50 + 60 = 10(m)이다.",
                variation_points: [
                    "context",
                    "rate_of_change",
                    "specific_value",
                ],
            },
        ],
        "FUN-008": [
            {
                item_id: "SEED-FUN-008-001",
                type: "conceptual",
                difficulty: "intermediate",
                content:
                    "일차함수 y = ax + b와 일차방정식 ax + by + c = 0의 관계를 설명하시오.",
                answer: "일차함수 y = ax + b는 y에 대해 정리된 형태이고, 일차방정식 ax + by + c = 0은 모든 항을 좌변으로 이항한 형태이다. 일차함수는 특정 x값에 대응하는 y값을 나타내는 관계이고, 일차방정식은 x와 y의 조건을 나타내는 관계이다.",
                explanation:
                    "일차함수 y = ax + b를 일차방정식 형태로 바꾸면, -ax + y - b = 0이 된다. 이처럼 일차방정식 ax + by + c = 0 형태에서 b ≠ 0인 경우, y에 대해 정리하면 y = -(a/b)x - (c/b) 형태의 일차함수가 된다. 일차함수는 함수관계를, 일차방정식은 좌표평면 위의 점들의 집합인 직선을 나타낸다.",
                variation_points: ["function_vs_equation", "representation"],
            },
            {
                item_id: "SEED-FUN-008-002",
                type: "application",
                difficulty: "advanced",
                content:
                    "연립방정식 { 2x + y = 7\n           { x - y = 1\n을 그래프를 이용하여 해결하는 방법을 설명하고, 해를 구하시오.",
                answer: "(x, y) = (2, 3)",
                explanation:
                    "첫 번째 방정식 2x + y = 7을 y에 대해 정리하면 y = -2x + 7이다. 두 번째 방정식 x - y = 1을 y에 대해 정리하면 y = x - 1이다. 이 두 일차함수의 그래프를 좌표평면 위에 그리면, 두 직선의 교점의 좌표가 연립방정식의 해가 된다. -2x + 7 = x - 1을 풀면, -3x = -8, x = 8/3이다. 이 값을 y = x - 1에 대입하면 y = 8/3 - 1 = 5/3이다. 따라서 해는 (8/3, 5/3)이다.",
                variation_points: ["system_of_equations", "graphical_solution"],
            },
            {
                item_id: "SEED-FUN-008-003",
                type: "analysis",
                difficulty: "advanced",
                content:
                    "두 일차함수 f(x) = ax + 2와 g(x) = -2x + b의 그래프가 x축 위의 한 점에서 만날 때, a + b의 값을 구하시오.",
                answer: "4",
                explanation:
                    "두 함수의 그래프가 x축 위의 점에서 만난다는 것은 y좌표가 0인 점에서 만난다는 의미이다. f(x) = 0을 만족하는 x값을 찾으면, ax + 2 = 0, x = -2/a이다. g(x) = 0을 만족하는 x값을 찾으면, -2x + b = 0, x = b/2이다. 두 함수의 그래프가 x축 위의 한 점에서 만나므로, -2/a = b/2이다. 이 식을 정리하면 b = -4/a이다. 따라서 a + b = a + (-4/a) = (a² - 4)/a이다. 두 함수의 그래프가 만나기 위해서는 a ≠ 0이어야 하고, 또한 두 그래프가 일치하지 않아야 하므로 a ≠ -2이어야 한다. 이 조건 하에서 a + b = 4이다.",
                variation_points: ["constraints", "algebraic_manipulation"],
            },
        ],
    },
    assessment_criteria: {
        accuracy: {
            description: "수학적 오류가 없어야 함",
            weight: 0.3,
            evaluation_method: "Expert review",
            passing_threshold: 0.95,
        },
        relevance: {
            description: "해당 개념 노드의 학습 목표에 부합해야 함",
            weight: 0.25,
            evaluation_method: "Concept mapping verification",
            passing_threshold: 0.9,
        },
        difficulty_appropriateness: {
            description: "설정된 난이도에 적합해야 함",
            weight: 0.15,
            evaluation_method: "Student testing",
            passing_threshold: 0.8,
        },
        clarity: {
            description: "문항이 명확하게 기술되어야 함",
            weight: 0.15,
            evaluation_method: "Readability analysis",
            passing_threshold: 0.85,
        },
        variation_quality: {
            description: "씨드 문항으로부터 의미 있는 변형이 이루어져야 함",
            weight: 0.15,
            evaluation_method: "Similarity analysis",
            passing_threshold: 0.7,
        },
    },
    generation_targets: {
        total_items: 72,
        difficulty_distribution: {
            basic: 24,
            intermediate: 30,
            advanced: 18,
        },
        item_type_distribution: {
            conceptual: 14,
            calculation: 22,
            graph_interpretation: 14,
            application: 16,
            problem_solving: 6,
        },
        node_distribution: {
            "FUN-001": 9,
            "FUN-002": 9,
            "FUN-003": 9,
            "FUN-004": 9,
            "FUN-005": 9,
            "FUN-006": 9,
            "FUN-007": 9,
            "FUN-008": 9,
        },
    },
};

// 디렉토리 생성
const dataDir = path.join(__dirname, "..", "prisma", "data");

if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
    console.log(`디렉토리 생성됨: ${dataDir}`);
}

// JSON 파일 저장
const filePath = path.join(dataDir, "knowledge-map-dataset.json");
fs.writeFileSync(
    filePath,
    JSON.stringify(knowledgeMapDataset, null, 2),
    "utf8"
);

console.log(`데이터셋 파일이 생성되었습니다: ${filePath}`);
console.log("이제 다음 명령을 실행하여 데이터베이스를 시드할 수 있습니다:");
console.log("npx ts-node prisma/seed.ts");
