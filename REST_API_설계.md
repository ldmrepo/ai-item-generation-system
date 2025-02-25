# 문항 생성 시스템 REST API 설계

## 기본 정보

-   기본 URL: `/api/v1`
-   인증: Bearer 토큰 기반 인증
-   응답 형식: JSON
-   상태 코드
    -   200: 성공
    -   201: 리소스 생성 성공
    -   400: 잘못된 요청
    -   401: 인증 실패
    -   403: 권한 없음
    -   404: 리소스 없음
    -   500: 서버 오류

## 엔드포인트

### 지식맵 관리

#### 지식맵 조회

```
GET /knowledge-maps
```

-   쿼리 파라미터:
    -   subject: 과목명 (선택)
    -   grade: 학년 (선택)
    -   unit: 단원명 (선택)
    -   page: 페이지 번호 (기본값: 1)
    -   limit: 페이지당 항목 수 (기본값: 10)

#### 지식맵 상세 조회

```
GET /knowledge-maps/{id}
```

#### 지식맵 생성

```
POST /knowledge-maps
```

-   요청 본문:
    -   subject: 과목명 (필수)
    -   grade: 학년 (필수)
    -   unit: 단원명 (필수)
    -   version: 버전 (필수)
    -   description: 설명 (선택)

#### 지식맵 수정

```
PUT /knowledge-maps/{id}
```

-   요청 본문: 지식맵 생성과 동일

#### 지식맵 삭제

```
DELETE /knowledge-maps/{id}
```

### 개념 노드 관리

#### 개념 노드 목록 조회

```
GET /knowledge-maps/{mapId}/nodes
```

-   쿼리 파라미터:
    -   difficulty_level: 난이도 (선택)
    -   page: 페이지 번호 (기본값: 1)
    -   limit: 페이지당 항목 수 (기본값: 20)

#### 개념 노드 상세 조회

```
GET /nodes/{nodeId}
```

#### 개념 노드 생성

```
POST /knowledge-maps/{mapId}/nodes
```

-   요청 본문:
    -   id: 노드 ID (예: FUN-001) (필수)
    -   concept_name: 개념명 (필수)
    -   definition: 정의 (필수)
    -   difficulty_level: 난이도 (필수)
    -   learning_time: 학습시간(분) (필수)
    -   curriculum_mapping: 교육과정 매핑 정보 (필수)
    -   misconceptions: 오개념 목록 (선택)

#### 개념 노드 수정

```
PUT /nodes/{nodeId}
```

-   요청 본문: 개념 노드 생성과 동일

#### 개념 노드 삭제

```
DELETE /nodes/{nodeId}
```

#### 노드 연결 관리

```
POST /nodes/{sourceNodeId}/connections
```

-   요청 본문:
    -   target_node_id: 대상 노드 ID (필수)
    -   connection_type: 연결 유형 (필수)
    -   strength: 연결 강도 (필수)

```
GET /nodes/{nodeId}/connections
```

-   쿼리 파라미터:
    -   direction: 방향 (incoming, outgoing, all) (기본값: all)
    -   type: 연결 유형 (선택)

```
DELETE /nodes/{sourceNodeId}/connections/{targetNodeId}
```

### 씨드 문항 관리

#### 씨드 문항 목록 조회

```
GET /nodes/{nodeId}/seed-items
```

-   쿼리 파라미터:
    -   item_type: 문항 유형 (선택)
    -   difficulty: 난이도 (선택)
    -   page: 페이지 번호 (기본값: 1)
    -   limit: 페이지당 항목 수 (기본값: 10)

#### 씨드 문항 상세 조회

```
GET /seed-items/{itemId}
```

#### 씨드 문항 생성

```
POST /nodes/{nodeId}/seed-items
```

-   요청 본문:
    -   id: 문항 ID (예: SEED-FUN-001-001) (필수)
    -   item_type: 문항 유형 (필수)
    -   difficulty: 난이도 (필수)
    -   content: 문항 내용 (필수)
    -   answer: 정답 (필수)
    -   explanation: 설명 (필수)
    -   variation_points: 변형 요소 배열 (선택)

#### 씨드 문항 수정

```
PUT /seed-items/{itemId}
```

-   요청 본문: 씨드 문항 생성과 동일

#### 씨드 문항 삭제

```
DELETE /seed-items/{itemId}
```

### 문항 생성 API

#### 문항 생성 요청

```
POST /item-generation/generate
```

-   요청 본문:
    -   node_ids: 개념 노드 ID 배열 (필수)
    -   count_per_node: 노드별 생성 문항 수 (필수)
    -   difficulty_distribution: 난이도 분포 객체 (선택)
    -   item_type_distribution: 문항 유형 분포 객체 (선택)
    -   request_id: 요청 ID (선택, 자동 생성됨)

#### 문항 생성 상태 조회

```
GET /item-generation/{requestId}/status
```

#### 생성된 문항 목록 조회

```
GET /item-generation/{requestId}/items
```

-   쿼리 파라미터:
    -   node_id: 개념 노드 ID (선택)
    -   difficulty: 난이도 (선택)
    -   item_type: 문항 유형 (선택)
    -   approved: 승인 여부 (선택)
    -   page: 페이지 번호 (기본값: 1)
    -   limit: 페이지당 항목 수 (기본값: 20)

#### 생성된 문항 상세 조회

```
GET /generated-items/{itemId}
```

#### 생성된 문항 승인/반려

```
PATCH /generated-items/{itemId}/approval
```

-   요청 본문:
    -   approved: 승인 여부 (필수)
    -   feedback: 피드백 (선택)

### 문항 평가 API

#### 문항 평가 요청

```
POST /generated-items/{itemId}/assessments
```

-   요청 본문:
    -   criteria: 평가 기준 배열
        -   criteria_id: 평가 기준 ID
        -   score: 점수
        -   feedback: 피드백 (선택)
    -   assessed_by: 평가자 ID (선택)

#### 문항 평가 결과 조회

```
GET /generated-items/{itemId}/assessments
```

### 시스템 정보 API

#### 평가 기준 목록 조회

```
GET /assessment-criteria
```

#### 문항 유형 목록 조회

```
GET /item-types
```

#### 난이도 수준 목록 조회

```
GET /difficulty-levels
```

## 예제 요청/응답

### 지식맵 조회 예제

요청:

```
GET /api/v1/knowledge-maps?subject=수학&grade=중학교%202학년
```

응답:

```json
{
    "data": [
        {
            "id": 1,
            "subject": "수학",
            "grade": "중학교 2학년",
            "unit": "함수",
            "version": "1.0",
            "creation_date": "2025-02-25T00:00:00.000Z",
            "description": "중학교 2학년 함수 영역의 핵심 개념과 관계를 구조화한 지식맵"
        }
    ],
    "meta": {
        "total": 1,
        "page": 1,
        "limit": 10
    }
}
```

### 개념 노드 조회 예제

요청:

```
GET /api/v1/nodes/FUN-005
```

응답:

```json
{
    "data": {
        "id": "FUN-005",
        "concept_name": "일차함수의 기울기",
        "definition": "일차함수 y = ax + b에서 x가 1단위 증가할 때 y의 증가량을 나타내는 값 a",
        "difficulty_level": "중급",
        "learning_time": 60,
        "curriculum_mapping": {
            "education_system": "국가교육과정",
            "grade": "중학교 2학년",
            "unit": "함수",
            "achievement_standard": "일차함수의 기울기를 이해하고, 이를 활용하여 문제를 해결할 수 있다."
        },
        "connections": {
            "prerequisites": [
                {
                    "id": "FUN-003",
                    "concept_name": "일차함수의 개념",
                    "strength": "strong"
                },
                {
                    "id": "FUN-004",
                    "concept_name": "일차함수의 그래프",
                    "strength": "strong"
                }
            ],
            "successors": [
                {
                    "id": "FUN-006",
                    "concept_name": "일차함수의 응용",
                    "strength": "strong"
                },
                {
                    "id": "FUN-007",
                    "concept_name": "그래프의 해석",
                    "strength": "medium"
                }
            ],
            "related": [
                {
                    "id": "FUN-008",
                    "concept_name": "일차방정식과의 관계",
                    "strength": "weak"
                }
            ]
        },
        "misconceptions": [
            "기울기와 y절편 혼동",
            "기울기의 부호와 함수의 증감 관계 오해",
            "두 점 사이의 기울기 계산 오류"
        ]
    }
}
```

### 문항 생성 요청 예제

요청:

```
POST /api/v1/item-generation/generate
Content-Type: application/json

{
  "node_ids": ["FUN-003", "FUN-004", "FUN-005"],
  "count_per_node": 3,
  "difficulty_distribution": {
    "basic": 0.3,
    "intermediate": 0.5,
    "advanced": 0.2
  },
  "item_type_distribution": {
    "conceptual": 0.2,
    "calculation": 0.3,
    "graph_interpretation": 0.3,
    "application": 0.2
  }
}
```

응답:

```json
{
    "data": {
        "request_id": "GEN-20250225-001",
        "status": "processing",
        "nodes_requested": ["FUN-003", "FUN-004", "FUN-005"],
        "total_items_requested": 9,
        "start_time": "2025-02-25T10:15:30.000Z",
        "estimated_completion_time": "2025-02-25T10:18:30.000Z"
    }
}
```
