// 지식맵 API 사용 예제 (Node.js - Axios)

// 1. 의존성 설치
// npm install axios

const axios = require("axios");

const API_BASE_URL = "http://localhost:3000/api/v1";
const AUTH_TOKEN = "your_auth_token"; // 실제 구현시 인증 토큰 사용

// API 클라이언트 설정
const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${AUTH_TOKEN}`,
    },
});

// 2. 지식맵 생성
async function createKnowledgeMap() {
    try {
        const response = await apiClient.post("/knowledge-maps", {
            subject: "수학",
            grade: "중학교 2학년",
            unit: "함수",
            version: "1.0",
            description:
                "중학교 2학년 함수 영역의 핵심 개념과 관계를 구조화한 지식맵",
        });

        console.log("지식맵 생성 성공:", response.data);
        return response.data.data.id; // 생성된 지식맵 ID 반환
    } catch (error) {
        console.error(
            "지식맵 생성 실패:",
            error.response?.data || error.message
        );
        throw error;
    }
}

// 3. 개념 노드 생성
async function createConceptNode(mapId, nodeData) {
    try {
        const response = await apiClient.post(
            `/knowledge-maps/${mapId}/nodes`,
            nodeData
        );
        console.log("개념 노드 생성 성공:", response.data);
        return response.data.data.id; // 생성된 노드 ID 반환
    } catch (error) {
        console.error(
            "개념 노드 생성 실패:",
            error.response?.data || error.message
        );
        throw error;
    }
}

// 4. 노드 연결 생성
async function createNodeConnection(
    sourceNodeId,
    targetNodeId,
    connectionType,
    strength
) {
    try {
        const response = await apiClient.post(
            `/nodes/${sourceNodeId}/connections`,
            {
                target_node_id: targetNodeId,
                connection_type: connectionType,
                strength: strength,
            }
        );
        console.log("노드 연결 생성 성공:", response.data);
        return response.data.data.id; // 생성된 연결 ID 반환
    } catch (error) {
        console.error(
            "노드 연결 생성 실패:",
            error.response?.data || error.message
        );
        throw error;
    }
}

// 5. 씨드 문항 생성
async function createSeedItem(nodeId, itemData) {
    try {
        const response = await apiClient.post(
            `/nodes/${nodeId}/seed-items`,
            itemData
        );
        console.log("씨드 문항 생성 성공:", response.data);
        return response.data.data.id; // 생성된 씨드 문항 ID 반환
    } catch (error) {
        console.error(
            "씨드 문항 생성 실패:",
            error.response?.data || error.message
        );
        throw error;
    }
}

// 6. 문항 생성 요청
async function generateItems(nodeIds, countPerNode) {
    try {
        const response = await apiClient.post("/item-generation/generate", {
            node_ids: nodeIds,
            count_per_node: countPerNode,
            difficulty_distribution: {
                basic: 0.3,
                intermediate: 0.5,
                advanced: 0.2,
            },
            item_type_distribution: {
                conceptual: 0.2,
                calculation: 0.3,
                graph_interpretation: 0.3,
                application: 0.2,
            },
        });

        console.log("문항 생성 요청 성공:", response.data);
        return response.data.data.request_id; // 생성 요청 ID 반환
    } catch (error) {
        console.error(
            "문항 생성 요청 실패:",
            error.response?.data || error.message
        );
        throw error;
    }
}

// 7. 생성 상태 확인
async function checkGenerationStatus(requestId) {
    try {
        const response = await apiClient.get(
            `/item-generation/${requestId}/status`
        );
        console.log("생성 상태:", response.data);
        return response.data.data.status;
    } catch (error) {
        console.error(
            "생성 상태 확인 실패:",
            error.response?.data || error.message
        );
        throw error;
    }
}

// 8. 생성된 문항 목록 조회
async function getGeneratedItems(requestId) {
    try {
        const response = await apiClient.get(
            `/item-generation/${requestId}/items`
        );
        console.log(`생성된 문항 수: ${response.data.data.length}`);
        return response.data.data;
    } catch (error) {
        console.error(
            "생성된 문항 조회 실패:",
            error.response?.data || error.message
        );
        throw error;
    }
}

// 9. 문항 승인/반려
async function approveItem(itemId, approved, feedback) {
    try {
        const response = await apiClient.patch(
            `/generated-items/${itemId}/approval`,
            {
                approved,
                feedback,
                assessed_by: "reviewer@example.com",
            }
        );
        console.log("문항 승인/반려 성공:", response.data);
        return response.data;
    } catch (error) {
        console.error(
            "문항 승인/반려 실패:",
            error.response?.data || error.message
        );
        throw error;
    }
}

// 10. 문항 평가
async function assessItem(itemId, criteriaAssessments) {
    try {
        const response = await apiClient.post(
            `/generated-items/${itemId}/assessments`,
            {
                criteria: criteriaAssessments,
                assessed_by: "evaluator@example.com",
            }
        );
        console.log("문항 평가 성공:", response.data);
        return response.data;
    } catch (error) {
        console.error("문항 평가 실패:", error.response?.data || error.message);
        throw error;
    }
}

// 전체 프로세스 실행 예제
async function runFullProcess() {
    try {
        // 1. 지식맵 생성
        const mapId = await createKnowledgeMap();

        // 2. 개념 노드 생성
        const node1Data = {
            id: "FUN-001",
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
            ],
        };

        const node2Data = {
            id: "FUN-003",
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
            ],
        };

        const node1Id = await createConceptNode(mapId, node1Data);
        const node2Id = await createConceptNode(mapId, node2Data);

        // 3. 노드 연결 생성
        await createNodeConnection(node1Id, node2Id, "prerequisite", "strong");

        // 4. 씨드 문항 생성
        const seedItem1Data = {
            id: "SEED-FUN-001-001",
            item_type: "conceptual",
            difficulty: "basic",
            content:
                "함수의 정의를 설명하고, 함수와 함수가 아닌 관계의 예를 각각 1개씩 제시하시오.",
            answer: "함수는 첫 번째 집합의 모든 원소가 두 번째 집합의 원소에 한 개씩만 대응되는 관계이다. 함수의 예: x와 2x의 관계, 함수가 아닌 예: x와 ±√x의 관계",
            explanation:
                "함수는 정의역의 모든 원소가 공역의 원소에 오직 하나씩만 대응되어야 한다. x와 2x의 관계는 각 x값에 대해 2x라는 하나의 값만 대응되므로 함수이다. 반면 x와 ±√x 관계에서는 x=4일 때 y=2 또는 y=-2가 대응되므로 함수가 아니다.",
            variation_points: ["definition", "examples"],
        };

        const seedItem2Data = {
            id: "SEED-FUN-003-001",
            item_type: "conceptual",
            difficulty: "basic",
            content:
                "일차함수의 정의를 쓰고, y = 3x - 2가 일차함수인 이유를 설명하시오.",
            answer: "일차함수는 y = ax + b (a≠0) 형태로 나타낼 수 있는 함수이다. y = 3x - 2는 a=3, b=-2인 일차함수이다.",
            explanation:
                "일차함수의 가장 큰 특징은 x의 계수 a가 0이 아닌 1차식으로 표현된다는 것이다. y = 3x - 2에서 x의 계수는 3으로 0이 아니고, x의 최고차항이 1차이므로 일차함수이다.",
            variation_points: ["definition", "example_analysis"],
        };

        await createSeedItem(node1Id, seedItem1Data);
        await createSeedItem(node2Id, seedItem2Data);

        // 5. 문항 생성 요청
        const requestId = await generateItems([node1Id, node2Id], 3);

        // 6. 생성 상태 확인 (실제 구현에서는 폴링 또는 웹훅 사용)
        console.log("문항 생성 중...");

        let status = await checkGenerationStatus(requestId);
        // 실제 구현에서는 비동기 처리
        while (status !== "completed") {
            await new Promise((resolve) => setTimeout(resolve, 1000));
            status = await checkGenerationStatus(requestId);
            console.log(`현재 상태: ${status}`);
        }

        // 7. 생성된 문항 조회
        const generatedItems = await getGeneratedItems(requestId);
        console.log(`생성된 문항 목록: ${generatedItems.length}개`);

        // 8. 문항 승인/평가 (첫 번째 문항)
        if (generatedItems.length > 0) {
            const firstItem = generatedItems[0];

            // 문항 승인
            await approveItem(
                firstItem.id,
                true,
                "좋은 문항입니다. 학생들의 개념 이해를 평가하기에 적합합니다."
            );

            // 문항 평가
            await assessItem(firstItem.id, [
                {
                    criteria_id: 1,
                    score: 0.9,
                    feedback: "수학적 정확성이 높음",
                },
                { criteria_id: 2, score: 0.85, feedback: "학습 목표에 부합함" },
                { criteria_id: 3, score: 0.8, feedback: "난이도 적절함" },
            ]);

            console.log("문항 승인 및 평가 완료");
        }

        console.log("전체 프로세스 완료");
    } catch (error) {
        console.error("프로세스 실행 중 오류 발생:", error);
    }
}

// 프로세스 실행
runFullProcess();
