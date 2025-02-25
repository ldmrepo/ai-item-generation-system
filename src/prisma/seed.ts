// prisma/seed.ts
import { PrismaClient } from "@prisma/client";
import * as fs from "fs";
import * as path from "path";

const prisma = new PrismaClient();

// JSON 파일 경로
const DATA_FILE_PATH = path.join(
    __dirname,
    "./data/knowledge-map-dataset.json"
);

async function main() {
    try {
        console.log("데이터 시드 시작...");

        // JSON 파일 읽기
        const rawData = fs.readFileSync(DATA_FILE_PATH, "utf8");
        const dataset = JSON.parse(rawData);

        // 1. 지식맵 생성
        console.log("지식맵 생성 중...");
        const knowledgeMapData = dataset.knowledge_map.map_info;

        const existingMap = await prisma.knowledgeMap.findFirst({
            where: {
                subject: knowledgeMapData.subject,
                grade: knowledgeMapData.grade,
                unit: knowledgeMapData.unit,
                version: knowledgeMapData.version,
            },
        });

        let knowledgeMap;
        if (existingMap) {
            console.log(`지식맵이 이미 존재합니다: ID ${existingMap.id}`);
            knowledgeMap = existingMap;
        } else {
            knowledgeMap = await prisma.knowledgeMap.create({
                data: {
                    subject: knowledgeMapData.subject,
                    grade: knowledgeMapData.grade,
                    unit: knowledgeMapData.unit,
                    version: knowledgeMapData.version,
                    creationDate: new Date(knowledgeMapData.creation_date),
                    description: knowledgeMapData.description,
                },
            });
            console.log(`지식맵이 생성되었습니다: ID ${knowledgeMap.id}`);
        }

        // 2. 개념 노드 생성
        console.log("개념 노드 생성 중...");
        const nodes = dataset.knowledge_map.nodes;

        for (const node of nodes) {
            const existingNode = await prisma.conceptNode.findUnique({
                where: { id: node.id },
            });

            if (existingNode) {
                console.log(`개념 노드가 이미 존재합니다: ${node.id}`);
                continue;
            }

            // 기본 노드 정보 생성
            const createdNode = await prisma.conceptNode.create({
                data: {
                    id: node.id,
                    mapId: knowledgeMap.id,
                    conceptName: node.concept_name,
                    definition: node.definition,
                    difficultyLevel: node.difficulty_level,
                    learningTime: node.learning_time,
                },
            });

            // 교육과정 매핑 생성
            await prisma.curriculumMapping.create({
                data: {
                    nodeId: createdNode.id,
                    educationSystem: node.curriculum_mapping.education_system,
                    grade: node.curriculum_mapping.grade,
                    unit: node.curriculum_mapping.unit,
                    achievementStandard:
                        node.curriculum_mapping.achievement_standard,
                },
            });

            // 오개념 정보 생성
            if (node.misconceptions && node.misconceptions.length > 0) {
                await prisma.misconception.createMany({
                    data: node.misconceptions.map((desc: string) => ({
                        nodeId: createdNode.id,
                        description: desc,
                    })),
                });
            }

            console.log(`개념 노드가 생성되었습니다: ${createdNode.id}`);
        }

        // 3. 노드 간 연결 생성
        console.log("노드 연결 생성 중...");
        const edges = dataset.knowledge_map.edges;

        for (const edge of edges) {
            const existingConnection = await prisma.nodeConnection.findFirst({
                where: {
                    sourceNodeId: edge.source,
                    targetNodeId: edge.target,
                },
            });

            if (existingConnection) {
                console.log(
                    `노드 연결이 이미 존재합니다: ${edge.source} -> ${edge.target}`
                );
                continue;
            }

            await prisma.nodeConnection.create({
                data: {
                    sourceNodeId: edge.source,
                    targetNodeId: edge.target,
                    connectionType: edge.type,
                    strength: edge.strength,
                },
            });

            console.log(
                `노드 연결이 생성되었습니다: ${edge.source} -> ${edge.target}`
            );
        }

        // 4. 씨드 문항 생성
        console.log("씨드 문항 생성 중...");
        const seedItems = dataset.seed_items;

        for (const [nodeId, items] of Object.entries(seedItems)) {
            for (const item of items as any[]) {
                const existingItem = await prisma.seedItem.findUnique({
                    where: { id: item.item_id },
                });

                if (existingItem) {
                    console.log(`씨드 문항이 이미 존재합니다: ${item.item_id}`);
                    continue;
                }

                // 씨드 문항 생성
                const createdItem = await prisma.seedItem.create({
                    data: {
                        id: item.item_id,
                        nodeId: nodeId,
                        itemType: item.type,
                        difficulty: item.difficulty,
                        content: item.content,
                        answer: item.answer,
                        explanation: item.explanation,
                    },
                });

                // 변형 요소 생성
                if (item.variation_points && item.variation_points.length > 0) {
                    await prisma.variationPoint.createMany({
                        data: item.variation_points.map((point: string) => ({
                            seedItemId: createdItem.id,
                            pointName: point,
                        })),
                    });
                }

                console.log(`씨드 문항이 생성되었습니다: ${createdItem.id}`);
            }
        }

        // 5. 평가 기준 생성
        console.log("평가 기준 생성 중...");
        const assessmentCriteria = dataset.assessment_criteria;

        for (const [criteriaName, criteriaData] of Object.entries(
            assessmentCriteria
        )) {
            const existingCriteria = await prisma.assessmentCriteria.findUnique(
                {
                    where: { name: criteriaName },
                }
            );

            if (existingCriteria) {
                console.log(`평가 기준이 이미 존재합니다: ${criteriaName}`);
                continue;
            }

            const data = criteriaData as any;

            await prisma.assessmentCriteria.create({
                data: {
                    name: criteriaName,
                    description: data.description,
                    weight: data.weight,
                    evaluationMethod: data.evaluation_method,
                    passingThreshold: data.passing_threshold,
                },
            });

            console.log(`평가 기준이 생성되었습니다: ${criteriaName}`);
        }

        console.log("데이터 시드가 완료되었습니다.");
    } catch (error) {
        console.error("데이터 시드 중 오류 발생:", error);
        throw error;
    } finally {
        await prisma.$disconnect();
    }
}

main()
    .then(async () => {
        await prisma.$disconnect();
    })
    .catch(async (e) => {
        console.error(e);
        await prisma.$disconnect();
        process.exit(1);
    });
