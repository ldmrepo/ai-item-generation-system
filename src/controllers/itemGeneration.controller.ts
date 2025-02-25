// src/controllers/itemGeneration.controller.ts
import { Request, Response, NextFunction } from "express";
import { prisma } from "../index";
import { ApiError } from "../utils/apiError";
import { v4 as uuidv4 } from "uuid";
import { Request, Response, NextFunction } from "express";
import { prisma } from "../index";
import { ApiError } from "../utils/apiError";
import { v4 as uuidv4 } from "uuid";

export const getGenerationStatus = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { requestId } = req.params;

        // 생성 이력 조회
        const generationHistory = await prisma.generationHistory.findFirst({
            where: { generationRequestId: requestId },
        });

        if (!generationHistory) {
            throw new ApiError(
                404,
                `Generation request with ID ${requestId} not found`
            );
        }

        // 생성된 문항 개수 조회
        const generatedItemsCount = await prisma.generationItemMapping.count({
            where: { generationHistoryId: generationHistory.id },
        });

        // 상태 정보 구성
        const responseData = {
            request_id: generationHistory.generationRequestId,
            status: generationHistory.status,
            items_requested: generationHistory.itemsRequested,
            items_generated: generatedItemsCount,
            start_time: generationHistory.startTime,
            end_time: generationHistory.endTime,
            creation_time: generationHistory.createdAt,
        };

        res.status(200).json({ data: responseData });
    } catch (error) {
        next(error);
    }
};

export const getGeneratedItems = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { requestId } = req.params;
        const {
            node_id,
            difficulty,
            item_type,
            approved,
            page = 1,
            limit = 20,
        } = req.query;

        // 생성 이력 조회
        const generationHistory = await prisma.generationHistory.findFirst({
            where: { generationRequestId: requestId },
        });

        if (!generationHistory) {
            throw new ApiError(
                404,
                `Generation request with ID ${requestId} not found`
            );
        }

        // 생성된 문항 ID 목록 조회
        const mappings = await prisma.generationItemMapping.findMany({
            where: { generationHistoryId: generationHistory.id },
            select: { generatedItemId: true },
        });

        const generatedItemIds = mappings.map(
            (mapping) => mapping.generatedItemId
        );

        // 필터 조건 구성
        const where: any = {
            id: { in: generatedItemIds },
        };

        if (node_id) {
            where.nodeId = node_id;
        }
        if (difficulty) {
            where.difficulty = difficulty;
        }
        if (item_type) {
            where.itemType = item_type;
        }
        if (approved !== undefined) {
            where.approved = approved === "true";
        }

        // 페이징 설정
        const skip = (Number(page) - 1) * Number(limit);
        const take = Number(limit);

        // 총 개수 조회
        const total = await prisma.generatedItem.count({ where });

        // 생성된 문항 목록 조회
        const generatedItems = await prisma.generatedItem.findMany({
            where,
            include: {
                conceptNode: {
                    select: {
                        conceptName: true,
                    },
                },
                seedItem: {
                    select: {
                        id: true,
                    },
                },
            },
            skip,
            take,
            orderBy: { generationTimestamp: "desc" },
        });

        // 응답 데이터 구성
        const formattedItems = generatedItems.map((item) => ({
            id: item.id,
            node_id: item.nodeId,
            node_name: item.conceptNode.conceptName,
            seed_item_id: item.seedItemId,
            item_type: item.itemType,
            difficulty: item.difficulty,
            content: item.content,
            answer: item.answer,
            explanation: item.explanation,
            approved: item.approved,
            quality_score: item.qualityScore,
            generation_timestamp: item.generationTimestamp,
        }));

        res.status(200).json({
            data: formattedItems,
            meta: {
                total,
                page: Number(page),
                limit: Number(limit),
                pages: Math.ceil(total / Number(limit)),
            },
        });
    } catch (error) {
        next(error);
    }
};

export const generateItems = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const {
            node_ids,
            count_per_node,
            difficulty_distribution,
            item_type_distribution,
            request_id = `GEN-${new Date()
                .toISOString()
                .slice(0, 10)
                .replace(/-/g, "")}-${uuidv4().slice(0, 8)}`,
        } = req.body;

        // 필수 필드 검증
        if (!node_ids || !count_per_node) {
            throw new ApiError(400, "Missing required fields");
        }

        if (!Array.isArray(node_ids) || node_ids.length === 0) {
            throw new ApiError(400, "node_ids must be a non-empty array");
        }

        if (isNaN(count_per_node) || count_per_node <= 0) {
            throw new ApiError(400, "count_per_node must be a positive number");
        }

        // 노드 존재 여부 확인
        const existingNodes = await prisma.conceptNode.findMany({
            where: {
                id: {
                    in: node_ids,
                },
            },
        });

        if (existingNodes.length !== node_ids.length) {
            const foundNodeIds = existingNodes.map((node) => node.id);
            const missingNodeIds = node_ids.filter(
                (id) => !foundNodeIds.includes(id)
            );
            throw new ApiError(
                400,
                `Following nodes not found: ${missingNodeIds.join(", ")}`
            );
        }

        // 생성 이력 기록
        const generationHistory = await prisma.generationHistory.create({
            data: {
                generationRequestId: request_id,
                userId: req.body.user_id, // 요청 사용자 ID (있을 경우)
                requestParams: req.body, // 모든 요청 파라미터 저장
                itemsRequested: node_ids.length * count_per_node,
                itemsGenerated: 0,
                startTime: new Date(),
                status: "processing",
            },
        });

        // 비동기로 문항 생성 프로세스 시작
        // 실제 구현에서는 이 부분을 별도의 작업 큐나 백그라운드 프로세스로 처리
        setTimeout(async () => {
            try {
                await generateItemsProcess(
                    generationHistory.id,
                    node_ids,
                    count_per_node,
                    difficulty_distribution,
                    item_type_distribution
                );
            } catch (error) {
                console.error("Error in item generation process:", error);
                // 에러 발생 시 상태 업데이트
                await prisma.generationHistory.update({
                    where: { id: generationHistory.id },
                    data: {
                        status: "failed",
                        endTime: new Date(),
                    },
                });
            }
        }, 0);

        // 즉시 응답 반환
        res.status(202).json({
            data: {
                request_id,
                status: "processing",
                nodes_requested: node_ids,
                total_items_requested: node_ids.length * count_per_node,
                start_time: generationHistory.startTime,
                estimated_completion_time: new Date(Date.now() + 60000), // 예상 완료 시간 (1분 후)
            },
        });
    } catch (error) {
        next(error);
    }
};

// 문항 생성 처리 함수 (백그라운드에서 실행)
async function generateItemsProcess(
    historyId: number,
    nodeIds: string[],
    countPerNode: number,
    difficultyDistribution?: any,
    itemTypeDistribution?: any
) {
    try {
        let generatedCount = 0;

        // 각 노드별로 문항 생성
        for (const nodeId of nodeIds) {
            // 노드의 씨드 문항 조회
            const seedItems = await prisma.seedItem.findMany({
                where: { nodeId },
                include: {
                    variationPoints: true,
                },
            });

            if (seedItems.length === 0) {
                console.warn(
                    `No seed items found for node ${nodeId}, skipping...`
                );
                continue;
            }

            // 노드별 요청 개수만큼 문항 생성
            for (let i = 0; i < countPerNode; i++) {
                // 씨드 문항 랜덤 선택
                const seedItem =
                    seedItems[Math.floor(Math.random() * seedItems.length)];

                // 난이도 분포에 따라 난이도 결정
                let difficulty = seedItem.difficulty;
                if (difficultyDistribution) {
                    const rand = Math.random();
                    let cumulative = 0;
                    for (const [diff, prob] of Object.entries(
                        difficultyDistribution
                    )) {
                        cumulative += Number(prob);
                        if (rand <= cumulative) {
                            difficulty = diff;
                            break;
                        }
                    }
                }

                // 문항 유형 분포에 따라 유형 결정
                let itemType = seedItem.itemType;
                if (itemTypeDistribution) {
                    const rand = Math.random();
                    let cumulative = 0;
                    for (const [type, prob] of Object.entries(
                        itemTypeDistribution
                    )) {
                        cumulative += Number(prob);
                        if (rand <= cumulative) {
                            itemType = type;
                            break;
                        }
                    }
                }

                // 생성 문항 ID 생성
                const generatedItemId = `GEN-${nodeId}-${Date.now()
                    .toString()
                    .slice(-6)}-${i + 1}`;

                // 문항 변형 생성 (실제 구현에서는 AI 모델 호출)
                // 여기서는 간단한 변형만 수행
                const variationPoints = seedItem.variationPoints.map(
                    (vp) => vp.pointName
                );
                let content = seedItem.content;
                let answer = seedItem.answer;
                let explanation = seedItem.explanation;

                // 간단한 변형 로직 (실제 구현에서는 더 복잡한 AI 기반 변형)
                if (
                    variationPoints.includes("coefficients") ||
                    variationPoints.includes("numbers")
                ) {
                    // 숫자 값 변경
                    const randomFactor = Math.floor(Math.random() * 5) + 1;
                    content = content.replace(/\d+/g, (match) => {
                        const num = parseInt(match, 10);
                        return (num * randomFactor).toString();
                    });

                    // 답변도 적절히 변경
                    answer = answer.replace(/\d+/g, (match) => {
                        const num = parseInt(match, 10);
                        return (num * randomFactor).toString();
                    });
                }

                if (
                    variationPoints.includes("context") ||
                    variationPoints.includes("examples")
                ) {
                    // 문맥 변형 (간단한 예시)
                    const contexts = [
                        "학교",
                        "가게",
                        "공원",
                        "도서관",
                        "운동장",
                    ];
                    const randomContext =
                        contexts[Math.floor(Math.random() * contexts.length)];
                    content = content.replace(
                        /물탱크|택시|회사/g,
                        randomContext
                    );
                }

                // 생성된 문항 저장
                const generatedItem = await prisma.generatedItem.create({
                    data: {
                        id: generatedItemId,
                        nodeId,
                        seedItemId: seedItem.id,
                        itemType,
                        difficulty,
                        content,
                        answer,
                        explanation:
                            explanation +
                            "\n\n(원본 문항 기반 자동 생성되었습니다)",
                        qualityScore: 0.7, // 기본 품질 점수
                    },
                });

                // 생성 매핑 저장
                await prisma.generationItemMapping.create({
                    data: {
                        generationHistoryId: historyId,
                        generatedItemId: generatedItem.id,
                    },
                });

                generatedCount++;
            }
        }

        // 생성 이력 업데이트
        await prisma.generationHistory.update({
            where: { id: historyId },
            data: {
                itemsGenerated: generatedCount,
                status: "completed",
                endTime: new Date(),
            },
        });

        console.log(
            `Item generation completed: ${generatedCount} items generated`
        );
    } catch (error) {
        console.error("Error in item generation process:", error);
        throw error;
    }
}
