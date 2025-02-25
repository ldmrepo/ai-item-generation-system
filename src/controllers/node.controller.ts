// src/controllers/node.controller.ts
import { Request, Response, NextFunction } from "express";
import { prisma } from "../index";
import { ApiError } from "../utils/apiError";

export const getNodeById = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { nodeId } = req.params;

        // 노드 기본 정보 조회
        const node = await prisma.conceptNode.findUnique({
            where: { id: nodeId },
        });

        if (!node) {
            throw new ApiError(404, `Node with ID ${nodeId} not found`);
        }

        // 교육과정 매핑 정보 조회
        const curriculumMapping = await prisma.curriculumMapping.findFirst({
            where: { nodeId },
        });

        // 연결 정보 조회
        const prerequisites = await prisma.nodeConnection.findMany({
            where: { targetNodeId: nodeId, connectionType: "prerequisite" },
            include: {
                sourceNode: {
                    select: {
                        id: true,
                        conceptName: true,
                    },
                },
            },
        });

        const successors = await prisma.nodeConnection.findMany({
            where: { sourceNodeId: nodeId, connectionType: "prerequisite" },
            include: {
                targetNode: {
                    select: {
                        id: true,
                        conceptName: true,
                    },
                },
            },
        });

        const related = await prisma.nodeConnection.findMany({
            where: {
                OR: [
                    { sourceNodeId: nodeId, connectionType: "related" },
                    { targetNodeId: nodeId, connectionType: "related" },
                ],
            },
            include: {
                sourceNode: {
                    select: {
                        id: true,
                        conceptName: true,
                    },
                },
                targetNode: {
                    select: {
                        id: true,
                        conceptName: true,
                    },
                },
            },
        });

        // 오개념 정보 조회
        const misconceptions = await prisma.misconception.findMany({
            where: { nodeId },
            select: { description: true },
        });

        // 응답 데이터 구성
        const responseData = {
            ...node,
            curriculum_mapping: curriculumMapping,
            connections: {
                prerequisites: prerequisites.map((conn) => ({
                    id: conn.sourceNode.id,
                    concept_name: conn.sourceNode.conceptName,
                    strength: conn.strength,
                })),
                successors: successors.map((conn) => ({
                    id: conn.targetNode.id,
                    concept_name: conn.targetNode.conceptName,
                    strength: conn.strength,
                })),
                related: related.map((conn) => {
                    // 현재 노드가 source인 경우 target 정보를, 현재 노드가 target인 경우 source 정보를 반환
                    const relatedNode =
                        conn.sourceNodeId === nodeId
                            ? conn.targetNode
                            : conn.sourceNode;
                    return {
                        id: relatedNode.id,
                        concept_name: relatedNode.conceptName,
                        strength: conn.strength,
                    };
                }),
            },
            misconceptions: misconceptions.map((m) => m.description),
        };

        res.status(200).json({ data: responseData });
    } catch (error) {
        next(error);
    }
};

export const getNodesByMapId = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { mapId } = req.params;
        const { difficulty_level, page = 1, limit = 20 } = req.query;

        const skip = (Number(page) - 1) * Number(limit);
        const take = Number(limit);

        // 필터 조건 구성
        const where: any = { mapId: Number(mapId) };
        if (difficulty_level) {
            where.difficultyLevel = difficulty_level;
        }

        // 총 개수 조회
        const total = await prisma.conceptNode.count({ where });

        // 노드 목록 조회
        const nodes = await prisma.conceptNode.findMany({
            where,
            skip,
            take,
            orderBy: { id: "asc" },
        });

        res.status(200).json({
            data: nodes,
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

export const createNode = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { mapId } = req.params;
        const {
            id,
            concept_name,
            definition,
            difficulty_level,
            learning_time,
            curriculum_mapping,
            misconceptions,
        } = req.body;

        // 필수 필드 검증
        if (
            !id ||
            !concept_name ||
            !definition ||
            !difficulty_level ||
            !learning_time ||
            !curriculum_mapping
        ) {
            throw new ApiError(400, "Missing required fields");
        }

        // 노드 생성
        const node = await prisma.conceptNode.create({
            data: {
                id,
                mapId: Number(mapId),
                conceptName: concept_name,
                definition,
                difficultyLevel: difficulty_level,
                learningTime: learning_time,
            },
        });

        // 교육과정 매핑 생성
        await prisma.curriculumMapping.create({
            data: {
                nodeId: node.id,
                educationSystem: curriculum_mapping.education_system,
                grade: curriculum_mapping.grade,
                unit: curriculum_mapping.unit,
                achievementStandard: curriculum_mapping.achievement_standard,
            },
        });

        // 오개념 정보 생성 (있을 경우)
        if (misconceptions && misconceptions.length > 0) {
            await prisma.misconception.createMany({
                data: misconceptions.map((desc: string) => ({
                    nodeId: node.id,
                    description: desc,
                })),
            });
        }

        res.status(201).json({
            data: node,
            message: "Node created successfully",
        });
    } catch (error) {
        next(error);
    }
};

export const deleteNode = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { nodeId } = req.params;

        // 노드 존재 여부 확인
        const nodeExists = await prisma.conceptNode.findUnique({
            where: { id: nodeId },
        });

        if (!nodeExists) {
            throw new ApiError(404, `Node with ID ${nodeId} not found`);
        }

        // 트랜잭션으로 관련 데이터 삭제
        await prisma.$transaction(async (tx) => {
            // 오개념 삭제
            await tx.misconception.deleteMany({
                where: { nodeId },
            });

            // 교육과정 매핑 삭제
            await tx.curriculumMapping.deleteMany({
                where: { nodeId },
            });

            // 연결 정보 삭제
            await tx.nodeConnection.deleteMany({
                where: {
                    OR: [{ sourceNodeId: nodeId }, { targetNodeId: nodeId }],
                },
            });

            // 씨드 문항 관련 변형 요소 삭제
            const seedItems = await tx.seedItem.findMany({
                where: { nodeId },
                select: { id: true },
            });

            const seedItemIds = seedItems.map((item) => item.id);

            await tx.variationPoint.deleteMany({
                where: {
                    seedItemId: {
                        in: seedItemIds,
                    },
                },
            });

            // 씨드 문항 삭제
            await tx.seedItem.deleteMany({
                where: { nodeId },
            });

            // 생성된 문항 평가 삭제
            const generatedItems = await tx.generatedItem.findMany({
                where: { nodeId },
                select: { id: true },
            });

            const generatedItemIds = generatedItems.map((item) => item.id);

            await tx.itemAssessment.deleteMany({
                where: {
                    generatedItemId: {
                        in: generatedItemIds,
                    },
                },
            });

            // 생성 매핑 삭제
            await tx.generationItemMapping.deleteMany({
                where: {
                    generatedItemId: {
                        in: generatedItemIds,
                    },
                },
            });

            // 생성된 문항 삭제
            await tx.generatedItem.deleteMany({
                where: { nodeId },
            });

            // 노드 삭제
            await tx.conceptNode.delete({
                where: { id: nodeId },
            });
        });

        res.status(200).json({
            message: `Node with ID ${nodeId} and all related data deleted successfully`,
        });
    } catch (error) {
        next(error);
    }
};

export const createNodeConnection = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { sourceNodeId } = req.params;
        const { target_node_id, connection_type, strength } = req.body;

        // 필수 필드 검증
        if (!target_node_id || !connection_type || !strength) {
            throw new ApiError(400, "Missing required fields");
        }

        // 소스 및 타겟 노드 존재 확인
        const sourceNode = await prisma.conceptNode.findUnique({
            where: { id: sourceNodeId },
        });

        const targetNode = await prisma.conceptNode.findUnique({
            where: { id: target_node_id },
        });

        if (!sourceNode) {
            throw new ApiError(
                404,
                `Source node with ID ${sourceNodeId} not found`
            );
        }

        if (!targetNode) {
            throw new ApiError(
                404,
                `Target node with ID ${target_node_id} not found`
            );
        }

        // 이미 존재하는 연결인지 확인
        const existingConnection = await prisma.nodeConnection.findFirst({
            where: {
                sourceNodeId,
                targetNodeId: target_node_id,
            },
        });

        if (existingConnection) {
            throw new ApiError(
                400,
                "Connection already exists between these nodes"
            );
        }

        // 연결 생성
        const connection = await prisma.nodeConnection.create({
            data: {
                sourceNodeId,
                targetNodeId: target_node_id,
                connectionType: connection_type,
                strength,
            },
        });

        res.status(201).json({
            data: connection,
            message: "Node connection created successfully",
        });
    } catch (error) {
        next(error);
    }
};

export const getNodeConnections = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { nodeId } = req.params;
        const { direction = "all", type } = req.query;

        // 노드 존재 확인
        const nodeExists = await prisma.conceptNode.findUnique({
            where: { id: nodeId },
        });

        if (!nodeExists) {
            throw new ApiError(404, `Node with ID ${nodeId} not found`);
        }

        // 방향에 따라 조회 조건 설정
        let whereCondition: any = {};

        if (direction === "incoming") {
            whereCondition.targetNodeId = nodeId;
        } else if (direction === "outgoing") {
            whereCondition.sourceNodeId = nodeId;
        } else {
            whereCondition = {
                OR: [{ sourceNodeId: nodeId }, { targetNodeId: nodeId }],
            };
        }

        // 연결 유형 필터 적용
        if (type) {
            whereCondition.connectionType = type;
        }

        // 연결 정보 조회
        const connections = await prisma.nodeConnection.findMany({
            where: whereCondition,
            include: {
                sourceNode: {
                    select: {
                        id: true,
                        conceptName: true,
                    },
                },
                targetNode: {
                    select: {
                        id: true,
                        conceptName: true,
                    },
                },
            },
        });

        // 응답 구성
        const formattedConnections = connections.map((conn) => ({
            id: conn.id,
            source_node: {
                id: conn.sourceNode.id,
                concept_name: conn.sourceNode.conceptName,
            },
            target_node: {
                id: conn.targetNode.id,
                concept_name: conn.targetNode.conceptName,
            },
            connection_type: conn.connectionType,
            strength: conn.strength,
            created_at: conn.createdAt,
        }));

        res.status(200).json({ data: formattedConnections });
    } catch (error) {
        next(error);
    }
};

export const deleteNodeConnection = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { sourceNodeId, targetNodeId } = req.params;

        // 연결 존재 확인
        const connection = await prisma.nodeConnection.findFirst({
            where: {
                sourceNodeId,
                targetNodeId,
            },
        });

        if (!connection) {
            throw new ApiError(
                404,
                `Connection between nodes ${sourceNodeId} and ${targetNodeId} not found`
            );
        }

        // 연결 삭제
        await prisma.nodeConnection.delete({
            where: {
                id: connection.id,
            },
        });

        res.status(200).json({
            message: `Connection between nodes ${sourceNodeId} and ${targetNodeId} deleted successfully`,
        });
    } catch (error) {
        next(error);
    }
};
