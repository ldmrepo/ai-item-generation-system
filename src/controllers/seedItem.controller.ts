// src/controllers/seedItem.controller.ts
import { Request, Response, NextFunction } from "express";
import { prisma } from "../index";
import { ApiError } from "../utils/apiError";

export const getSeedItemById = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { itemId } = req.params;

        // 씨드 문항 조회
        const seedItem = await prisma.seedItem.findUnique({
            where: { id: itemId },
            include: {
                conceptNode: {
                    select: {
                        id: true,
                        conceptName: true,
                    },
                },
                variationPoints: {
                    select: {
                        pointName: true,
                    },
                },
            },
        });

        if (!seedItem) {
            throw new ApiError(404, `Seed item with ID ${itemId} not found`);
        }

        // 응답 데이터 구성
        const responseData = {
            id: seedItem.id,
            node_id: seedItem.nodeId,
            node_name: seedItem.conceptNode.conceptName,
            item_type: seedItem.itemType,
            difficulty: seedItem.difficulty,
            content: seedItem.content,
            answer: seedItem.answer,
            explanation: seedItem.explanation,
            variation_points: seedItem.variationPoints.map(
                (vp) => vp.pointName
            ),
            created_at: seedItem.createdAt,
            updated_at: seedItem.updatedAt,
        };

        res.status(200).json({ data: responseData });
    } catch (error) {
        next(error);
    }
};

export const getSeedItemsByNodeId = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { nodeId } = req.params;
        const { item_type, difficulty, page = 1, limit = 10 } = req.query;

        const skip = (Number(page) - 1) * Number(limit);
        const take = Number(limit);

        // 필터 조건 구성
        const where: any = { nodeId };
        if (item_type) {
            where.itemType = item_type;
        }
        if (difficulty) {
            where.difficulty = difficulty;
        }

        // 총 개수 조회
        const total = await prisma.seedItem.count({ where });

        // 씨드 문항 목록 조회
        const seedItems = await prisma.seedItem.findMany({
            where,
            include: {
                variationPoints: {
                    select: {
                        pointName: true,
                    },
                },
            },
            skip,
            take,
            orderBy: { id: "asc" },
        });

        // 응답 데이터 구성
        const formattedItems = seedItems.map((item) => ({
            id: item.id,
            node_id: item.nodeId,
            item_type: item.itemType,
            difficulty: item.difficulty,
            content: item.content,
            answer: item.answer,
            explanation: item.explanation,
            variation_points: item.variationPoints.map((vp) => vp.pointName),
            created_at: item.createdAt,
            updated_at: item.updatedAt,
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

export const createSeedItem = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { nodeId } = req.params;
        const {
            id,
            item_type,
            difficulty,
            content,
            answer,
            explanation,
            variation_points,
        } = req.body;

        // 필수 필드 검증
        if (
            !id ||
            !item_type ||
            !difficulty ||
            !content ||
            !answer ||
            !explanation
        ) {
            throw new ApiError(400, "Missing required fields");
        }

        // 노드 존재 확인
        const nodeExists = await prisma.conceptNode.findUnique({
            where: { id: nodeId },
        });

        if (!nodeExists) {
            throw new ApiError(404, `Node with ID ${nodeId} not found`);
        }

        // 중복 ID 검사
        const existingItem = await prisma.seedItem.findUnique({
            where: { id },
        });

        if (existingItem) {
            throw new ApiError(400, `Seed item with ID ${id} already exists`);
        }

        // 트랜잭션으로 씨드 문항 및 변형 요소 생성
        const result = await prisma.$transaction(async (tx) => {
            // 씨드 문항 생성
            const seedItem = await tx.seedItem.create({
                data: {
                    id,
                    nodeId,
                    itemType: item_type,
                    difficulty,
                    content,
                    answer,
                    explanation,
                },
            });

            // 변형 요소 생성 (있을 경우)
            if (variation_points && variation_points.length > 0) {
                await tx.variationPoint.createMany({
                    data: variation_points.map((point: string) => ({
                        seedItemId: seedItem.id,
                        pointName: point,
                    })),
                });
            }

            return seedItem;
        });

        res.status(201).json({
            data: {
                ...result,
                variation_points: variation_points || [],
            },
            message: "Seed item created successfully",
        });
    } catch (error) {
        next(error);
    }
};

export const updateSeedItem = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { itemId } = req.params;
        const {
            item_type,
            difficulty,
            content,
            answer,
            explanation,
            variation_points,
        } = req.body;

        // 씨드 문항 존재 확인
        const existingItem = await prisma.seedItem.findUnique({
            where: { id: itemId },
            include: {
                variationPoints: true,
            },
        });

        if (!existingItem) {
            throw new ApiError(404, `Seed item with ID ${itemId} not found`);
        }

        // 트랜잭션으로 씨드 문항 및 변형 요소 업데이트
        const result = await prisma.$transaction(async (tx) => {
            // 씨드 문항 업데이트
            const updatedItem = await tx.seedItem.update({
                where: { id: itemId },
                data: {
                    itemType: item_type || existingItem.itemType,
                    difficulty: difficulty || existingItem.difficulty,
                    content: content || existingItem.content,
                    answer: answer || existingItem.answer,
                    explanation: explanation || existingItem.explanation,
                    updatedAt: new Date(),
                },
            });

            // 변형 요소 업데이트 (있을 경우)
            if (variation_points) {
                // 기존 변형 요소 삭제
                await tx.variationPoint.deleteMany({
                    where: { seedItemId: itemId },
                });

                // 새 변형 요소 생성
                await tx.variationPoint.createMany({
                    data: variation_points.map((point: string) => ({
                        seedItemId: itemId,
                        pointName: point,
                    })),
                });
            }

            return updatedItem;
        });

        // 응답 데이터 구성
        const updatedVariationPoints =
            variation_points ||
            existingItem.variationPoints.map((vp) => vp.pointName);

        res.status(200).json({
            data: {
                ...result,
                variation_points: updatedVariationPoints,
            },
            message: "Seed item updated successfully",
        });
    } catch (error) {
        next(error);
    }
};

export const deleteSeedItem = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { itemId } = req.params;

        // 씨드 문항 존재 확인
        const existingItem = await prisma.seedItem.findUnique({
            where: { id: itemId },
        });

        if (!existingItem) {
            throw new ApiError(404, `Seed item with ID ${itemId} not found`);
        }

        // 트랜잭션으로 씨드 문항 및 관련 데이터 삭제
        await prisma.$transaction(async (tx) => {
            // 변형 요소 삭제
            await tx.variationPoint.deleteMany({
                where: { seedItemId: itemId },
            });

            // 생성된 문항의 참조 제거 (null로 설정)
            await tx.generatedItem.updateMany({
                where: { seedItemId: itemId },
                data: { seedItemId: null },
            });

            // 씨드 문항 삭제
            await tx.seedItem.delete({
                where: { id: itemId },
            });
        });

        res.status(200).json({
            message: `Seed item with ID ${itemId} deleted successfully`,
        });
    } catch (error) {
        next(error);
    }
};
