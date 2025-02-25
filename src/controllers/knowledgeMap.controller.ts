// src/controllers/knowledgeMap.controller.ts
import { Request, Response, NextFunction } from "express";
import { prisma } from "../index";
import { ApiError } from "../utils/apiError";

export const getKnowledgeMaps = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { subject, grade, unit, page = 1, limit = 10 } = req.query;

        const skip = (Number(page) - 1) * Number(limit);
        const take = Number(limit);

        // 필터 조건 구성
        const where: any = {};
        if (subject) {
            where.subject = subject;
        }
        if (grade) {
            where.grade = grade;
        }
        if (unit) {
            where.unit = unit;
        }

        // 총 개수 조회
        const total = await prisma.knowledgeMap.count({ where });

        // 지식맵 목록 조회
        const knowledgeMaps = await prisma.knowledgeMap.findMany({
            where,
            skip,
            take,
            orderBy: { id: "desc" },
        });

        res.status(200).json({
            data: knowledgeMaps,
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

export const getKnowledgeMapById = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { id } = req.params;

        // 지식맵 조회
        const knowledgeMap = await prisma.knowledgeMap.findUnique({
            where: { id: Number(id) },
            include: {
                conceptNodes: {
                    select: {
                        id: true,
                        conceptName: true,
                        difficultyLevel: true,
                    },
                },
            },
        });

        if (!knowledgeMap) {
            throw new ApiError(404, `Knowledge Map with ID ${id} not found`);
        }

        // 연결 정보 조회
        const connections = await prisma.nodeConnection.findMany({
            where: {
                sourceNode: {
                    mapId: Number(id),
                },
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

        // 응답 데이터 구성
        const responseData = {
            ...knowledgeMap,
            nodes: knowledgeMap.conceptNodes,
            edges: connections.map((conn) => ({
                source: conn.sourceNodeId,
                target: conn.targetNodeId,
                type: conn.connectionType,
                strength: conn.strength,
            })),
        };

        res.status(200).json({ data: responseData });
    } catch (error) {
        next(error);
    }
};

export const createKnowledgeMap = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { subject, grade, unit, version, description } = req.body;

        // 필수 필드 검증
        if (!subject || !grade || !unit || !version) {
            throw new ApiError(400, "Missing required fields");
        }

        // 지식맵 생성
        const knowledgeMap = await prisma.knowledgeMap.create({
            data: {
                subject,
                grade,
                unit,
                version,
                creationDate: new Date(),
                description,
            },
        });

        res.status(201).json({
            data: knowledgeMap,
            message: "Knowledge Map created successfully",
        });
    } catch (error) {
        next(error);
    }
};

export const updateKnowledgeMap = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { id } = req.params;
        const { subject, grade, unit, version, description } = req.body;

        // 지식맵 존재 확인
        const existingMap = await prisma.knowledgeMap.findUnique({
            where: { id: Number(id) },
        });

        if (!existingMap) {
            throw new ApiError(404, `Knowledge Map with ID ${id} not found`);
        }

        // 지식맵 업데이트
        const updatedMap = await prisma.knowledgeMap.update({
            where: { id: Number(id) },
            data: {
                subject: subject || existingMap.subject,
                grade: grade || existingMap.grade,
                unit: unit || existingMap.unit,
                version: version || existingMap.version,
                description:
                    description !== undefined
                        ? description
                        : existingMap.description,
            },
        });

        res.status(200).json({
            data: updatedMap,
            message: "Knowledge Map updated successfully",
        });
    } catch (error) {
        next(error);
    }
};

export const deleteKnowledgeMap = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { id } = req.params;

        // 지식맵 존재 확인
        const existingMap = await prisma.knowledgeMap.findUnique({
            where: { id: Number(id) },
            include: {
                conceptNodes: true,
            },
        });

        if (!existingMap) {
            throw new ApiError(404, `Knowledge Map with ID ${id} not found`);
        }

        // 지식맵에 노드가 있는 경우 삭제 불가
        if (existingMap.conceptNodes.length > 0) {
            throw new ApiError(
                400,
                "Cannot delete Knowledge Map with existing nodes. Delete all nodes first."
            );
        }

        // 지식맵 삭제
        await prisma.knowledgeMap.delete({
            where: { id: Number(id) },
        });

        res.status(200).json({
            message: `Knowledge Map with ID ${id} deleted successfully`,
        });
    } catch (error) {
        next(error);
    }
};
