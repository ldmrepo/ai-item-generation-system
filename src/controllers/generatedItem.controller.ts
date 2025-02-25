// src/controllers/generatedItem.controller.ts
import { Request, Response, NextFunction } from "express";
import { prisma } from "../index";
import { ApiError } from "../utils/apiError";

export const getGeneratedItemById = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { itemId } = req.params;

        // 생성된 문항 조회
        const generatedItem = await prisma.generatedItem.findUnique({
            where: { id: itemId },
            include: {
                conceptNode: {
                    select: {
                        id: true,
                        conceptName: true,
                    },
                },
                seedItem: {
                    select: {
                        id: true,
                    },
                },
                assessments: {
                    include: {
                        criteria: true,
                    },
                },
            },
        });

        if (!generatedItem) {
            throw new ApiError(
                404,
                `Generated item with ID ${itemId} not found`
            );
        }

        // 응답 데이터 구성
        const responseData = {
            id: generatedItem.id,
            node_id: generatedItem.nodeId,
            node_name: generatedItem.conceptNode.conceptName,
            seed_item_id: generatedItem.seedItemId,
            item_type: generatedItem.itemType,
            difficulty: generatedItem.difficulty,
            content: generatedItem.content,
            answer: generatedItem.answer,
            explanation: generatedItem.explanation,
            approved: generatedItem.approved,
            quality_score: generatedItem.qualityScore,
            generation_timestamp: generatedItem.generationTimestamp,
            assessments: generatedItem.assessments.map((assessment) => ({
                criteria_name: assessment.criteria.name,
                score: assessment.score,
                feedback: assessment.feedback,
                assessed_by: assessment.assessedBy,
                assessed_at: assessment.assessedAt,
            })),
        };

        res.status(200).json({ data: responseData });
    } catch (error) {
        next(error);
    }
};

export const updateApprovalStatus = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { itemId } = req.params;
        const { approved, feedback } = req.body;

        // 필수 필드 검증
        if (approved === undefined) {
            throw new ApiError(400, "Missing required field: approved");
        }

        // 생성된 문항 존재 확인
        const existingItem = await prisma.generatedItem.findUnique({
            where: { id: itemId },
        });

        if (!existingItem) {
            throw new ApiError(
                404,
                `Generated item with ID ${itemId} not found`
            );
        }

        // 승인 상태 업데이트
        const updatedItem = await prisma.generatedItem.update({
            where: { id: itemId },
            data: {
                approved: approved,
            },
        });

        // 피드백이 있을 경우 평가 정보 추가
        if (feedback) {
            // 'approval' 평가 기준 조회 또는 생성
            let approvalCriteria = await prisma.assessmentCriteria.findFirst({
                where: { name: "approval_feedback" },
            });

            if (!approvalCriteria) {
                approvalCriteria = await prisma.assessmentCriteria.create({
                    data: {
                        name: "approval_feedback",
                        description: "문항 승인/반려 시 제공된 피드백",
                        weight: 1.0,
                        evaluationMethod: "manual",
                        passingThreshold: 0.5,
                    },
                });
            }

            // 기존 승인 피드백 확인
            const existingFeedback = await prisma.itemAssessment.findFirst({
                where: {
                    generatedItemId: itemId,
                    criteriaId: approvalCriteria.id,
                },
            });

            // 피드백 저장 또는 업데이트
            if (existingFeedback) {
                await prisma.itemAssessment.update({
                    where: { id: existingFeedback.id },
                    data: {
                        score: approved ? 1.0 : 0.0,
                        feedback: feedback,
                        assessedBy: req.body.assessed_by,
                        assessedAt: new Date(),
                    },
                });
            } else {
                await prisma.itemAssessment.create({
                    data: {
                        generatedItemId: itemId,
                        criteriaId: approvalCriteria.id,
                        score: approved ? 1.0 : 0.0,
                        feedback: feedback,
                        assessedBy: req.body.assessed_by,
                        assessedAt: new Date(),
                    },
                });
            }
        }

        res.status(200).json({
            data: {
                id: updatedItem.id,
                approved: updatedItem.approved,
            },
            message: `Item ${approved ? "approved" : "rejected"} successfully`,
        });
    } catch (error) {
        next(error);
    }
};

export const assessItem = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { itemId } = req.params;
        const { criteria, assessed_by } = req.body;

        // 필수 필드 검증
        if (!criteria || !Array.isArray(criteria) || criteria.length === 0) {
            throw new ApiError(400, "Missing or invalid criteria array");
        }

        // 생성된 문항 존재 확인
        const existingItem = await prisma.generatedItem.findUnique({
            where: { id: itemId },
        });

        if (!existingItem) {
            throw new ApiError(
                404,
                `Generated item with ID ${itemId} not found`
            );
        }

        // 평가 기준 ID 목록 확인
        const criteriaIds = criteria.map((c) => c.criteria_id);
        const existingCriteria = await prisma.assessmentCriteria.findMany({
            where: {
                id: {
                    in: criteriaIds,
                },
            },
        });

        if (existingCriteria.length !== criteriaIds.length) {
            throw new ApiError(400, "One or more criteria IDs are invalid");
        }

        // 트랜잭션으로 평가 정보 저장
        const assessments = await prisma.$transaction(async (tx) => {
            const results = [];

            for (const criterionData of criteria) {
                // 기존 평가 확인
                const existingAssessment = await tx.itemAssessment.findFirst({
                    where: {
                        generatedItemId: itemId,
                        criteriaId: criterionData.criteria_id,
                    },
                });

                // 평가 저장 또는 업데이트
                let assessment;
                if (existingAssessment) {
                    assessment = await tx.itemAssessment.update({
                        where: { id: existingAssessment.id },
                        data: {
                            score: criterionData.score,
                            feedback: criterionData.feedback,
                            assessedBy: assessed_by,
                            assessedAt: new Date(),
                        },
                        include: {
                            criteria: true,
                        },
                    });
                } else {
                    assessment = await tx.itemAssessment.create({
                        data: {
                            generatedItemId: itemId,
                            criteriaId: criterionData.criteria_id,
                            score: criterionData.score,
                            feedback: criterionData.feedback,
                            assessedBy: assessed_by,
                            assessedAt: new Date(),
                        },
                        include: {
                            criteria: true,
                        },
                    });
                }

                results.push(assessment);
            }

            // 평균 품질 점수 계산 및 업데이트
            const allAssessments = await tx.itemAssessment.findMany({
                where: { generatedItemId: itemId },
                include: { criteria: true },
            });

            if (allAssessments.length > 0) {
                const weightedSum = allAssessments.reduce(
                    (sum, assessment) =>
                        sum + assessment.score * assessment.criteria.weight,
                    0
                );
                const totalWeight = allAssessments.reduce(
                    (sum, assessment) => sum + assessment.criteria.weight,
                    0
                );
                const averageScore =
                    totalWeight > 0 ? weightedSum / totalWeight : 0;

                await tx.generatedItem.update({
                    where: { id: itemId },
                    data: {
                        qualityScore: averageScore,
                    },
                });
            }

            return results;
        });

        // 응답 데이터 구성
        const responseData = assessments.map((assessment) => ({
            criteria_id: assessment.criteriaId,
            criteria_name: assessment.criteria.name,
            score: assessment.score,
            feedback: assessment.feedback,
            assessed_by: assessment.assessedBy,
            assessed_at: assessment.assessedAt,
        }));

        res.status(200).json({
            data: responseData,
            message: "Item assessment saved successfully",
        });
    } catch (error) {
        next(error);
    }
};

export const getItemAssessments = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const { itemId } = req.params;

        // 생성된 문항 존재 확인
        const existingItem = await prisma.generatedItem.findUnique({
            where: { id: itemId },
        });

        if (!existingItem) {
            throw new ApiError(
                404,
                `Generated item with ID ${itemId} not found`
            );
        }

        // 평가 정보 조회
        const assessments = await prisma.itemAssessment.findMany({
            where: { generatedItemId: itemId },
            include: {
                criteria: true,
            },
            orderBy: { assessedAt: "desc" },
        });

        // 응답 데이터 구성
        const responseData = assessments.map((assessment) => ({
            id: assessment.id,
            criteria_id: assessment.criteriaId,
            criteria_name: assessment.criteria.name,
            score: assessment.score,
            feedback: assessment.feedback,
            assessed_by: assessment.assessedBy,
            assessed_at: assessment.assessedAt,
        }));

        res.status(200).json({ data: responseData });
    } catch (error) {
        next(error);
    }
};
