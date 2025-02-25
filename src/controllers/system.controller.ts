// src/controllers/system.controller.ts
import { Request, Response, NextFunction } from "express";
import { prisma } from "../index";

// 평가 기준 목록 조회
export const getAssessmentCriteria = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const criteria = await prisma.assessmentCriteria.findMany({
            orderBy: { name: "asc" },
        });

        res.status(200).json({ data: criteria });
    } catch (error) {
        next(error);
    }
};

// 문항 유형 목록 조회
export const getItemTypes = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        // 문항 유형은 데이터베이스에서 enum 값이 아니기 때문에
        // 기존 데이터에서 distinct 값을 추출하거나 정의된 상수 배열 사용
        const itemTypes = [
            {
                id: "conceptual",
                name: "개념 이해",
                description: "개념에 대한 이해를 평가하는 문항",
            },
            {
                id: "calculation",
                name: "계산 적용",
                description: "수식이나 공식을 적용하여 계산하는 문항",
            },
            {
                id: "graph_interpretation",
                name: "그래프 해석",
                description: "그래프를 해석하고 분석하는 문항",
            },
            {
                id: "application",
                name: "실생활 응용",
                description: "실생활 상황에 개념을 적용하는 문항",
            },
            {
                id: "problem_solving",
                name: "문제 해결",
                description: "복합적인 문제 해결 능력을 평가하는 문항",
            },
            {
                id: "plotting",
                name: "그래프 그리기",
                description: "주어진 조건에 맞는 그래프를 그리는 문항",
            },
            {
                id: "identification",
                name: "식별하기",
                description: "주어진 조건에 맞는 대상을 식별하는 문항",
            },
            {
                id: "analysis",
                name: "분석",
                description: "주어진 상황이나 데이터를 분석하는 문항",
            },
            {
                id: "interpretation",
                name: "해석하기",
                description: "주어진 정보를 해석하는 문항",
            },
            {
                id: "graphing",
                name: "작도하기",
                description: "수학적 대상을 작도하는 문항",
            },
            {
                id: "real_world",
                name: "실제 상황",
                description: "실제 상황에서의 문제를 해결하는 문항",
            },
        ];

        res.status(200).json({ data: itemTypes });
    } catch (error) {
        next(error);
    }
};

// 난이도 수준 목록 조회
export const getDifficultyLevels = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        // 난이도 수준도 데이터베이스에서 enum 값이 아니기 때문에
        // 기존 데이터에서 distinct 값을 추출하거나 정의된 상수 배열 사용
        const difficultyLevels = [
            {
                id: "basic",
                name: "기초",
                description: "기본 개념과 원리를 이해하고 있는지 평가하는 수준",
            },
            {
                id: "intermediate",
                name: "중급",
                description:
                    "개념을 응용하여 문제를 해결할 수 있는지 평가하는 수준",
            },
            {
                id: "advanced",
                name: "고급",
                description:
                    "복합적 사고와 창의적인 문제 해결 능력을 평가하는 수준",
            },
        ];

        res.status(200).json({ data: difficultyLevels });
    } catch (error) {
        next(error);
    }
};
