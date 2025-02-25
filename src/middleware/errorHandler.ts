// src/middleware/errorHandler.ts
import { Request, Response, NextFunction } from "express";
import { ApiError } from "../utils/apiError";
import { Prisma } from "@prisma/client";

const errorHandler = (
    err: Error,
    req: Request,
    res: Response,
    next: NextFunction
) => {
    console.error("Error:", err);

    // Prisma 관련 에러 처리
    if (err instanceof Prisma.PrismaClientKnownRequestError) {
        if (err.code === "P2002") {
            // 유니크 제약 조건 위반
            return res.status(409).json({
                status: "error",
                message: "Resource with this identifier already exists",
                error: err.message,
            });
        } else if (err.code === "P2025") {
            // 레코드를 찾을 수 없음
            return res.status(404).json({
                status: "error",
                message: "Resource not found",
                error: err.message,
            });
        }

        return res.status(400).json({
            status: "error",
            message: "Database error",
            error: err.message,
        });
    }

    // API 에러 처리
    if (err instanceof ApiError) {
        return res.status(err.statusCode).json({
            status: err.statusCode < 500 ? "fail" : "error",
            message: err.message,
        });
    }

    // 그 외 에러는 500 Internal Server Error로 처리
    return res.status(500).json({
        status: "error",
        message: "Internal server error",
        error: process.env.NODE_ENV === "development" ? err.message : undefined,
    });
};

export default errorHandler;
