// src/utils/asyncHandler.ts
import { Request, Response, NextFunction } from "express";

// 비동기 컨트롤러 래퍼 함수 (try-catch 자동화)
export const asyncHandler = (
    fn: (req: Request, res: Response, next: NextFunction) => Promise<any>
) => {
    return (req: Request, res: Response, next: NextFunction) => {
        Promise.resolve(fn(req, res, next)).catch((err) => next(err));
    };
};
