// src/routes/node.routes.ts
import { Router } from "express";
import * as nodeController from "../controllers/node.controller";

const router = Router();

// 노드 조회
router.get("/:nodeId", nodeController.getNodeById);

// 노드별 연결 정보 조회
router.get("/:nodeId/connections", nodeController.getNodeConnections);

// 노드 삭제
router.delete("/:nodeId", nodeController.deleteNode);

// 노드 연결 생성
router.post("/:sourceNodeId/connections", nodeController.createNodeConnection);

// 노드 연결 삭제
router.delete(
    "/:sourceNodeId/connections/:targetNodeId",
    nodeController.deleteNodeConnection
);

export const nodeRouter = router;

// src/routes/knowledgeMap.routes.ts
import { Router } from "express";
import * as knowledgeMapController from "../controllers/knowledgeMap.controller";
import * as nodeController from "../controllers/node.controller";

const router = Router();

// 지식맵 목록 조회
router.get("/", knowledgeMapController.getKnowledgeMaps);

// 지식맵 상세 조회
router.get("/:id", knowledgeMapController.getKnowledgeMapById);

// 지식맵 생성
router.post("/", knowledgeMapController.createKnowledgeMap);

// 지식맵 수정
router.put("/:id", knowledgeMapController.updateKnowledgeMap);

// 지식맵 삭제
router.delete("/:id", knowledgeMapController.deleteKnowledgeMap);

// 지식맵별 노드 목록 조회
router.get("/:mapId/nodes", nodeController.getNodesByMapId);

// 지식맵에 노드 생성
router.post("/:mapId/nodes", nodeController.createNode);

export const knowledgeMapRouter = router;

// src/routes/seedItem.routes.ts
import { Router } from "express";
import * as seedItemController from "../controllers/seedItem.controller";

const router = Router();

// 씨드 문항 상세 조회
router.get("/:itemId", seedItemController.getSeedItemById);

// 씨드 문항 수정
router.put("/:itemId", seedItemController.updateSeedItem);

// 씨드 문항 삭제
router.delete("/:itemId", seedItemController.deleteSeedItem);

// 노드별 씨드 문항 목록 조회 및 생성 라우트는 node.routes.ts에 추가
// 노드별 씨드 문항 목록 조회
router.get(
    "/nodes/:nodeId/seed-items",
    seedItemController.getSeedItemsByNodeId
);

// 노드에 씨드 문항 생성
router.post("/nodes/:nodeId/seed-items", seedItemController.createSeedItem);

export const seedItemRouter = router;

// src/routes/itemGeneration.routes.ts
import { Router } from "express";
import * as itemGenerationController from "../controllers/itemGeneration.controller";

const router = Router();

// 문항 생성 요청
router.post("/generate", itemGenerationController.generateItems);

// 문항 생성 상태 조회
router.get("/:requestId/status", itemGenerationController.getGenerationStatus);

// 생성된 문항 목록 조회
router.get("/:requestId/items", itemGenerationController.getGeneratedItems);

export const itemGenerationRouter = router;

// src/routes/generatedItem.routes.ts
import { Router } from "express";
import * as generatedItemController from "../controllers/generatedItem.controller";

const router = Router();

// 생성된 문항 상세 조회
router.get("/:itemId", generatedItemController.getGeneratedItemById);

// 생성된 문항 승인/반려
router.patch("/:itemId/approval", generatedItemController.updateApprovalStatus);

// 생성된 문항 평가
router.post("/:itemId/assessments", generatedItemController.assessItem);

// 문항 평가 결과 조회
router.get("/:itemId/assessments", generatedItemController.getItemAssessments);

export const generatedItemRouter = router;

// src/routes/system.routes.ts
import { Router } from "express";
import * as systemController from "../controllers/system.controller";

const router = Router();

// 평가 기준 목록 조회
router.get("/assessment-criteria", systemController.getAssessmentCriteria);

// 문항 유형 목록 조회
router.get("/item-types", systemController.getItemTypes);

// 난이도 수준 목록 조회
router.get("/difficulty-levels", systemController.getDifficultyLevels);

export const systemRouter = router;
