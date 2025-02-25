// src/index.ts
import express from "express";
import cors from "cors";
import helmet from "helmet";
import { PrismaClient } from "@prisma/client";
import { nodeRouter } from "./routes/node.routes";
import { seedItemRouter } from "./routes/seedItem.routes";
import { knowledgeMapRouter } from "./routes/knowledgeMap.routes";
import { itemGenerationRouter } from "./routes/itemGeneration.routes";
import { generatedItemRouter } from "./routes/generatedItem.routes";
import { systemRouter } from "./routes/system.routes";
import errorHandler from "./middleware/errorHandler";

const app = express();
const prisma = new PrismaClient();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// Routes
app.use("/api/v1/knowledge-maps", knowledgeMapRouter);
app.use("/api/v1/nodes", nodeRouter);
app.use("/api/v1/seed-items", seedItemRouter);
app.use("/api/v1/item-generation", itemGenerationRouter);
app.use("/api/v1/generated-items", generatedItemRouter);
app.use("/api/v1", systemRouter);

// Error handling
app.use(errorHandler);

// Start server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

// Graceful shutdown
process.on("SIGINT", async () => {
    await prisma.$disconnect();
    process.exit(0);
});

export { prisma };
