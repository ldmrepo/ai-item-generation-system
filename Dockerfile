# Dockerfile
FROM node:18-alpine

WORKDIR /app

# 앱 의존성 설치
COPY package*.json ./
RUN npm ci

# 앱 소스 복사
COPY . .

# Prisma 클라이언트 생성
RUN npx prisma generate

# TypeScript 컴파일
RUN npm run build

# 포트 노출
EXPOSE 3000

# 앱 실행
CMD ["node", "dist/index.js"]

# ------ docker-compose.yml ------
version: '3.8'

services:
  # PostgreSQL 데이터베이스
  postgres:
    image: postgres:14-alpine
    container_name: knowledge-map-db
    restart: always
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: knowledge_map_db
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - app-network

  # 백엔드 API 서버
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: knowledge-map-api
    restart: always
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/knowledge_map_db?schema=public
    depends_on:
      - postgres
    networks:
      - app-network
    command: >
      sh -c "
        echo 'Waiting for PostgreSQL to be ready...' &&
        npx prisma migrate deploy &&
        node dist/index.js
      "

volumes:
  postgres-data:

networks:
  app-network:
    driver: bridge

# ------ .dockerignore ------
node_modules
npm-debug.log
yarn-error.log
dist
build
.env
.env.local
.env.development
.