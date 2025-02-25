# AI 기반 문항 자동 생성 시스템

지식맵과 씨드 문항을 기반으로 교육용 문항을 자동으로 생성하는 AI 시스템입니다.

## 기능 개요

-   교과별 지식맵 구축 및 관리
-   개념 노드 간 연결 관계 설정
-   씨드 문항 기반 AI 문항 자동 생성
-   생성된 문항 평가 및 관리
-   RESTful API를 통한 접근

## 기술 스택

-   **Backend**: Node.js, Express, TypeScript
-   **ORM**: Prisma
-   **Database**: PostgreSQL
-   **API**: RESTful API
-   **문항 생성**: 템플릿 기반 AI 변형 생성

## 빠른 시작

### 사전 요구사항

-   Node.js 16.x 이상
-   npm 또는 yarn
-   PostgreSQL 13.x 이상

### 설치 및 실행

1. 저장소 클론

```bash
git clone https://github.com/yourusername/ai-item-generation-system.git
cd ai-item-generation-system
```

2. 자동 설정 스크립트 실행

```bash
chmod +x setup.sh
./setup.sh
```

이 스크립트는 다음 작업을 자동으로 수행합니다:

-   의존성 설치
-   프로젝트 구조 생성
-   데이터베이스 생성 및 마이그레이션
-   초기 데이터 시딩

3. 애플리케이션 실행

```bash
npm run dev
```

4. API 접근

API는 기본적으로 `http://localhost:3000/api/v1/`에서 접근할 수 있습니다.

## 수동 설정

자동 설정 스크립트가 작동하지 않는 경우 다음 단계를 순서대로
실행하세요:

1. 환경 설정

`.env` 파일을 생성하고 데이터베이스 연결 정보를 설정합니다:

```
DATABASE_URL="postgresql://username:password@localhost:5432/knowledge_map_db?schema=public"
PORT=3000
NODE_ENV=development
```

2. 의존성 설치

```bash
npm install
```

3. 데이터베이스 마이그레이션

```bash
npx prisma migrate dev --name init
```

4. 데이터 준비 및 시딩

```bash
node scripts/prepare-dataset.js
npx ts-node prisma/seed.ts
```

5. 애플리케이션 실행

```bash
npm run dev
```

## 디렉토리 구조

```
.
├── prisma/                  # Prisma 설정 및 마이그레이션
│   ├── schema.prisma        # 데이터베이스 스키마 정의
│   ├── seed.ts              # 데이터베이스 시드 스크립트
│   └── data/                # 시드 데이터 파일
├── scripts/                 # 유틸리티 스크립트
│   └── prepare-dataset.js   # 데이터셋 준비 스크립트
├── src/                     # 소스 코드
│   ├── controllers/         # API 컨트롤러
│   ├── routes/              # API 라우트 정의
│   ├── middleware/          # Express 미들웨어
│   ├── utils/               # 유틸리티 함수
│   └── index.ts             # 애플리케이션 진입점
├── .env                     # 환경 변수
├── package.json             # npm 패키지 정의
└── tsconfig.json            # TypeScript 설정
```

## API 문서

### 지식맵 API

-   `GET /api/v1/knowledge-maps` - 지식맵 목록 조회
-   `GET /api/v1/knowledge-maps/{id}` - 지식맵 상세 조회
-   `POST /api/v1/knowledge-maps` - 지식맵 생성
-   `PUT /api/v1/knowledge-maps/{id}` - 지식맵 수정
-   `DELETE /api/v1/knowledge-maps/{id}` - 지식맵 삭제

### 개념 노드 API

-   `GET /api/v1/knowledge-maps/{mapId}/nodes` - 지식맵 내 노드 목록 조회
-   `GET /api/v1/nodes/{nodeId}` - 노드 상세 조회
-   `POST /api/v1/knowledge-maps/{mapId}/nodes` - 노드 생성
-   `DELETE /api/v1/nodes/{nodeId}` - 노드 삭제

### 씨드 문항 API

-   `GET /api/v1/nodes/{nodeId}/seed-items` - 노드별 씨드 문항 목록 조회
-   `GET /api/v1/seed-items/{itemId}` - 씨드 문항 상세 조회
-   `POST /api/v1/nodes/{nodeId}/seed-items` - 씨드 문항 생성
-   `PUT /api/v1/seed-items/{itemId}` - 씨드 문항 수정
-   `DELETE /api/v1/seed-items/{itemId}` - 씨드 문항 삭제

### 문항 생성 API

-   `POST /api/v1/item-generation/generate` - 문항 생성 요청
-   `GET /api/v1/item-generation/{requestId}/status` - 생성 상태 조회
-   `GET /api/v1/item-generation/{requestId}/items` - 생성된 문항 목록 조회

### 생성된 문항 관리 API

-   `GET /api/v1/generated-items/{itemId}` - 생성된 문항 상세 조회
-   `PATCH /api/v1/generated-items/{itemId}/approval` - 문항 승인/반려
-   `POST /api/v1/generated-items/{itemId}/assessments` - 문항 평가

## 개발 정보

### 데이터베이스 살펴보기

Prisma Studio를 사용해 데이터베이스를 시각적으로 탐색할 수 있습니다:

```bash
npm run prisma:studio
```

브라우저에서 `http://localhost:5555`로 접속하여 데이터를 확인하세요.

### API 테스트

Postman 또는 유사한 API 클라이언트를 사용하여 API를 테스트할 수 있습니다.
`api-usage-example.js` 파일에 API 사용 예제가 포함되어 있습니다.

## 라이선스

MIT

## 기여 방법

1. 이 저장소를 포크합니다
2. 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다
