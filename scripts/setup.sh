#!/bin/bash
# setup.sh - AI 문항 생성 시스템 설정 스크립트

echo "AI 문항 생성 시스템 설정을 시작합니다..."

# Node.js 설치 확인
if ! command -v node &> /dev/null; then
    echo "Node.js가 설치되어 있지 않습니다. 먼저 Node.js를 설치해주세요."
    exit 1
fi

# npm 설치 확인
if ! command -v npm &> /dev/null; then
    echo "npm이 설치되어 있지 않습니다. 먼저 npm을 설치해주세요."
    exit 1
fi

# 프로젝트 초기화 (package.json이 없는 경우)
if [ ! -f "package.json" ]; then
    echo "package.json 파일이 없습니다. 프로젝트를 초기화합니다."
    npm init -y
    
    # 필요한 의존성 추가
    npm pkg set scripts.start="node dist/index.js"
    npm pkg set scripts.dev="nodemon src/index.ts"
    npm pkg set scripts.build="tsc"
    npm pkg set scripts.prisma:generate="prisma generate"
    npm pkg set scripts.prisma:migrate="prisma migrate dev"
    npm pkg set scripts.prisma:studio="prisma studio"
    npm pkg set scripts.seed="ts-node prisma/seed.ts"
    npm pkg set scripts.test="jest"
fi

# 필요한 의존성 설치
echo "필요한 의존성을 설치합니다..."
npm install @prisma/client cors dotenv express helmet uuid
npm install -D @types/cors @types/express @types/node @types/uuid jest nodemon prisma ts-node typescript

# 프로젝트 디렉토리 구조 생성
echo "프로젝트 디렉토리 구조를 생성합니다..."
mkdir -p src/{controllers,routes,middleware,utils,tests}
mkdir -p prisma/{migrations,data}

# TypeScript 설정 파일 생성
if [ ! -f "tsconfig.json" ]; then
    echo "TypeScript 설정 파일을 생성합니다..."
    cat > tsconfig.json << EOL
{
  "compilerOptions": {
    "target": "es2016",
    "module": "commonjs",
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "strict": true,
    "skipLibCheck": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "typeRoots": ["./node_modules/@types"]
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "**/*.test.ts"]
}
EOL
fi

# Git 초기화 및 .gitignore 생성
if [ ! -d ".git" ]; then
    echo "Git 저장소를 초기화합니다..."
    git init
    
    cat > .gitignore << EOL
# Dependencies
node_modules/
npm-debug.log
yarn-error.log
yarn-debug.log

# Production build
dist/
build/

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDE files
.vscode/
.idea/
*.sublime-*

# Logs
logs/
*.log

# Misc
.DS_Store
.cache/
coverage/
EOL
fi

# 데이터베이스 설정 스크립트 권한 설정 및 실행
if [ -f "database-setup.sh" ]; then
    echo "데이터베이스 설정을 시작합니다..."
    chmod +x database-setup.sh
    ./database-setup.sh
else
    echo "database-setup.sh 스크립트가 없습니다. 스크립트를 확인해주세요."
fi

echo "AI 문항 생성 시스템 설정이 완료되었습니다."
echo "이제 'npm run dev' 명령으로 애플리케이션을 시작할 수 있습니다."