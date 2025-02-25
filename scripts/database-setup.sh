#!/bin/bash
# database-setup.sh - 프로젝트 데이터베이스 초기 설정 스크립트

echo "AI 문항 생성 시스템 데이터베이스 설정을 시작합니다..."

# PostgreSQL 서비스 확인
if ! command -v pg_isready &> /dev/null; then
    echo "PostgreSQL이 설치되어 있지 않습니다. 먼저 PostgreSQL을 설치해주세요."
    exit 1
fi

# 환경 변수 파일 확인
if [ ! -f ".env" ]; then
    echo ".env 파일이 없습니다. 샘플 파일을 생성합니다."
    cat > .env << EOL
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/knowledge_map_db?schema=public"
PORT=3000
NODE_ENV=development
EOL
    echo ".env 파일이 생성되었습니다. 필요에 따라 데이터베이스 연결 정보를 수정해주세요."
fi

# 데이터베이스 연결 문자열에서 데이터베이스 이름 추출
DB_URL=$(grep DATABASE_URL .env | cut -d '=' -f2 | tr -d '"')
DB_NAME=$(echo $DB_URL | sed -n 's/.*\/\([^?]*\).*/\1/p')
DB_USER=$(echo $DB_URL | sed -n 's/.*:\/\/\([^:]*\).*/\1/p')
DB_PASSWORD=$(echo $DB_URL | sed -n 's/.*:\([^@]*\)@.*/\1/p')
DB_HOST=$(echo $DB_URL | sed -n 's/.*@\([^:]*\).*/\1/p')
DB_PORT=$(echo $DB_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')

echo "데이터베이스 정보:"
echo "  - 호스트: $DB_HOST"
echo "  - 포트: $DB_PORT"
echo "  - 사용자: $DB_USER"
echo "  - 데이터베이스 이름: $DB_NAME"

# PostgreSQL 데이터베이스 존재 여부 확인
if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
    echo "데이터베이스 '$DB_NAME'이 이미 존재합니다."
else
    echo "데이터베이스 '$DB_NAME'을 생성합니다."
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "CREATE DATABASE $DB_NAME;"
    if [ $? -eq 0 ]; then
        echo "데이터베이스가 성공적으로 생성되었습니다."
    else
        echo "데이터베이스 생성에 실패했습니다. PostgreSQL 연결 정보를 확인해주세요."
        exit 1
    fi
fi

# Prisma 마이그레이션 실행
echo "Prisma 마이그레이션을 시작합니다..."
npx prisma migrate dev --name init

if [ $? -eq 0 ]; then
    echo "Prisma 마이그레이션이 성공적으로 완료되었습니다."
else
    echo "Prisma 마이그레이션에 실패했습니다."
    exit 1
fi

# 데이터셋 JSON 생성
echo "데이터셋 JSON 파일을 생성합니다..."
mkdir -p scripts
if [ ! -f "scripts/prepare-dataset.js" ]; then
    echo "prepare-dataset.js 스크립트가 없습니다. 스크립트를 확인해주세요."
    exit 1
fi

node scripts/prepare-dataset.js

if [ $? -eq 0 ]; then
    echo "데이터셋 JSON 파일이 성공적으로 생성되었습니다."
else
    echo "데이터셋 JSON 파일 생성에 실패했습니다."
    exit 1
fi

# 시드 데이터 로드
echo "데이터베이스에 시드 데이터를 로드합니다..."
npx ts-node prisma/seed.ts

if [ $? -eq 0 ]; then
    echo "시드 데이터가 성공적으로 로드되었습니다."
else
    echo "시드 데이터 로드에 실패했습니다."
    exit 1
fi

echo "AI 문항 생성 시스템 데이터베이스 설정이 완료되었습니다."
echo "다음 명령으로 애플리케이션을 시작할 수 있습니다: npm run dev"