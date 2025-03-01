#!/bin/bash

# Neo4j 도커 이미지 풀링
echo "Neo4j 도커 이미지 다운로드 중..."
docker pull neo4j:5.13.0

# Neo4j 데이터, 로그, 설정 파일을 위한 볼륨 디렉토리 생성
echo "Neo4j를 위한 볼륨 디렉토리 생성 중..."
mkdir -p $HOME/neo4j/data
mkdir -p $HOME/neo4j/logs
mkdir -p $HOME/neo4j/conf
mkdir -p $HOME/neo4j/import

# Neo4j 도커 컨테이너 실행
echo "Neo4j 도커 컨테이너 실행 중..."
docker run \
    --name neo4j-curriculum \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/conf:/conf \
    -v $HOME/neo4j/import:/import \
    -e NEO4J_AUTH=neo4j/curriculum2022 \
    -e NEO4J_dbms_memory_pagecache_size=1G \
    -e NEO4J_dbms.memory.heap.initial_size=1G \
    -e NEO4J_dbms_memory_heap_max__size=2G \
    neo4j:5.13.0

echo "Neo4j 컨테이너가 시작되었습니다."
echo "브라우저 접속 주소: http://localhost:7474"
echo "사용자 이름: neo4j"
echo "비밀번호: curriculum2022"
echo ""
echo "컨테이너 상태 확인:"
docker ps | grep neo4j-curriculum