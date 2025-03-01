#!/usr/bin/env python3

"""
세 개의 영역 JSON 파일을 Neo4j 데이터베이스에 로딩하는 코드를 작성해드리겠습니다.

이 스크립트는 '수와 연산', '변화와 관계', '도형과 측정' 세 개의 영역 JSON 파일을 Neo4j 데이터베이스에 로딩하는 코드입니다. 주요 기능은 다음과 같습니다:

## 주요 기능

1. **기본 노드 및 제약 조건 생성**:
   - 교육과정(Curriculum) 노드 생성
   - 학년군(GradeGroup) 노드 생성
   - 내용 범주(ContentCategory) 노드 생성
   - 제약 조건 및 인덱스 생성

2. **영역별 데이터 로딩**:
   - 영역(Domain) 노드 생성 및 교육과정과 연결
   - 핵심 아이디어(CoreIdea) 노드 생성 및 영역과 연결
   - 내용 요소(ContentElement) 노드 생성 및 관계 설정
   - 영역과 학년군 간의 관계(APPLICABLE_TO) 설정

3. **오류 처리 및 로깅**:
   - 각 단계별 진행 상황 로깅
   - 파일 존재 여부 확인
   - 데이터 불일치 확인 및 처리

## 사용 방법

1. 각 영역 JSON 파일이 `data/curriculum` 디렉토리에 저장되어 있어야 합니다:
   - `수와 연산.json`
   - `변화와 관계.json`
   - `도형과 측정.json`

2. Neo4j 데이터베이스가 실행 중이어야 합니다:
   ```bash
   docker start neo4j-curriculum
   ```

3. 필요한 Python 패키지 설치:
   ```bash
   pip install neo4j
   ```

4. 스크립트 실행:
   ```bash
   python domain-loading-script.py
   ```

이 스크립트를 실행하면 세 개의 영역에 대한 모든 데이터(영역 정보, 핵심 아이디어, 내용 요소)가 Neo4j 데이터베이스에 로드됩니다. 이로써 교육과정의 내용 체계를 그래프 데이터베이스에서 효과적으로 관리하고 탐색할 수 있습니다.

파일이 없거나 데이터 형식이 맞지 않는 경우 적절한 경고 메시지가 표시되며, 가능한 경우 오류를 복구하고 로딩을 계속합니다.
"""

"""
2022 개정 교육과정 수학과 영역별 내용 체계 JSON 파일을 Neo4j에 로딩하는 스크립트
'수와 연산', '변화와 관계', '도형과 측정' 영역 데이터를 로딩합니다.
"""

import json
import os
import logging
import sys
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('domain_loading.log')
    ]
)

# Neo4j 연결 정보
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "curriculum2022"

# 데이터 파일 경로
DATA_DIR = "data/curriculum"
DOMAIN_FILES = {
    "D01": os.path.join(DATA_DIR, "수와 연산.json"),
    "D02": os.path.join(DATA_DIR, "변화와 관계.json"),
    "D03": os.path.join(DATA_DIR, "도형과 측정.json")
}

class DomainLoader:
    def __init__(self, uri, user, password):
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logging.info("Neo4j 데이터베이스에 연결됨")
        except ServiceUnavailable as e:
            logging.error(f"Neo4j 데이터베이스 연결 실패: {str(e)}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            logging.info("Neo4j 데이터베이스 연결 종료")

    def load_all_domains(self, domain_files):
        """모든 영역 데이터 로딩"""
        try:
            # 기본 노드 및 제약조건 생성
            self._create_constraints_and_indexes()
            self._create_curriculum_node()
            self._create_grade_groups()
            self._create_content_categories()
            
            # 영역별 데이터 로딩
            for domain_id, file_path in domain_files.items():
                if not os.path.exists(file_path):
                    logging.warning(f"파일이 존재하지 않음: {file_path}")
                    continue
                
                self.load_domain_data(domain_id, file_path)
            
            logging.info("모든 영역 데이터 로딩 완료")
            return True
        
        except Exception as e:
            logging.error(f"영역 데이터 로딩 중 오류 발생: {str(e)}")
            return False

    def load_domain_data(self, domain_id, file_path):
        """특정 영역 데이터 로딩"""
        try:
            # JSON 파일 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            domain = data.get("domain")
            core_idea = data.get("coreIdea")
            content_elements = data.get("contentElements", [])
            relationships = data.get("relationships", [])
            
            # 도메인 ID 확인 및 설정
            if domain and domain.get("id") != domain_id:
                logging.warning(f"파일 내 도메인 ID({domain.get('id')})가 예상 ID({domain_id})와 다릅니다. 예상 ID로 대체합니다.")
                domain["id"] = domain_id
            
            # 데이터 로딩
            self._create_domain_node(domain)
            if core_idea:
                self._create_core_idea_node(core_idea, domain_id)
            
            self._create_content_elements(content_elements)
            self._create_domain_relationships(relationships)
            
            logging.info(f"영역 '{domain.get('name')}' 데이터 로딩 완료")
            return True
        
        except Exception as e:
            logging.error(f"영역 '{domain_id}' 데이터 로딩 중 오류 발생: {str(e)}")
            return False

    def _create_constraints_and_indexes(self):
        """제약 조건 및 인덱스 생성"""
        with self.driver.session() as session:
            # 제약 조건 생성
            constraints = [
                "CREATE CONSTRAINT curriculum_id_constraint IF NOT EXISTS FOR (c:Curriculum) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT domain_id_constraint IF NOT EXISTS FOR (d:Domain) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT grade_group_id_constraint IF NOT EXISTS FOR (g:GradeGroup) REQUIRE g.id IS UNIQUE",
                "CREATE CONSTRAINT content_category_id_constraint IF NOT EXISTS FOR (cc:ContentCategory) REQUIRE cc.id IS UNIQUE",
                "CREATE CONSTRAINT content_element_id_constraint IF NOT EXISTS FOR (ce:ContentElement) REQUIRE ce.id IS UNIQUE",
                "CREATE CONSTRAINT core_idea_id_constraint IF NOT EXISTS FOR (ci:CoreIdea) REQUIRE ci.id IS UNIQUE"
            ]
            
            # 인덱스 생성
            indexes = [
                "CREATE INDEX domain_name_index IF NOT EXISTS FOR (d:Domain) ON (d.name)",
                "CREATE INDEX content_element_content_index IF NOT EXISTS FOR (ce:ContentElement) ON (ce.content)",
                "CREATE INDEX grade_group_name_index IF NOT EXISTS FOR (g:GradeGroup) ON (g.name)"
            ]
            
            for constraint in constraints:
                session.run(constraint)
                logging.info(f"제약 조건 생성: {constraint}")
                
            for index in indexes:
                session.run(index)
                logging.info(f"인덱스 생성: {index}")

    def _create_curriculum_node(self):
        """교육과정 노드 생성"""
        with self.driver.session() as session:
            query = """
            MERGE (c:Curriculum {
              id: '2022-math',
              name: '2022 개정 교육과정 수학',
              description: '포용성과 창의성을 갖춘 주도적인 사람 양성을 위한 수학과 교육과정',
              year: 2022
            })
            RETURN c.id
            """
            result = session.run(query)
            curriculum_id = result.single()[0]
            logging.info(f"교육과정 노드 생성됨: {curriculum_id}")
            
            # 교과 역량 생성 및 연결
            competencies = [
                {"id": "C01", "name": "문제해결", "description": "수학적 지식을 이해하고 활용하여 다양한 문제를 해결할 수 있는 능력"},
                {"id": "C02", "name": "추론", "description": "수학적 현상을 탐구하여 논리적으로 추론하고 일반화할 수 있는 능력"},
                {"id": "C03", "name": "의사소통", "description": "수학적 아이디어를 다양한 방식으로 표현하고 다른 사람의 아이디어를 이해할 수 있는 능력"},
                {"id": "C04", "name": "연결", "description": "수학의 개념, 원리, 법칙 간의 관련성을 탐구하고 실생활이나 타 교과에 수학을 적용하여 수학의 유용성을 인식하는 능력"},
                {"id": "C05", "name": "정보처리", "description": "다양한 자료와 정보를 수집, 분석, 활용하고 적절한 공학적 도구를 선택하여 활용할 수 있는 능력"}
            ]
            
            for comp in competencies:
                query = """
                MERGE (comp:Competency {id: $id})
                ON CREATE SET 
                  comp.name = $name,
                  comp.description = $description
                WITH comp
                MATCH (c:Curriculum {id: '2022-math'})
                MERGE (c)-[:HAS_COMPETENCY]->(comp)
                RETURN comp.id
                """
                result = session.run(query, id=comp["id"], name=comp["name"], description=comp["description"])
                comp_id = result.single()[0]
                logging.info(f"교과 역량 생성 및 연결됨: {comp_id}")

    def _create_grade_groups(self):
        """학년군 노드 생성"""
        with self.driver.session() as session:
            grade_groups = [
                {"id": "1-2", "name": "초등학교 1~2학년"},
                {"id": "3-4", "name": "초등학교 3~4학년"},
                {"id": "5-6", "name": "초등학교 5~6학년"},
                {"id": "1-6", "name": "초등학교 1~6학년"},
                {"id": "7-9", "name": "중학교 1~3학년"}
            ]
            
            for gg in grade_groups:
                query = """
                MERGE (g:GradeGroup {id: $id})
                ON CREATE SET g.name = $name
                RETURN g.id
                """
                result = session.run(query, id=gg["id"], name=gg["name"])
                gg_id = result.single()[0]
                logging.info(f"학년군 생성됨: {gg_id}")

    def _create_content_categories(self):
        """내용 범주 노드 생성"""
        with self.driver.session() as session:
            categories = [
                {"id": "KU", "name": "지식·이해"},
                {"id": "PF", "name": "과정·기능"},
                {"id": "VA", "name": "가치·태도"}
            ]
            
            for cat in categories:
                query = """
                MERGE (cc:ContentCategory {id: $id})
                ON CREATE SET cc.name = $name
                RETURN cc.id
                """
                result = session.run(query, id=cat["id"], name=cat["name"])
                cat_id = result.single()[0]
                logging.info(f"내용 범주 생성됨: {cat_id}")

    def _create_domain_node(self, domain):
        """영역 노드 생성 및 교육과정과 연결"""
        with self.driver.session() as session:
            query = """
            MERGE (d:Domain {id: $id})
            ON CREATE SET 
              d.name = $name,
              d.code = $code,
              d.description = $description
            WITH d
            MATCH (c:Curriculum {id: '2022-math'})
            MERGE (c)-[:HAS_DOMAIN]->(d)
            RETURN d.id
            """
            result = session.run(
                query, 
                id=domain["id"],
                name=domain["name"],
                code=domain["code"],
                description=domain.get("description", "")
            )
            domain_id = result.single()[0]
            logging.info(f"영역 노드 생성 및 연결됨: {domain_id}")

    def _create_core_idea_node(self, core_idea, domain_id):
        """핵심 아이디어 노드 생성 및 영역과 연결"""
        with self.driver.session() as session:
            query = """
            MERGE (ci:CoreIdea {id: $id})
            ON CREATE SET ci.content = $content
            WITH ci
            MATCH (d:Domain {id: $domain_id})
            MERGE (d)-[:HAS_CORE_IDEA]->(ci)
            RETURN ci.id
            """
            result = session.run(
                query, 
                id=core_idea["id"],
                content=core_idea["content"],
                domain_id=domain_id
            )
            core_idea_id = result.single()[0]
            logging.info(f"핵심 아이디어 노드 생성 및 연결됨: {core_idea_id}")

    def _create_content_elements(self, content_elements):
        """내용 요소 노드 생성 및 관계 설정"""
        with self.driver.session() as session:
            for element in content_elements:
                query = """
                MERGE (ce:ContentElement {id: $id})
                ON CREATE SET 
                  ce.content = $content,
                  ce.category = $category
                WITH ce
                MATCH (d:Domain {id: $domain_id})
                MERGE (ce)-[:BELONGS_TO_DOMAIN]->(d)
                WITH ce
                MATCH (cc:ContentCategory {id: $category})
                MERGE (ce)-[:BELONGS_TO_CATEGORY]->(cc)
                WITH ce
                MATCH (g:GradeGroup {id: $grade_group_id})
                MERGE (ce)-[:FOR_GRADE_GROUP]->(g)
                RETURN ce.id
                """
                try:
                    result = session.run(
                        query, 
                        id=element["id"],
                        content=element["content"],
                        category=element.get("categoryId", ""),
                        domain_id=element["domainId"],
                        grade_group_id=element["gradeGroupId"]
                    )
                    element_id = result.single()[0]
                    logging.info(f"내용 요소 노드 생성 및 연결됨: {element_id}")
                except Exception as e:
                    logging.error(f"내용 요소 생성 중 오류: {str(e)}, 데이터: {element}")

    def _create_domain_relationships(self, relationships):
        """영역 관련 관계 설정"""
        with self.driver.session() as session:
            for rel in relationships:
                rel_type = rel["type"]
                source_id = rel["sourceNodeId"]
                target_id = rel["targetNodeId"]
                
                if rel_type == "HAS_CORE_IDEA":
                    # 이미 _create_core_idea_node에서 처리되므로 스킵
                    continue
                
                if rel_type == "APPLICABLE_TO":
                    query = """
                    MATCH (d:Domain {id: $source_id})
                    MATCH (g:GradeGroup {id: $target_id})
                    MERGE (d)-[:APPLICABLE_TO]->(g)
                    RETURN d.id, g.id
                    """
                    try:
                        result = session.run(query, source_id=source_id, target_id=target_id)
                        source, target = result.single()
                        logging.info(f"관계 생성됨: ({source})-[:APPLICABLE_TO]->({target})")
                    except Exception as e:
                        logging.error(f"관계 생성 중 오류: {str(e)}, 관계: {rel}")


def main():
    loader = None
    try:
        # 데이터 디렉토리 생성
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # JSON 파일 경로 확인
        for domain_id, file_path in DOMAIN_FILES.items():
            if not os.path.exists(file_path):
                logging.warning(f"영역 파일을 찾을 수 없음: {file_path}")
        
        # 데이터 로더 생성 및 실행
        loader = DomainLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        if loader.load_all_domains(DOMAIN_FILES):
            logging.info("영역 데이터 로딩 성공")
        else:
            logging.error("영역 데이터 로딩 실패")
        
    except Exception as e:
        logging.error(f"메인 프로세스 오류: {str(e)}")
        sys.exit(1)
    finally:
        if loader:
            loader.close()


if __name__ == "__main__":
    main()