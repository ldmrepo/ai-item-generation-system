#!/usr/bin/env python3
"""
성취기준별 성취수준과 영역별 성취수준 JSON 데이터를 Neo4j에 로딩하는 스크립트
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
        logging.FileHandler('achievement_level_loading.log')
    ]
)

# Neo4j 연결 정보
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "curriculum2022"

# 데이터 파일 경로
DATA_DIR = "data/curriculum"
ACHIEVEMENT_LEVEL_STANDARDS_FILE = os.path.join(DATA_DIR, "achievement_level_standards.json")
DOMAIN_ACHIEVEMENT_LEVELS_FILE = os.path.join(DATA_DIR, "domain_achievement_levels.json")

class AchievementLevelLoader:
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

    def load_achievement_level_standards(self, file_path):
        """성취기준별 성취수준 데이터 로딩"""
        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                logging.error(f"파일이 존재하지 않음: {file_path}")
                return False
            
            # JSON 파일 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            achievement_levels = data.get("achievementLevels", [])
            achievement_standards = data.get("achievementStandards", [])
            achievement_level_standards = data.get("achievementLevelStandards", [])
            
            # 데이터 로딩
            self._load_achievement_levels(achievement_levels)
            self._load_achievement_standards(achievement_standards)
            self._load_achievement_level_standards(achievement_level_standards)
            
            logging.info("성취기준별 성취수준 데이터 로딩 완료")
            return True
        
        except Exception as e:
            logging.error(f"성취기준별 성취수준 데이터 로딩 중 오류 발생: {str(e)}")
            return False

    def load_domain_achievement_levels(self, file_path):
        """영역별 성취수준 데이터 로딩"""
        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                logging.error(f"파일이 존재하지 않음: {file_path}")
                return False
            
            # JSON 파일 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            domain = data.get("domain", {})
            achievement_levels = data.get("achievementLevels", [])
            content_categories = data.get("contentCategories", [])
            domain_achievement_levels = data.get("domainAchievementLevels", [])
            
            # 도메인 데이터가 이미 로딩되어 있을 수 있으므로 MERGE 사용
            if domain:
                self._merge_domain(domain)
            
            # 성취수준 데이터가 이미 로딩되어 있을 수 있으므로 중복 로딩 방지
            self._load_achievement_levels(achievement_levels)
            
            # 내용 범주 로딩
            self._load_content_categories(content_categories)
            
            # 영역별 성취수준 로딩
            self._load_domain_achievement_levels(domain_achievement_levels)
            
            logging.info("영역별 성취수준 데이터 로딩 완료")
            return True
        
        except Exception as e:
            logging.error(f"영역별 성취수준 데이터 로딩 중 오류 발생: {str(e)}")
            return False

    def _load_achievement_levels(self, achievement_levels):
        """성취수준(A~E) 노드 생성"""
        with self.driver.session() as session:
            for level in achievement_levels:
                query = """
                MERGE (al:AchievementLevel {id: $id})
                ON CREATE SET
                  al.name = $name,
                  al.description = $description,
                  al.score_range = $score_range
                RETURN al.id
                """
                try:
                    result = session.run(
                        query,
                        id=level["id"],
                        name=level["name"],
                        description=level.get("description", ""),
                        score_range=level.get("score_range", "")
                    )
                    level_id = result.single()[0]
                    logging.info(f"성취수준 노드 생성됨: {level_id}")
                except Exception as e:
                    logging.error(f"성취수준 노드 생성 중 오류: {str(e)}, 데이터: {level}")

    def _load_achievement_standards(self, achievement_standards):
        """성취기준 노드 생성"""
        with self.driver.session() as session:
            for standard in achievement_standards:
                query = """
                MERGE (as:AchievementStandard {id: $id})
                ON CREATE SET
                  as.content = $content,
                  as.explanation = $explanation,
                  as.considerations = $considerations
                WITH as
                MATCH (d:Domain {id: $domainId})
                MERGE (as)-[:BELONGS_TO_DOMAIN]->(d)
                RETURN as.id
                """
                try:
                    result = session.run(
                        query,
                        id=standard["id"],
                        content=standard["content"],
                        explanation=standard.get("explanation", ""),
                        considerations=standard.get("considerations", ""),
                        domainId=standard["domainId"]
                    )
                    standard_id = result.single()[0]
                    logging.info(f"성취기준 노드 생성됨: {standard_id}")
                except Exception as e:
                    logging.error(f"성취기준 노드 생성 중 오류: {str(e)}, 데이터: {standard}")

    def _load_achievement_level_standards(self, achievement_level_standards):
        """성취기준별 성취수준 노드 생성"""
        with self.driver.session() as session:
            for level_standard in achievement_level_standards:
                query = """
                MERGE (als:AchievementLevelStandard {id: $id})
                ON CREATE SET
                  als.standardId = $standardId,
                  als.level = $level,
                  als.content = $content
                WITH als
                MATCH (as:AchievementStandard {id: $standardId})
                MERGE (als)-[:BASED_ON]->(as)
                WITH als
                MATCH (al:AchievementLevel {id: $levelId})
                MERGE (als)-[:HAS_LEVEL]->(al)
                RETURN als.id
                """
                try:
                    # 레벨 ID를 표준화 (예: "A" -> "level-a")
                    level = level_standard["level"]
                    level_id = f"level-{level.lower()}"
                    
                    result = session.run(
                        query,
                        id=level_standard["id"],
                        standardId=level_standard["standardId"],
                        level=level,
                        content=level_standard["content"],
                        levelId=level_id
                    )
                    level_standard_id = result.single()[0]
                    logging.info(f"성취기준별 성취수준 노드 생성됨: {level_standard_id}")
                except Exception as e:
                    logging.error(f"성취기준별 성취수준 노드 생성 중 오류: {str(e)}, 데이터: {level_standard}")

    def _merge_domain(self, domain):
        """영역 노드 생성 또는 병합"""
        with self.driver.session() as session:
            query = """
            MERGE (d:Domain {id: $id})
            ON CREATE SET
              d.name = $name,
              d.code = $code,
              d.description = $description
            RETURN d.id
            """
            try:
                result = session.run(
                    query,
                    id=domain["id"],
                    name=domain["name"],
                    code=domain["code"],
                    description=domain.get("description", "")
                )
                domain_id = result.single()[0]
                logging.info(f"영역 노드 생성/병합됨: {domain_id}")
            except Exception as e:
                logging.error(f"영역 노드 생성/병합 중 오류: {str(e)}, 데이터: {domain}")

    def _load_content_categories(self, content_categories):
        """내용 범주 노드 생성"""
        with self.driver.session() as session:
            for category in content_categories:
                query = """
                MERGE (cc:ContentCategory {id: $id})
                ON CREATE SET
                  cc.name = $name
                RETURN cc.id
                """
                try:
                    result = session.run(
                        query,
                        id=category["id"],
                        name=category["name"]
                    )
                    category_id = result.single()[0]
                    logging.info(f"내용 범주 노드 생성됨: {category_id}")
                except Exception as e:
                    logging.error(f"내용 범주 노드 생성 중 오류: {str(e)}, 데이터: {category}")

    def _load_domain_achievement_levels(self, domain_achievement_levels):
        """영역별 성취수준 노드 생성"""
        with self.driver.session() as session:
            for dal in domain_achievement_levels:
                query = """
                MERGE (dal:DomainAchievementLevel {id: $id})
                ON CREATE SET
                  dal.domainId = $domainId,
                  dal.level = $level,
                  dal.category = $category,
                  dal.content = $content
                WITH dal
                MATCH (d:Domain {id: $domainId})
                MERGE (dal)-[:BELONGS_TO_DOMAIN]->(d)
                WITH dal
                MATCH (al:AchievementLevel {id: $levelId})
                MERGE (dal)-[:HAS_LEVEL]->(al)
                WITH dal
                MATCH (cc:ContentCategory {id: $category})
                MERGE (dal)-[:BELONGS_TO_CATEGORY]->(cc)
                RETURN dal.id
                """
                try:
                    # 레벨 ID를 표준화
                    level = dal["level"]
                    level_id = f"level-{level.lower()}"
                    
                    result = session.run(
                        query,
                        id=dal["id"],
                        domainId=dal["domainId"],
                        level=level,
                        category=dal["category"],
                        content=dal["content"],
                        levelId=level_id
                    )
                    dal_id = result.single()[0]
                    logging.info(f"영역별 성취수준 노드 생성됨: {dal_id}")
                except Exception as e:
                    logging.error(f"영역별 성취수준 노드 생성 중 오류: {str(e)}, 데이터: {dal}")


def main():
    # 데이터 디렉토리 생성
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # JSON 파일 저장
    save_sample_data()
    
    # 데이터 로더 생성
    loader = None
    try:
        loader = AchievementLevelLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # 성취기준별 성취수준 로딩
        if loader.load_achievement_level_standards(ACHIEVEMENT_LEVEL_STANDARDS_FILE):
            logging.info("성취기준별 성취수준 데이터 로딩 성공")
        else:
            logging.error("성취기준별 성취수준 데이터 로딩 실패")
        
        # 영역별 성취수준 로딩
        if loader.load_domain_achievement_levels(DOMAIN_ACHIEVEMENT_LEVELS_FILE):
            logging.info("영역별 성취수준 데이터 로딩 성공")
        else:
            logging.error("영역별 성취수준 데이터 로딩 실패")
        
    except Exception as e:
        logging.error(f"메인 프로세스 오류: {str(e)}")
    finally:
        if loader:
            loader.close()


def save_sample_data():
    """샘플 JSON 파일 저장"""
    try:
        # 성취기준별 성취수준 데이터
        achievement_level_standards_data = {
            "achievementLevels": [
                {
                    "id": "level-a",
                    "name": "A",
                    "description": "매우 우수함",
                    "score_range": "90% 이상"
                },
                # 다른 성취수준 정보...
            ],
            "achievementStandards": [
                {
                    "id": "9수01-01",
                    "content": "소인수분해의 뜻을 알고, 자연수를 소인수분해할 수 있다",
                    "explanation": "",
                    "considerations": "",
                    "domainId": "D01"
                },
                # 다른 성취기준 정보...
            ],
            "achievementLevelStandards": [
                {
                    "id": "9수01-01-A",
                    "standardId": "9수01-01",
                    "level": "A",
                    "content": "소인수분해의 뜻을 설명하고, 자연수를 소인수분해 할 수 있다."
                },
                # 다른 성취기준별 성취수준 정보...
            ]
        }
        
        # 영역별 성취수준 데이터
        domain_achievement_levels_data = {
            "domain": {
                "id": "D01",
                "name": "수와 연산",
                "code": "01",
                "description": "수와 연산 영역은 초·중학교에서 다루는 수학적 대상과 기본적인 개념을 드러내는 영역으로..."
            },
            "achievementLevels": [
                {
                    "id": "level-a",
                    "name": "A",
                    "description": "매우 우수함",
                    "score_range": "90% 이상"
                },
                # 다른 성취수준 정보...
            ],
            "contentCategories": [
                {
                    "id": "KU",
                    "name": "지식·이해"
                },
                # 다른 내용 범주 정보...
            ],
            "domainAchievementLevels": [
                {
                    "id": "D01-A-KU",
                    "domainId": "D01",
                    "level": "A",
                    "category": "KU",
                    "content": "소인수분해의 뜻을 설명할 수 있다. 양수와 음수, 정수와 유리수의 개념을 이해하고, 정수와 유리수의 사칙계산의 원리를 설명할 수 있다..."
                },
                # 다른 영역별 성취수준 정보...
            ]
        }
        
        # JSON 파일 저장
        with open(ACHIEVEMENT_LEVEL_STANDARDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(achievement_level_standards_data, f, ensure_ascii=False, indent=2)
        
        with open(DOMAIN_ACHIEVEMENT_LEVELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(domain_achievement_levels_data, f, ensure_ascii=False, indent=2)
        
        logging.info("샘플 JSON 파일이 저장되었습니다.")
    except Exception as e:
        logging.error(f"샘플 JSON 파일 저장 중 오류: {str(e)}")


if __name__ == "__main__":
    main()