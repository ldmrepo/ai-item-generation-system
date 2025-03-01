#!/usr/bin/env python3

"""
이제 평가 도구 데이터를 Neo4j 데이터베이스에 로딩하는 스크립트를 작성해 드리겠습니다.

평가 도구 데이터를 Neo4j 데이터베이스에 로딩하는 스크립트를 작성했습니다. 이 스크립트는 앞서 추출한 JSON 형식의 평가 도구 데이터를 로드하여 Neo4j 그래프 데이터베이스에 저장합니다.

## 스크립트 주요 기능

1. **제약 조건 생성**
   - EvaluationTool, EvaluationGuidance, QuestionContent, Solution, ScoringCriteria 노드에 대한 ID 고유성 제약 조건 설정

2. **노드 생성**
   - 성취기준(AchievementStandard) 노드 생성
   - 성취기준별 성취수준(AchievementLevelStandard) 노드 생성
   - 평가 도구(EvaluationTool) 노드 생성
   - 평가 지도 방안(EvaluationGuidance) 노드 생성
   - 문항 내용(QuestionContent) 노드 생성
   - 답안 해설(Solution) 노드 생성
   - 채점 기준(ScoringCriteria) 노드 생성

3. **관계 생성**
   - BASED_ON: 평가 도구와 성취기준 연결
   - TARGETS_LEVEL: 평가 도구와 성취수준 연결
   - BELONGS_TO_DOMAIN: 평가 도구와 영역 연결
   - FOR_GRADE_GROUP: 평가 도구와 학년군 연결
   - HAS_GUIDANCE: 평가 도구와 평가 지도 방안 연결
   - HAS_CONTENT: 평가 도구와 문항 내용 연결
   - HAS_SOLUTION: 평가 도구와 답안 해설 연결
   - HAS_SCORING_CRITERIA: 평가 도구와 채점 기준 연결

## 사용 방법

1. 먼저 평가 도구 JSON 데이터를 `data/curriculum/평가도구데이타.json` 경로에 저장합니다.

2. Neo4j 데이터베이스가 실행 중인지 확인합니다:
   ```bash
   docker ps | grep neo4j-curriculum
   ```

3. 필요한 Python 패키지를 설치합니다:
   ```bash
   pip install neo4j
   ```

4. 스크립트를 실행합니다:
   ```bash
   python evaluation-tool-loading-script.py
   ```

이 스크립트를 실행하면 평가 도구 데이터가 Neo4j 데이터베이스에 로드됩니다. 로드된 데이터는 교사들이 적절한 평가 도구를 검색하고 활용할 수 있는 교육과정 메타데이터 시스템의 일부로 활용될 수 있습니다.
"""

"""
2022 개정 교육과정 수학과 평가 도구 JSON 데이터를 Neo4j에 로딩하는 스크립트
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
        logging.FileHandler('evaluation_tool_loading.log')
    ]
)

# Neo4j 연결 정보
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "curriculum2022"

# 데이터 파일 경로
DATA_DIR = "data/curriculum"
EVALUATION_TOOLS_FILE = os.path.join(DATA_DIR, "평가도구데이타.json")

class EvaluationToolLoader:
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

    def load_evaluation_tools(self, file_path):
        """평가 도구 데이터 로딩"""
        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                logging.error(f"파일이 존재하지 않음: {file_path}")
                return False
            
            # JSON 파일 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 제약 조건 생성
            self._create_constraints()
            
            # 데이터 로딩
            evaluation_tool = data.get("evaluationTool", {})
            achievement_standard = data.get("achievement_standard", {})
            achievement_level_standards = data.get("achievement_level_standards", [])
            evaluation_guidance = data.get("evaluationGuidance", {})
            question_content = data.get("questionContent", {})
            solution = data.get("solution", {})
            scoring_criteria = data.get("scoringCriteria", {})
            relationships = data.get("relationships", [])
            
            # 데이터 로딩
            if achievement_standard:
                self._create_achievement_standard(achievement_standard)
            
            if achievement_level_standards:
                self._create_achievement_level_standards(achievement_level_standards)
            
            if evaluation_tool:
                self._create_evaluation_tool(evaluation_tool)
            
            if evaluation_guidance:
                self._create_evaluation_guidance(evaluation_guidance)
            
            if question_content:
                self._create_question_content(question_content)
            
            if solution:
                self._create_solution(solution)
            
            if scoring_criteria:
                self._create_scoring_criteria(scoring_criteria)
            
            if relationships:
                self._create_relationships(relationships)
            
            logging.info("평가 도구 데이터 로딩 완료")
            return True
        
        except Exception as e:
            logging.error(f"평가 도구 데이터 로딩 중 오류 발생: {str(e)}")
            return False

    def _create_constraints(self):
        """제약 조건 생성"""
        with self.driver.session() as session:
            # 제약 조건 생성
            constraints = [
                "CREATE CONSTRAINT evaluation_tool_id_constraint IF NOT EXISTS FOR (et:EvaluationTool) REQUIRE et.id IS UNIQUE",
                "CREATE CONSTRAINT evaluation_guidance_id_constraint IF NOT EXISTS FOR (eg:EvaluationGuidance) REQUIRE eg.id IS UNIQUE",
                "CREATE CONSTRAINT question_content_id_constraint IF NOT EXISTS FOR (qc:QuestionContent) REQUIRE qc.id IS UNIQUE",
                "CREATE CONSTRAINT solution_id_constraint IF NOT EXISTS FOR (sol:Solution) REQUIRE sol.id IS UNIQUE",
                "CREATE CONSTRAINT scoring_criteria_id_constraint IF NOT EXISTS FOR (sc:ScoringCriteria) REQUIRE sc.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logging.info(f"제약 조건 생성: {constraint}")
                except Exception as e:
                    logging.warning(f"제약 조건 생성 중 오류 (이미 존재할 수 있음): {str(e)}")

    def _create_achievement_standard(self, achievement_standard):
        """성취기준 노드 생성"""
        with self.driver.session() as session:
            query = """
            MERGE (as:AchievementStandard {id: $id})
            ON CREATE SET
              as.content = $content
            WITH as
            MATCH (d:Domain {id: $domainId})
            MERGE (as)-[:BELONGS_TO_DOMAIN]->(d)
            RETURN as.id
            """
            
            try:
                result = session.run(
                    query,
                    id=achievement_standard["id"],
                    content=achievement_standard["content"],
                    domainId=achievement_standard["domainId"]
                )
                standard_id = result.single()[0]
                logging.info(f"성취기준 노드 생성됨: {standard_id}")
            except Exception as e:
                logging.error(f"성취기준 노드 생성 중 오류: {str(e)}")

    def _create_achievement_level_standards(self, achievement_level_standards):
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
                    # 레벨 ID 변환 (예: "A" -> "level-a")
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
                    logging.error(f"성취기준별 성취수준 노드 생성 중 오류: {str(e)}")

    def _create_evaluation_tool(self, evaluation_tool):
        """평가 도구 노드 생성"""
        with self.driver.session() as session:
            query = """
            MERGE (et:EvaluationTool {id: $id})
            ON CREATE SET
              et.title = $title,
              et.schoolLevel = $schoolLevel,
              et.subject = $subject,
              et.gradeGroup = $gradeGroup,
              et.domainName = $domainName,
              et.standardId = $standardId,
              et.evaluationFocus = $evaluationFocus,
              et.itemType = $itemType,
              et.score = $score,
              et.correctAnswer = $correctAnswer
            RETURN et.id
            """
            
            try:
                result = session.run(
                    query,
                    id=evaluation_tool["id"],
                    title=evaluation_tool["title"],
                    schoolLevel=evaluation_tool["schoolLevel"],
                    subject=evaluation_tool["subject"],
                    gradeGroup=evaluation_tool["gradeGroup"],
                    domainName=evaluation_tool["domainName"],
                    standardId=evaluation_tool["standardId"],
                    evaluationFocus=evaluation_tool["evaluationFocus"],
                    itemType=evaluation_tool["itemType"],
                    score=evaluation_tool["score"],
                    correctAnswer=evaluation_tool["correctAnswer"]
                )
                tool_id = result.single()[0]
                logging.info(f"평가 도구 노드 생성됨: {tool_id}")
            except Exception as e:
                logging.error(f"평가 도구 노드 생성 중 오류: {str(e)}")

    def _create_evaluation_guidance(self, evaluation_guidance):
        """평가 지도 방안 노드 생성"""
        with self.driver.session() as session:
            query = """
            MERGE (eg:EvaluationGuidance {id: $id})
            ON CREATE SET
              eg.evaluationToolId = $evaluationToolId,
              eg.purpose = $purpose,
              eg.considerations = $considerations
            RETURN eg.id
            """
            
            try:
                result = session.run(
                    query,
                    id=evaluation_guidance["id"],
                    evaluationToolId=evaluation_guidance["evaluationToolId"],
                    purpose=evaluation_guidance["purpose"],
                    considerations=evaluation_guidance["considerations"]
                )
                guidance_id = result.single()[0]
                logging.info(f"평가 지도 방안 노드 생성됨: {guidance_id}")
            except Exception as e:
                logging.error(f"평가 지도 방안 노드 생성 중 오류: {str(e)}")

    def _create_question_content(self, question_content):
        """문항 내용 노드 생성"""
        with self.driver.session() as session:
            query = """
            MERGE (qc:QuestionContent {id: $id})
            ON CREATE SET
              qc.evaluationToolId = $evaluationToolId,
              qc.content = $content,
              qc.options = $options
            RETURN qc.id
            """
            
            try:
                result = session.run(
                    query,
                    id=question_content["id"],
                    evaluationToolId=question_content["evaluationToolId"],
                    content=question_content["content"],
                    options=question_content["options"]
                )
                content_id = result.single()[0]
                logging.info(f"문항 내용 노드 생성됨: {content_id}")
            except Exception as e:
                logging.error(f"문항 내용 노드 생성 중 오류: {str(e)}")

    def _create_solution(self, solution):
        """답안 해설 노드 생성"""
        with self.driver.session() as session:
            query = """
            MERGE (sol:Solution {id: $id})
            ON CREATE SET
              sol.evaluationToolId = $evaluationToolId,
              sol.explanation = $explanation,
              sol.correctOptions = $correctOptions
            RETURN sol.id
            """
            
            try:
                result = session.run(
                    query,
                    id=solution["id"],
                    evaluationToolId=solution["evaluationToolId"],
                    explanation=solution["explanation"],
                    correctOptions=solution["correctOptions"]
                )
                solution_id = result.single()[0]
                logging.info(f"답안 해설 노드 생성됨: {solution_id}")
            except Exception as e:
                logging.error(f"답안 해설 노드 생성 중 오류: {str(e)}")

    def _create_scoring_criteria(self, scoring_criteria):
        """채점 기준 노드 생성"""
        with self.driver.session() as session:
            query = """
            MERGE (sc:ScoringCriteria {id: $id})
            ON CREATE SET
              sc.evaluationToolId = $evaluationToolId,
              sc.fullScoreCriteria = $fullScoreCriteria,
              sc.partialScoreCriteria = $partialScoreCriteria,
              sc.commonErrors = $commonErrors
            RETURN sc.id
            """
            
            try:
                result = session.run(
                    query,
                    id=scoring_criteria["id"],
                    evaluationToolId=scoring_criteria["evaluationToolId"],
                    fullScoreCriteria=scoring_criteria["fullScoreCriteria"],
                    partialScoreCriteria=scoring_criteria["partialScoreCriteria"],
                    commonErrors=scoring_criteria["commonErrors"]
                )
                criteria_id = result.single()[0]
                logging.info(f"채점 기준 노드 생성됨: {criteria_id}")
            except Exception as e:
                logging.error(f"채점 기준 노드 생성 중 오류: {str(e)}")

    def _create_relationships(self, relationships):
        """관계 생성"""
        with self.driver.session() as session:
            for rel in relationships:
                rel_type = rel["type"]
                source_id = rel["sourceNodeId"]
                target_id = rel["targetNodeId"]
                
                if rel_type == "BASED_ON":
                    query = """
                    MATCH (et:EvaluationTool {id: $source_id})
                    MATCH (as:AchievementStandard {id: $target_id})
                    MERGE (et)-[:BASED_ON]->(as)
                    RETURN et.id, as.id
                    """
                
                elif rel_type == "TARGETS_LEVEL":
                    query = """
                    MATCH (et:EvaluationTool {id: $source_id})
                    MATCH (al:AchievementLevel {id: $target_id})
                    MERGE (et)-[:TARGETS_LEVEL]->(al)
                    RETURN et.id, al.id
                    """
                
                elif rel_type == "BELONGS_TO_DOMAIN":
                    query = """
                    MATCH (et:EvaluationTool {id: $source_id})
                    MATCH (d:Domain {id: $target_id})
                    MERGE (et)-[:BELONGS_TO_DOMAIN]->(d)
                    RETURN et.id, d.id
                    """
                
                elif rel_type == "FOR_GRADE_GROUP":
                    query = """
                    MATCH (et:EvaluationTool {id: $source_id})
                    MATCH (g:GradeGroup {id: $target_id})
                    MERGE (et)-[:FOR_GRADE_GROUP]->(g)
                    RETURN et.id, g.id
                    """
                
                elif rel_type == "HAS_GUIDANCE":
                    query = """
                    MATCH (et:EvaluationTool {id: $source_id})
                    MATCH (eg:EvaluationGuidance {id: $target_id})
                    MERGE (et)-[:HAS_GUIDANCE]->(eg)
                    RETURN et.id, eg.id
                    """
                
                elif rel_type == "HAS_CONTENT":
                    query = """
                    MATCH (et:EvaluationTool {id: $source_id})
                    MATCH (qc:QuestionContent {id: $target_id})
                    MERGE (et)-[:HAS_CONTENT]->(qc)
                    RETURN et.id, qc.id
                    """
                
                elif rel_type == "HAS_SOLUTION":
                    query = """
                    MATCH (et:EvaluationTool {id: $source_id})
                    MATCH (sol:Solution {id: $target_id})
                    MERGE (et)-[:HAS_SOLUTION]->(sol)
                    RETURN et.id, sol.id
                    """
                
                elif rel_type == "HAS_SCORING_CRITERIA":
                    query = """
                    MATCH (et:EvaluationTool {id: $source_id})
                    MATCH (sc:ScoringCriteria {id: $target_id})
                    MERGE (et)-[:HAS_SCORING_CRITERIA]->(sc)
                    RETURN et.id, sc.id
                    """
                else:
                    logging.warning(f"알 수 없는 관계 유형: {rel_type}, 건너뜁니다.")
                    continue
                
                try:
                    result = session.run(query, source_id=source_id, target_id=target_id)
                    source, target = result.single()
                    logging.info(f"관계 생성됨: ({source})-[:{rel_type}]->({target})")
                except Exception as e:
                    logging.error(f"관계 생성 중 오류: {str(e)}, 관계: {rel}")


def main():
    loader = None
    try:
        # 데이터 디렉토리 생성
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # 데이터 로더 생성 및 실행
        loader = EvaluationToolLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        if loader.load_evaluation_tools(EVALUATION_TOOLS_FILE):
            logging.info("평가 도구 데이터 로딩 성공")
        else:
            logging.error("평가 도구 데이터 로딩 실패")
        
    except Exception as e:
        logging.error(f"메인 프로세스 오류: {str(e)}")
        sys.exit(1)
    finally:
        if loader:
            loader.close()


if __name__ == "__main__":
    main()