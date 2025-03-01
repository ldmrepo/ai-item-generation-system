// 제약 조건 및 인덱스 생성
CREATE CONSTRAINT curriculum_id_constraint IF NOT EXISTS
FOR (c:Curriculum) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT domain_id_constraint IF NOT EXISTS
FOR (d:Domain) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT grade_group_id_constraint IF NOT EXISTS
FOR (g:GradeGroup) REQUIRE g.id IS UNIQUE;

CREATE CONSTRAINT achievement_standard_id_constraint IF NOT EXISTS
FOR (as:AchievementStandard) REQUIRE as.id IS UNIQUE;

CREATE CONSTRAINT achievement_level_id_constraint IF NOT EXISTS
FOR (al:AchievementLevel) REQUIRE al.id IS UNIQUE;

CREATE CONSTRAINT content_category_id_constraint IF NOT EXISTS
FOR (cc:ContentCategory) REQUIRE cc.id IS UNIQUE;

CREATE CONSTRAINT content_element_id_constraint IF NOT EXISTS
FOR (ce:ContentElement) REQUIRE ce.id IS UNIQUE;

CREATE CONSTRAINT evaluation_tool_id_constraint IF NOT EXISTS
FOR (et:EvaluationTool) REQUIRE et.id IS UNIQUE;

CREATE CONSTRAINT core_idea_id_constraint IF NOT EXISTS
FOR (ci:CoreIdea) REQUIRE ci.id IS UNIQUE;

CREATE CONSTRAINT competency_id_constraint IF NOT EXISTS
FOR (c:Competency) REQUIRE c.id IS UNIQUE;

// 인덱스 생성
CREATE INDEX domain_name_index IF NOT EXISTS FOR (d:Domain) ON (d.name);
CREATE INDEX achievement_standard_content_index IF NOT EXISTS FOR (as:AchievementStandard) ON (as.content);
CREATE INDEX content_element_content_index IF NOT EXISTS FOR (ce:ContentElement) ON (ce.content);
CREATE INDEX grade_group_name_index IF NOT EXISTS FOR (g:GradeGroup) ON (g.name);

// 교육과정 노드 생성
CREATE (c:Curriculum {
  id: '2022-math',
  name: '2022 개정 교육과정 수학',
  description: '포용성과 창의성을 갖춘 주도적인 사람 양성을 위한 수학과 교육과정',
  year: 2022
});

// 학년군 노드 생성
CREATE (g1:GradeGroup {id: '1-2', name: '초등학교 1~2학년'});
CREATE (g2:GradeGroup {id: '3-4', name: '초등학교 3~4학년'});
CREATE (g3:GradeGroup {id: '5-6', name: '초등학교 5~6학년'});
CREATE (g4:GradeGroup {id: '7-9', name: '중학교 1~3학년'});
CREATE (g5:GradeGroup {id: '10-12', name: '고등학교 1~3학년'});

// 내용 범주 노드 생성
CREATE (cc1:ContentCategory {id: 'KU', name: '지식·이해'});
CREATE (cc2:ContentCategory {id: 'PS', name: '문제 해결'});
CREATE (cc3:ContentCategory {id: 'CR', name: '추론'});
CREATE (cc4:ContentCategory {id: 'CO', name: '의사소통'});
CREATE (cc5:ContentCategory {id: 'AT', name: '태도 및 실천'});

// 성취수준 노드 생성
CREATE (al1:AchievementLevel {id: 'level-a', name: 'A', score_range: '90% 이상', description: '성취기준에 대한 이해와 수행이 매우 우수함'});
CREATE (al2:AchievementLevel {id: 'level-b', name: 'B', score_range: '80% 이상 90% 미만', description: '성취기준에 대한 이해와 수행이 우수함'});
CREATE (al3:AchievementLevel {id: 'level-c', name: 'C', score_range: '70% 이상 80% 미만', description: '성취기준에 대한 이해와 수행이 보통임'});
CREATE (al4:AchievementLevel {id: 'level-d', name: 'D', score_range: '60% 이상 70% 미만', description: '성취기준에 대한 이해와 수행이 미흡함'});
CREATE (al5:AchievementLevel {id: 'level-e', name: 'E', score_range: '60% 미만', description: '성취기준에 대한 이해와 수행이 매우 미흡함'});

// 교과 역량 노드 생성
CREATE (comp1:Competency {id: 'C01', name: '문제해결', description: '수학적 지식을 이해하고 활용하여 다양한 문제를 해결할 수 있는 능력'});
CREATE (comp2:Competency {id: 'C02', name: '추론', description: '수학적 현상을 탐구하여 논리적으로 추론하고 일반화할 수 있는 능력'});
CREATE (comp3:Competency {id: 'C03', name: '의사소통', description: '수학적 아이디어를 다양한 방식으로 표현하고 다른 사람의 아이디어를 이해할 수 있는 능력'});
CREATE (comp4:Competency {id: 'C04', name: '창의·융합', description: '수학의 지식과 기능을 다양한 상황에 창의적으로 활용하고 다른 교과와 융합할 수 있는 능력'});
CREATE (comp5:Competency {id: 'C05', name: '정보처리', description: '다양한 자료와 정보를 수집, 분석, 활용하고 적절한 공학적 도구를 선택하여 활용할 수 있는 능력'});
CREATE (comp6:Competency {id: 'C06', name: '태도 및 실천', description: '수학의 가치를 인식하고 자신감을 가지며 수학적 능력을 활용하여 사회에 공헌할 수 있는 능력'});

// 교육과정과 교과 역량 연결
MATCH (c:Curriculum {id: '2022-math'})
MATCH (comp:Competency)
CREATE (c)-[:HAS_COMPETENCY]->(comp);

// 영역 노드 생성 및 교육과정과 연결
MATCH (c:Curriculum {id: '2022-math'})
CREATE (d1:Domain {
  id: 'D01', 
  name: '수와 연산', 
  code: '01', 
  description: '수와 연산 영역은 초·중학교에서 다루는 수학적 대상과 기본적인 개념을 드러내는 영역으로...'
})
CREATE (c)-[:HAS_DOMAIN]->(d1)
CREATE (d2:Domain {
  id: 'D02', 
  name: '변화와 관계', 
  code: '02', 
  description: '변화와 관계 영역은 초·중학교에서 다루는 수학적 대상과 기본적인 개념을 드러내는 영역으로...'
})
CREATE (c)-[:HAS_DOMAIN]->(d2)
CREATE (d3:Domain {
  id: 'D03', 
  name: '공간과 도형', 
  code: '03', 
  description: '공간과 도형 영역은 초·중학교에서 다루는 도형의 개념과 성질, 공간 감각을 드러내는 영역으로...'
})
CREATE (c)-[:HAS_DOMAIN]->(d3)
CREATE (d4:Domain {
  id: 'D04', 
  name: '자료와 가능성', 
  code: '04', 
  description: '자료와 가능성 영역은 초·중학교에서 다루는 자료의 수집, 분석, 해석 및 확률과 통계적 개념을 드러내는 영역으로...'
})
CREATE (c)-[:HAS_DOMAIN]->(d4);

// 영역과 학년군 연결
MATCH (d:Domain)
MATCH (g:GradeGroup)
CREATE (d)-[:APPLICABLE_TO]->(g);

// 핵심 아이디어 노드 생성 및 영역과 연결
MATCH (d1:Domain {id: 'D01'})
CREATE (ci1:CoreIdea {
  id: 'CI0101', 
  content: '사물의 양은 자연수, 분수, 소수 등으로 표현되며, 연산은 이러한 수들 사이의 관계를 구성하고 다양한 맥락에서 활용된다.'
})
CREATE (d1)-[:HAS_CORE_IDEA]->(ci1);

MATCH (d2:Domain {id: 'D02'})
CREATE (ci2:CoreIdea {
  id: 'CI0201', 
  content: '규칙성은 반복되는 현상이나 사물의 배열에서 나타나며, 비와 비율은 서로 다른 두 양 사이의 관계를 나타내는 방법이다.'
})
CREATE (d2)-[:HAS_CORE_IDEA]->(ci2);

// 내용 요소 노드 생성 및 영역과 내용 범주에 연결
MATCH (d1:Domain {id: 'D01'})
MATCH (g4:GradeGroup {id: '7-9'})
MATCH (cc1:ContentCategory {id: 'KU'})
CREATE (ce1:ContentElement {
  id: 'KU010701', 
  content: '소인수분해',
  category: 'KU'
})
CREATE (ce1)-[:BELONGS_TO_DOMAIN]->(d1)
CREATE (ce1)-[:BELONGS_TO_CATEGORY]->(cc1)
CREATE (ce1)-[:FOR_GRADE_GROUP]->(g4);

// 성취기준 노드 생성 및 영역, 학년군과 연결
MATCH (d1:Domain {id: 'D01'})
MATCH (g4:GradeGroup {id: '7-9'})
MATCH (ce1:ContentElement {id: 'KU010701'})
CREATE (as1:AchievementStandard {
  id: '9수01-01',
  content: '소인수분해의 뜻을 알고, 자연수를 소인수분해 할 수 있다.',
  explanation: '소인수분해는 자연수를 소수의 곱으로 나타내는 것으로, 약수와 배수를 이해하는 데 기초가 된다.',
  considerations: '소인수분해 알고리즘의 효율성에 집중하기보다 개념 이해에 중점을 둔다.'
})
CREATE (as1)-[:BELONGS_TO_DOMAIN]->(d1)
CREATE (as1)-[:FOR_GRADE_GROUP]->(g4)
CREATE (as1)-[:RELATED_TO_CONTENT]->(ce1);

// 성취기준별 성취수준 노드 생성 및 연결
MATCH (as1:AchievementStandard {id: '9수01-01'})
MATCH (al1:AchievementLevel {id: 'level-a'})
CREATE (als1:AchievementLevelStandard {
  id: '9수01-01-A',
  standardId: '9수01-01',
  level: 'A',
  content: '소인수분해의 뜻을 설명하고, 자연수를 소인수분해 할 수 있다.'
})
CREATE (als1)-[:BASED_ON]->(as1)
CREATE (als1)-[:HAS_LEVEL]->(al1);

MATCH (as1:AchievementStandard {id: '9수01-01'})
MATCH (al3:AchievementLevel {id: 'level-c'})
CREATE (als2:AchievementLevelStandard {
  id: '9수01-01-C',
  standardId: '9수01-01',
  level: 'C',
  content: '소인수분해의 뜻을 알고, 자연수를 소인수의 곱으로 표현할 수 있다.'
})
CREATE (als2)-[:BASED_ON]->(as1)
CREATE (als2)-[:HAS_LEVEL]->(al3);

MATCH (as1:AchievementStandard {id: '9수01-01'})
MATCH (al5:AchievementLevel {id: 'level-e'})
CREATE (als3:AchievementLevelStandard {
  id: '9수01-01-E',
  standardId: '9수01-01',
  level: 'E',
  content: '소인수를 알고, 안내된 절차에 따라 자연수를 소인수의 곱으로 표현할 수 있다.'
})
CREATE (als3)-[:BASED_ON]->(as1)
CREATE (als3)-[:HAS_LEVEL]->(al5);

// 영역별 성취수준 노드 생성 및 연결
MATCH (d1:Domain {id: 'D01'})
MATCH (al1:AchievementLevel {id: 'level-a'})
MATCH (cc1:ContentCategory {id: 'KU'})
CREATE (dal1:DomainAchievementLevel {
  id: 'D01-A-KU',
  domainId: 'D01',
  level: 'A',
  category: 'KU',
  content: '∙ 소인수분해의 뜻을 설명할 수 있다. 양수와 음수, 정수와 유리수의 개념을 이해하고 설명할 수 있다.'
})
CREATE (dal1)-[:BELONGS_TO_DOMAIN]->(d1)
CREATE (dal1)-[:HAS_LEVEL]->(al1)
CREATE (dal1)-[:BELONGS_TO_CATEGORY]->(cc1);

MATCH (d1:Domain {id: 'D01'})
MATCH (al1:AchievementLevel {id: 'level-a'})
MATCH (cc2:ContentCategory {id: 'PS'})
CREATE (dal2:DomainAchievementLevel {
  id: 'D01-A-PS',
  domainId: 'D01',
  level: 'A',
  category: 'PS',
  content: '∙ 자연수를 소인수분해하여 최대공약수와 최소공배수를 구하고 그 원리를 설명할 수 있다.'
})
CREATE (dal2)-[:BELONGS_TO_DOMAIN]->(d1)
CREATE (dal2)-[:HAS_LEVEL]->(al1)
CREATE (dal2)-[:BELONGS_TO_CATEGORY]->(cc2);

// 함수 관련 성취기준 추가 (평가 도구 예시용)
MATCH (d2:Domain {id: 'D02'})
MATCH (g4:GradeGroup {id: '7-9'})
MATCH (cc1:ContentCategory {id: 'KU'})
CREATE (ce2:ContentElement {
  id: 'KU020701', 
  content: '함수의 개념',
  category: 'KU'
})
CREATE (ce2)-[:BELONGS_TO_DOMAIN]->(d2)
CREATE (ce2)-[:BELONGS_TO_CATEGORY]->(cc1)
CREATE (ce2)-[:FOR_GRADE_GROUP]->(g4);

MATCH (d2:Domain {id: 'D02'})
MATCH (g4:GradeGroup {id: '7-9'})
MATCH (ce2:ContentElement {id: 'KU020701'})
CREATE (as2:AchievementStandard {
  id: '9수02-14',
  content: '함수의 개념을 이해하고, 함숫값을 구할 수 있다.',
  explanation: '함수는 한 양이 다른 양에 대응하는 관계를 나타내는 것으로, 변화하는 현상을 표현하는 데 유용하다.',
  considerations: '일상생활이나 타 교과에서 함수 개념이 활용되는 예를 찾아보도록 한다.'
})
CREATE (as2)-[:BELONGS_TO_DOMAIN]->(d2)
CREATE (as2)-[:FOR_GRADE_GROUP]->(g4)
CREATE (as2)-[:RELATED_TO_CONTENT]->(ce2);

// 평가 도구 관련 노드 생성
MATCH (as2:AchievementStandard {id: '9수02-14'})
MATCH (al2:AchievementLevel {id: 'level-b'})
MATCH (d2:Domain {id: 'D02'})
MATCH (g4:GradeGroup {id: '7-9'})
CREATE (et1:EvaluationTool {
  id: 'funcEval-B1',
  title: '문항 개요_B수준 - 함수 개념 평가',
  itemType: '지필평가(선택형)',
  score: 1,
  correctAnswer: '②',
  evaluationFocus: '함수의 개념을 이해하고 다양한 상황에서 두 양 사이의 관계가 함수인지 판단'
})
CREATE (eg1:EvaluationGuidance {
  id: 'funcEval-B1-guidance',
  purpose: '함수의 개념을 이해하고 두 양 사이의 관계가 함수인지 판단할 수 있는지 평가하기 위함',
  considerations: '함수 개념을 고려하여 오답을 고른 학생에게는 표나 그래프 등 다양한 표현으로 함수를 재인식하고, 판단 근거를 설명하게 하여 보충 지도를 한다.'
})
CREATE (qc1:QuestionContent {
  id: 'funcEval-B1-content',
  content: '<보기>에서 y가 x의 함수인 것을 모두 고른 것은?\n\n<보기>\nㄱ. 자연수 x보다 5만큼 큰 수 y\nㄴ. x의 약수의 개수 y\nㄷ. 이차방정식 x²+xy+2=0의 해 y\nㄹ. |x|=y',
  options: 'ㄱ, ㄴ, ㄹ'
})
CREATE (sol1:Solution {
  id: 'funcEval-B1-solution',
  explanation: 'ㄱ. 자연수 x보다 5만큼 큰 수 y는 y=x+5로 나타낼 수 있으며, 각 x에 대해 하나의 y가 대응하므로 함수이다.\nㄴ. x의 약수의 개수 y는 각 x에 대해 유일한 y값이 결정되므로 함수이다.\nㄷ. 이차방정식 x²+xy+2=0의 해 y는 x에 따라 두 개의 해를 가질 수 있으므로 함수가 아니다.\nㄹ. |x|=y는 y=|x|로 나타낼 수 있으며, 각 x에 대해 하나의 y가 대응하므로 함수이다.',
  correctOptions: 'ㄱ, ㄴ, ㄹ'
})
CREATE (sc1:ScoringCriteria {
  id: 'funcEval-B1-scoring',
  fullScoreCriteria: '문항의 정답인 ②번을 선택한 경우',
  partialScoreCriteria: '없음',
  commonErrors: '이차방정식의 해가 여러 개일 수 있음을 인식하지 못하는 경우'
})
CREATE (et1)-[:BASED_ON]->(as2)
CREATE (et1)-[:TARGETS_LEVEL]->(al2)
CREATE (et1)-[:BELONGS_TO_DOMAIN]->(d2)
CREATE (et1)-[:FOR_GRADE_GROUP]->(g4)
CREATE (et1)-[:HAS_GUIDANCE]->(eg1)
CREATE (et1)-[:HAS_CONTENT]->(qc1)
CREATE (et1)-[:HAS_SOLUTION]->(sol1)
CREATE (et1)-[:HAS_SCORING_CRITERIA]->(sc1);

// 성취기준과 역량 연결
MATCH (as1:AchievementStandard {id: '9수01-01'})
MATCH (comp1:Competency {id: 'C01'})
CREATE (as1)-[:DEVELOPS_COMPETENCY]->(comp1);

MATCH (as2:AchievementStandard {id: '9수02-14'})
MATCH (comp2:Competency {id: 'C02'})
CREATE (as2)-[:DEVELOPS_COMPETENCY]->(comp2);