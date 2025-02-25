-- 지식맵 메타정보
CREATE TABLE knowledge_map (
  id SERIAL PRIMARY KEY,
  subject VARCHAR(50) NOT NULL,
  grade VARCHAR(50) NOT NULL,
  unit VARCHAR(100) NOT NULL,
  version VARCHAR(20) NOT NULL,
  creation_date DATE NOT NULL,
  description TEXT
);

-- 개념 노드
CREATE TABLE concept_node (
  id VARCHAR(20) PRIMARY KEY, -- 예: FUN-001
  map_id INTEGER NOT NULL REFERENCES knowledge_map(id),
  concept_name VARCHAR(100) NOT NULL,
  definition TEXT NOT NULL,
  difficulty_level VARCHAR(20) NOT NULL,
  learning_time INTEGER NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(map_id, concept_name)
);

-- 노드 교육과정 매핑
CREATE TABLE curriculum_mapping (
  id SERIAL PRIMARY KEY,
  node_id VARCHAR(20) NOT NULL REFERENCES concept_node(id),
  education_system VARCHAR(100) NOT NULL,
  grade VARCHAR(50) NOT NULL,
  unit VARCHAR(100) NOT NULL,
  achievement_standard TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 노드 간 연결 관계
CREATE TABLE node_connection (
  id SERIAL PRIMARY KEY,
  source_node_id VARCHAR(20) NOT NULL REFERENCES concept_node(id),
  target_node_id VARCHAR(20) NOT NULL REFERENCES concept_node(id),
  connection_type VARCHAR(20) NOT NULL, -- prerequisite, successor, related
  strength VARCHAR(20) NOT NULL, -- strong, medium, weak
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(source_node_id, target_node_id)
);

-- 오개념 정보
CREATE TABLE misconception (
  id SERIAL PRIMARY KEY,
  node_id VARCHAR(20) NOT NULL REFERENCES concept_node(id),
  description TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 씨드 문항
CREATE TABLE seed_item (
  id VARCHAR(50) PRIMARY KEY, -- 예: SEED-FUN-001-001
  node_id VARCHAR(20) NOT NULL REFERENCES concept_node(id),
  item_type VARCHAR(50) NOT NULL, -- conceptual, calculation, graphing 등
  difficulty VARCHAR(20) NOT NULL, -- basic, intermediate, advanced
  content TEXT NOT NULL,
  answer TEXT NOT NULL,
  explanation TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 변형 요소
CREATE TABLE variation_point (
  id SERIAL PRIMARY KEY,
  seed_item_id VARCHAR(50) NOT NULL REFERENCES seed_item(id),
  point_name VARCHAR(100) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(seed_item_id, point_name)
);

-- 생성된 문항
CREATE TABLE generated_item (
  id VARCHAR(50) PRIMARY KEY, -- 예: GEN-FUN-001-001
  seed_item_id VARCHAR(50) REFERENCES seed_item(id),
  node_id VARCHAR(20) NOT NULL REFERENCES concept_node(id),
  item_type VARCHAR(50) NOT NULL,
  difficulty VARCHAR(20) NOT NULL,
  content TEXT NOT NULL,
  answer TEXT NOT NULL,
  explanation TEXT NOT NULL,
  generation_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  approved BOOLEAN DEFAULT FALSE,
  quality_score DECIMAL(3,2)
);

-- 문항 생성 이력
CREATE TABLE generation_history (
  id SERIAL PRIMARY KEY,
  generation_request_id VARCHAR(50) NOT NULL, -- 외부에서 생성된 요청 ID
  user_id VARCHAR(50), -- 요청한 사용자 ID (있을 경우)
  request_params JSONB NOT NULL, -- 요청 파라미터
  items_requested INTEGER NOT NULL,
  items_generated INTEGER NOT NULL,
  start_time TIMESTAMP NOT NULL,
  end_time TIMESTAMP,
  status VARCHAR(20) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 생성 요청과 생성된 문항 매핑
CREATE TABLE generation_item_mapping (
  id SERIAL PRIMARY KEY,
  generation_history_id INTEGER NOT NULL REFERENCES generation_history(id),
  generated_item_id VARCHAR(50) NOT NULL REFERENCES generated_item(id),
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(generation_history_id, generated_item_id)
);

-- 평가 기준
CREATE TABLE assessment_criteria (
  id SERIAL PRIMARY KEY,
  name VARCHAR(50) NOT NULL UNIQUE,
  description TEXT NOT NULL,
  weight DECIMAL(3,2) NOT NULL,
  evaluation_method VARCHAR(100) NOT NULL,
  passing_threshold DECIMAL(3,2) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 문항 평가 결과
CREATE TABLE item_assessment (
  id SERIAL PRIMARY KEY,
  generated_item_id VARCHAR(50) NOT NULL REFERENCES generated_item(id),
  criteria_id INTEGER NOT NULL REFERENCES assessment_criteria(id),
  score DECIMAL(3,2) NOT NULL,
  feedback TEXT,
  assessed_by VARCHAR(50),
  assessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(generated_item_id, criteria_id)
);

-- 인덱스 생성
CREATE INDEX idx_concept_node_map_id ON concept_node(map_id);
CREATE INDEX idx_seed_item_node_id ON seed_item(node_id);
CREATE INDEX idx_generated_item_node_id ON generated_item(node_id);
CREATE INDEX idx_generated_item_seed_item_id ON generated_item(seed_item_id);
CREATE INDEX idx_misconception_node_id ON misconception(node_id);
CREATE INDEX idx_node_connection_source ON node_connection(source_node_id);
CREATE INDEX idx_node_connection_target ON node_connection(target_node_id);