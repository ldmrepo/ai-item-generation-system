// src/utils/apiClient.ts
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from "axios";

/**
 * API 클라이언트 클래스
 * 모든 API 요청을 중앙에서 관리하고 공통 설정을 적용
 */
class ApiClient {
    private client: AxiosInstance;

    constructor(baseURL: string) {
        this.client = axios.create({
            baseURL,
            headers: {
                "Content-Type": "application/json",
            },
        });

        // 요청 인터셉터 설정
        this.client.interceptors.request.use(
            (config) => {
                // 토큰이 있다면 헤더에 추가 (향후 인증 기능 구현시)
                const token = localStorage.getItem("authToken");
                if (token) {
                    config.headers.Authorization = `Bearer ${token}`;
                }
                return config;
            },
            (error) => {
                return Promise.reject(error);
            }
        );

        // 응답 인터셉터 설정
        this.client.interceptors.response.use(
            (response) => {
                return response;
            },
            (error) => {
                // 오류 처리 로직
                if (error.response) {
                    // 서버 응답이 있는 오류 (4xx, 5xx 상태 코드)
                    console.error("API Error:", error.response.data);

                    // 인증 오류 (401) 처리
                    if (error.response.status === 401) {
                        // 리디렉션 또는 로그아웃 처리
                        // window.location.href = '/login';
                    }
                } else if (error.request) {
                    // 요청은 보냈지만 응답을 받지 못한 경우
                    console.error("No response received:", error.request);
                } else {
                    // 요청 설정 중 오류 발생
                    console.error(
                        "Request configuration error:",
                        error.message
                    );
                }

                return Promise.reject(error);
            }
        );
    }

    /**
     * GET 요청
     * @param url 엔드포인트 URL
     * @param params 쿼리 파라미터
     * @param config 추가 설정
     * @returns Promise<AxiosResponse>
     */
    async get<T = any>(
        url: string,
        params?: any,
        config?: AxiosRequestConfig
    ): Promise<AxiosResponse<T>> {
        return this.client.get<T>(url, { ...config, params });
    }

    /**
     * POST 요청
     * @param url 엔드포인트 URL
     * @param data 요청 데이터
     * @param config 추가 설정
     * @returns Promise<AxiosResponse>
     */
    async post<T = any>(
        url: string,
        data?: any,
        config?: AxiosRequestConfig
    ): Promise<AxiosResponse<T>> {
        return this.client.post<T>(url, data, config);
    }

    /**
     * PUT 요청
     * @param url 엔드포인트 URL
     * @param data 요청 데이터
     * @param config 추가 설정
     * @returns Promise<AxiosResponse>
     */
    async put<T = any>(
        url: string,
        data?: any,
        config?: AxiosRequestConfig
    ): Promise<AxiosResponse<T>> {
        return this.client.put<T>(url, data, config);
    }

    /**
     * PATCH 요청
     * @param url 엔드포인트 URL
     * @param data 요청 데이터
     * @param config 추가 설정
     * @returns Promise<AxiosResponse>
     */
    async patch<T = any>(
        url: string,
        data?: any,
        config?: AxiosRequestConfig
    ): Promise<AxiosResponse<T>> {
        return this.client.patch<T>(url, data, config);
    }

    /**
     * DELETE 요청
     * @param url 엔드포인트 URL
     * @param config 추가 설정
     * @returns Promise<AxiosResponse>
     */
    async delete<T = any>(
        url: string,
        config?: AxiosRequestConfig
    ): Promise<AxiosResponse<T>> {
        return this.client.delete<T>(url, config);
    }
}

// API 기본 URL 설정
const API_BASE_URL =
    process.env.REACT_APP_API_URL || "http://localhost:3000/api/v1";

// 싱글톤 인스턴스 생성 및 내보내기
const apiClient = new ApiClient(API_BASE_URL);
export default apiClient;

// API 서비스 타입 예시
export interface KnowledgeMap {
    id: number;
    subject: string;
    grade: string;
    unit: string;
    version: string;
    creationDate: string;
    description?: string;
}

export interface ConceptNode {
    id: string;
    mapId: number;
    conceptName: string;
    definition: string;
    difficultyLevel: string;
    learningTime: number;
    createdAt: string;
    updatedAt: string;
}

export interface SeedItem {
    id: string;
    nodeId: string;
    itemType: string;
    difficulty: string;
    content: string;
    answer: string;
    explanation: string;
    variationPoints: string[];
    createdAt: string;
    updatedAt: string;
}

export interface GeneratedItem {
    id: string;
    nodeId: string;
    seedItemId?: string;
    itemType: string;
    difficulty: string;
    content: string;
    answer: string;
    explanation: string;
    approved: boolean;
    qualityScore?: number;
    generationTimestamp: string;
}

export interface GenerationRequest {
    request_id: string;
    status: "processing" | "completed" | "failed";
    nodes_requested: string[];
    total_items_requested: number;
    start_time: string;
    estimated_completion_time?: string;
}

// API 응답 타입 예시
export interface ApiResponse<T> {
    data: T;
    message?: string;
}

export interface PaginatedResponse<T> {
    data: T[];
    meta: {
        total: number;
        page: number;
        limit: number;
        pages: number;
    };
}

// API 서비스 클래스 예시
export class KnowledgeMapService {
    /**
     * 지식맵 목록 조회
     */
    static async getKnowledgeMaps(
        params?: any
    ): Promise<PaginatedResponse<KnowledgeMap>> {
        const response = await apiClient.get<PaginatedResponse<KnowledgeMap>>(
            "/knowledge-maps",
            params
        );
        return response.data;
    }

    /**
     * 지식맵 상세 조회
     */
    static async getKnowledgeMap(
        id: number
    ): Promise<ApiResponse<KnowledgeMap>> {
        const response = await apiClient.get<ApiResponse<KnowledgeMap>>(
            `/knowledge-maps/${id}`
        );
        return response.data;
    }

    /**
     * 지식맵 생성
     */
    static async createKnowledgeMap(
        data: Omit<KnowledgeMap, "id" | "creationDate">
    ): Promise<ApiResponse<KnowledgeMap>> {
        const response = await apiClient.post<ApiResponse<KnowledgeMap>>(
            "/knowledge-maps",
            data
        );
        return response.data;
    }

    /**
     * 지식맵 수정
     */
    static async updateKnowledgeMap(
        id: number,
        data: Partial<KnowledgeMap>
    ): Promise<ApiResponse<KnowledgeMap>> {
        const response = await apiClient.put<ApiResponse<KnowledgeMap>>(
            `/knowledge-maps/${id}`,
            data
        );
        return response.data;
    }

    /**
     * 지식맵 삭제
     */
    static async deleteKnowledgeMap(id: number): Promise<ApiResponse<void>> {
        const response = await apiClient.delete<ApiResponse<void>>(
            `/knowledge-maps/${id}`
        );
        return response.data;
    }
}

// 사용 예제
/*
async function fetchKnowledgeMaps() {
  try {
    const result = await KnowledgeMapService.getKnowledgeMaps({ subject: '수학', page: 1, limit: 10 });
    console.log('지식맵 목록:', result.data);
    console.log('총 개수:', result.meta.total);
  } catch (error) {
    console.error('지식맵 조회 실패:', error);
  }
}
*/
