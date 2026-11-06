import { API_URL } from '../config';

export interface ImageMeta {
  id: string;
  url: string;
  prompt: string;
  seed: number;
  steps: number;
  cfg: number;
}

export const generateImages = async (params: any): Promise<ImageMeta[]> => {
  const response = await fetch(`${API_URL}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const detailMsg = errorData.detail 
      ? (typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail)) 
      : `HTTP ${response.status}: Generation failed`;
    throw new Error(detailMsg);
  }
  const data = await response.json();
  
  return data.images.map((img: any) => ({
    ...img,
    id: img.id || Math.random().toString(36).substring(7),
    url: img.url.startsWith('http') ? img.url : `${API_URL}${img.url}`,
    prompt: params.prompt,
    steps: params.steps,
    cfg: params.guidance
  }));
};

export interface ContinuousStatus {
  active: boolean;
  prompt: string;
  total_generated: number;
  last_image?: string | null;
  started_at?: string | null;
}

export const startContinuous = async (params: any): Promise<{ status: string; stats: ContinuousStatus }> => {
  const response = await fetch(`${API_URL}/continuous/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to start continuous generation');
  }
  return await response.json();
};

export const stopContinuous = async (): Promise<{ status: string; stats: ContinuousStatus }> => {
  const response = await fetch(`${API_URL}/continuous/stop`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to stop continuous generation');
  return await response.json();
};

export const getContinuousStatus = async (): Promise<ContinuousStatus> => {
  const response = await fetch(`${API_URL}/continuous/status`);
  if (!response.ok) throw new Error('Failed to fetch continuous status');
  return await response.json();
};

export const fetchGallery = async (): Promise<ImageMeta[]> => {
  const response = await fetch(`${API_URL}/gallery`);
  if (!response.ok) throw new Error('Failed to fetch gallery');
  const data = await response.json();
  
  return data.map((img: any) => ({
    ...img,
    id: img.id || Math.random().toString(36).substring(7),
    url: img.url.startsWith('http') ? img.url : `${API_URL}${img.url}`,
    steps: img.params?.steps || img.steps || 28,
    cfg: img.params?.guidance || img.cfg || img.guidance || 6.5,
    prompt: img.prompt || img.params?.prompt || ''
  }));
};