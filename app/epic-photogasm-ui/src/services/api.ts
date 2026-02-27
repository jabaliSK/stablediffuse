import { GoogleGenAI } from '@google/genai';

export const getApiUrl = () => localStorage.getItem('API_URL') || 'mock';
export const setApiUrl = (url: string) => localStorage.setItem('API_URL', url);

export interface ImageMeta {
  id: string;
  url: string;
  prompt: string;
  seed: number;
  steps: number;
  cfg: number;
}

// In-memory store for mock mode
let mockGallery: ImageMeta[] = [];

export const generateImages = async (params: any): Promise<ImageMeta[]> => {
  const apiUrl = getApiUrl();
  
  if (apiUrl === 'mock' || !apiUrl) {
    // Use Gemini to mock the image generation
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      const batchSize = params.batch_size || 1;
      const results: ImageMeta[] = [];
      
      for (let i = 0; i < batchSize; i++) {
        const response = await ai.models.generateContent({
          model: 'gemini-2.5-flash-image',
          contents: params.prompt || 'A photorealistic portrait',
          config: {
            imageConfig: {
              aspectRatio: "1:1",
              imageSize: "1K"
            }
          }
        });
        
        let base64Image = '';
        for (const part of response.candidates?.[0]?.content?.parts || []) {
          if (part.inlineData) {
            base64Image = `data:image/png;base64,${part.inlineData.data}`;
            break;
          }
        }
        
        if (base64Image) {
          const newImg = {
            id: Math.random().toString(36).substring(7),
            url: base64Image,
            prompt: params.prompt,
            seed: params.seed || Math.floor(Math.random() * 1000000),
            steps: params.steps || 28,
            cfg: params.guidance || 6.5
          };
          results.push(newImg);
          mockGallery = [newImg, ...mockGallery];
        }
      }
      return results;
    } catch (e) {
      console.error("Mock generation failed", e);
      throw new Error("Failed to generate image using mock API.");
    }
  }

  // Real backend call
  const response = await fetch(`${apiUrl}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  
  if (!response.ok) throw new Error('Generation failed');
  const data = await response.json();
  // Assuming backend returns { images: [...] }
  return data.images.map((img: any) => ({
    ...img,
    id: Math.random().toString(36).substring(7),
    url: img.url.startsWith('http') ? img.url : `${apiUrl}${img.url}`
  }));
};

export const fetchGallery = async (): Promise<ImageMeta[]> => {
  const apiUrl = getApiUrl();
  
  if (apiUrl === 'mock' || !apiUrl) {
    return [...mockGallery];
  }

  const response = await fetch(`${apiUrl}/gallery`);
  if (!response.ok) throw new Error('Failed to fetch gallery');
  const data = await response.json();
  return data.map((img: any) => ({
    ...img,
    id: img.id || Math.random().toString(36).substring(7),
    url: img.url.startsWith('http') ? img.url : `${apiUrl}${img.url}`
  }));
};
