// src/GenerationPage.js
import React, { useState } from 'react';
import axios from 'axios';

// The base URL of your Python API
const API_URL = 'http://167.179.138.57:41106';

function GenerationPage() {
    const [prompt, setPrompt] = useState('');
    const [negativePrompt, setNegativePrompt] = useState('blurry, low-res, text, watermark');
    
    // ## NEW CONTROL STATE ##
    const [batchSize, setBatchSize] = useState(2); 

    const [steps, setSteps] = useState(28);
    const [guidance, setGuidance] = useState(6.5);
    const [seed, setSeed] = useState('');

    const [loading, setLoading] = useState(false);
    const [generatedImages, setGeneratedImages] = useState([]);
    const [error, setError] = useState(null);

    const handleGenerate = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setGeneratedImages([]);

        const requestData = {
            prompt,
            negative_prompt: negativePrompt,
            batch_size: parseInt(batchSize, 10), // This line already uses the state
            steps: parseInt(steps, 10),
            guidance: parseFloat(guidance),
            seed: seed.trim() ? parseInt(seed, 10) : null,
            width: 512,
            height: 512,
        };

        try {
            const response = await axios.post(`${API_URL}/generate`, requestData);
            setGeneratedImages(response.data.images);
        } catch (err) {
            setError('An error occurred during image generation.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <h1>epiCPhotoGasm Generator</h1>
            <form onSubmit={handleGenerate}>
                <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="A photorealistic portrait..."
                    rows={4}
                    required
                />
                <textarea
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    placeholder="Negative prompt..."
                    rows={2}
                />
                
                {/* ## NEW INPUT FIELD ## */}
                <label>
                    Number of images (1-8):
                    <input
                        type="number"
                        value={batchSize}
                        onChange={(e) => setBatchSize(e.target.value)}
                        min="1"
                        max="8" // Set a reasonable max to avoid overloading the backend
                    />
                </label>

                <input
                    type="text"
                    value={seed}
                    onChange={(e) => setSeed(e.target.value)}
                    placeholder="Seed (leave blank for random)"
                />
                <button type="submit" disabled={loading}>
                    {loading ? 'Generating...' : 'Generate'}
                </button>
            </form>

            {error && <p style={{ color: 'red' }}>{error}</p>}

            <div className="image-grid">
                {generatedImages.map((image, index) => (
                    <div key={index} className="image-card">
                        <img src={`${API_URL}${image.url}`} alt={`Generated image with seed ${image.seed}`} />
                        <p>Seed: {image.seed}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default GenerationPage;