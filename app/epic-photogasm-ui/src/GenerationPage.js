// src/GenerationPage.js
import React, { useState } from 'react';
import axios from 'axios';

// The base URL of your Python API
const API_URL = 'http://127.0.0.1:8000'; // <-- REMEMBER TO SET YOUR IP:PORT

function GenerationPage() {
    const [prompt, setPrompt] = useState('');
    const [negativePrompt, setNegativePrompt] = useState('blurry, low-res, text, watermark');
    
    const [batchSize, setBatchSize] = useState(1); // Default to 1 for video
    
    // ## NEW STATE for Video ##
    const [numFrames, setNumFrames] = useState(16);

    const [steps, setSteps] = useState(28);
    const [guidance, setGuidance] = useState(6.5);
    const [seed, setSeed] = useState('');

    const [loading, setLoading] = useState(false);
    const [generatedImages, setGeneratedImages] = useState([]); // State name is fine
    const [error, setError] = useState(null);

    const handleGenerate = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setGeneratedImages([]);

        const requestData = {
            prompt,
            negative_prompt: negativePrompt,
            batch_size: parseInt(batchSize, 10),
            steps: parseInt(steps, 10),
            guidance: parseFloat(guidance),
            num_frames: parseInt(numFrames, 10), // ## ADD num_frames ##
            seed: seed.trim() ? parseInt(seed, 10) : null,
            width: 512,
            height: 512,
        };

        try {
            const response = await axios.post(`${API_URL}/generate`, requestData);
            setGeneratedImages(response.data.images);
        } catch (err) {
            setError('An error occurred during video generation.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <h1>HunyuanVideo Generator</h1>
            <form onSubmit={handleGenerate}>
                <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="A cinematic video of..."
                    rows={4}
                    required
                />
                <textarea
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    placeholder="Negative prompt..."
                    rows={2}
                />
                
                <label>
                    Number of videos (1-4):
                    <input
                        type="number"
                        value={batchSize}
                        onChange={(e) => setBatchSize(e.target.value)}
                        min="1"
                        max="4" // Keep this low for video
                    />
                </label>
                
                {/* ## NEW INPUT FIELD for Video ## */}
                <label>
                    Number of frames (e.g., 16-32):
                    <input
                        type="number"
                        value={numFrames}
                        onChange={(e) => setNumFrames(e.target.value)}
                        min="8"
                        max="64"
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
                {generatedImages.map((video, index) => (
                    <div key={index} className="image-card">
                        {/* ## CHANGED from <img> to <video> ## */}
                        <video
                            src={`${API_URL}${video.url}`}
                            controls
                            loop
                            autoPlay
                            muted
                            alt={`Generated video with seed ${video.seed}`}
                        />
                        <p>Seed: {video.seed}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default GenerationPage;