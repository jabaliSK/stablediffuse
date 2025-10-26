// src/Img2ImgPage.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';

// IMPORTANT: Replace with your actual API URL
const API_URL = 'IP:PORT';

function Img2ImgPage() {
    // Form States
    const [prompt, setPrompt] = useState('');
    const [negativePrompt, setNegativePrompt] = useState('blurry, low-res, text, watermark');
    const [strength, setStrength] = useState(0.7);
    const [steps, setSteps] = useState(40);
    const [guidance, setGuidance] = useState(7.0);
    const [batchSize, setBatchSize] = useState(2); 
    const [seed, setSeed] = useState('');
    
    // Image File States
    const [initImage, setInitImage] = useState(null);
    const [initImagePreview, setInitImagePreview] = useState(null);

    // API States
    const [loading, setLoading] = useState(false);
    const [generatedImages, setGeneratedImages] = useState([]);
    const [error, setError] = useState(null);

    // Handle file input changes and create a preview
    const handleImageChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setInitImage(file);
            setInitImagePreview(URL.createObjectURL(file));
        }
    };

    // Clean up the object URL to prevent memory leaks
    useEffect(() => {
        return () => {
            if (initImagePreview) {
                URL.revokeObjectURL(initImagePreview);
            }
        };
    }, [initImagePreview]);

    const handleGenerate = async (e) => {
        e.preventDefault();
        
        if (!initImage) {
            setError('Please upload an initial image.');
            return;
        }

        setLoading(true);
        setError(null);
        setGeneratedImages([]);

        // We MUST use FormData because we are uploading a file
        const formData = new FormData();
        formData.append('image', initImage);
        formData.append('prompt', prompt);
        formData.append('negative_prompt', negativePrompt);
        formData.append('strength', strength);
        formData.append('steps', parseInt(steps, 10));
        formData.append('guidance', parseFloat(guidance));
        formData.append('batch_size', parseInt(batchSize, 10));
        formData.append('seed', seed.trim() ? parseInt(seed, 10) : null);
        formData.append('width', 512); // Or make these form inputs
        formData.append('height', 512);

        try {
            const response = await axios.post(`${API_URL}/img2img`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setGeneratedImages(response.data.images);
        } catch (err) {
            setError('An error occurred during image generation.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="page-container">
            <h1>Image-to-Image Generator</h1>
            <div className="form-and-preview-container">
                <form onSubmit={handleGenerate} className="generation-form">
                    <textarea
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="A person wearing a sweater..."
                        rows={3}
                        required
                    />
                    <textarea
                        value={negativePrompt}
                        onChange={(e) => setNegativePrompt(e.target.value)}
                        placeholder="Negative prompt..."
                        rows={2}
                    />
                    
                    <div className="form-row">
                        <label>
                            Upload Image:
                            <input
                                type="file"
                                accept="image/png, image/jpeg"
                                onChange={handleImageChange}
                                required
                            />
                        </label>
                    </div>

                    <div className="form-row slider-container">
                        <label>Strength: {strength}</label>
                        <input
                            type="range"
                            min="0.1"
                            max="1.0"
                            step="0.05"
                            value={strength}
                            onChange={(e) => setStrength(e.target.value)}
                        />
                    </div>

                    <div className="form-row slider-container">
                        <label>Steps: {steps}</label>
                        <input
                            type="range"
                            min="10"
                            max="100"
                            step="1"
                            value={steps}
                            onChange={(e) => setSteps(e.target.value)}
                        />
                    </div>
                    
                    <div className="form-row slider-container">
                        <label>Guidance (CFG): {guidance}</label>
                        <input
                            type="range"
                            min="1"
                            max="20"
                            step="0.5"
                            value={guidance}
                            onChange={(e) => setGuidance(e.target.value)}
                        />
                    </div>

                    <div className="form-row">
                        <label>
                            Number of images (1-8):
                            <input
                                type="number"
                                value={batchSize}
                                onChange={(e) => setBatchSize(e.target.value)}
                                min="1"
                                max="8"
                            />
                        </label>
                        <label>
                            Seed:
                            <input
                                type="text"
                                value={seed}
                                onChange={(e) => setSeed(e.target.value)}
                                placeholder="(blank for random)"
                            />
                        </label>
                    </div>

                    <button type="submit" disabled={loading || !initImage}>
                        {loading ? 'Generating...' : 'Generate'}
                    </button>
                </form>

                <div className="image-preview-container">
                    <h3>Input Image Preview</h3>
                    {initImagePreview ? (
                        <img src={initImagePreview} alt="Input preview" />
                    ) : (
                        <div className="image-placeholder">
                            <p>Your uploaded image will appear here.</p>
                        </div>
                    )}
                </div>
            </div>

            {error && <p className="error-message">{error}</p>}

            <h2>Results</h2>
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

export default Img2ImgPage;