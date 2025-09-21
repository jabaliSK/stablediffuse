// src/GalleryPage.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

function GalleryPage() {
    const [images, setImages] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        // Fetch gallery data when the component loads
        const fetchGallery = async () => {
            try {
                const response = await axios.get(`${API_URL}/gallery`);
                setImages(response.data);
            } catch (err) {
                setError('Failed to load gallery.');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchGallery();
    }, []); // The empty array [] means this effect runs once on mount

    if (loading) return <p>Loading gallery...</p>;
    if (error) return <p style={{ color: 'red' }}>{error}</p>;
    if (images.length === 0) return <p>No images found. Generate some first!</p>;

    return (
        <div>
            <h1>Past Images</h1>
            <div className="gallery-grid">
                {images.map((imgMeta, index) => (
                    <div key={index} className="gallery-item">
                        <img src={`${API_URL}${imgMeta.url}`} alt={imgMeta.prompt} />
                        <div className="gallery-info">
                            <p><strong>Prompt:</strong> {imgMeta.prompt.substring(0, 100)}...</p>
                            <p><strong>Seed:</strong> {imgMeta.seed}</p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default GalleryPage;