// src/GalleryPage.js
import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

// IMPORTANT: Replace with your actual API URL
const API_URL = 'IP:PORT';

function GalleryPage() {
    const [images, setImages] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selectedIndex, setSelectedIndex] = useState(null);
    const [touchStart, setTouchStart] = useState(null);

    useEffect(() => {
        const fetchGallery = async () => {
            try {
                // Set cache-busting param to get fresh data
                const response = await axios.get(`${API_URL}/gallery?t=${new Date().getTime()}`);
                setImages(response.data);
            } catch (err) {
                setError('Failed to load gallery.');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchGallery();
    }, []);

    const openModal = (index) => setSelectedIndex(index);
    const closeModal = () => setSelectedIndex(null);

    const goToNext = useCallback(() => {
        if (selectedIndex === null) return;
        setSelectedIndex((prevIndex) => (prevIndex + 1) % images.length);
    }, [selectedIndex, images.length]);

    const goToPrevious = useCallback(() => {
        if (selectedIndex === null) return;
        setSelectedIndex((prevIndex) => (prevIndex - 1 + images.length) % images.length);
    }, [selectedIndex, images.length]);

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (selectedIndex === null) return;
            if (e.key === 'ArrowRight') goToNext();
            else if (e.key === 'ArrowLeft') goToPrevious();
            else if (e.key === 'Escape') closeModal();
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [selectedIndex, goToNext, goToPrevious]);
    
    const handleTouchStart = (e) => setTouchStart(e.targetTouches[0].clientX);
    
    const handleTouchMove = (e) => {
        if (touchStart === null) return;
        const touchEnd = e.targetTouches[0].clientX;
        const diff = touchStart - touchEnd;
        if (diff > 50) {
            goToNext();
            setTouchStart(null);
        }
        if (diff < -50) {
            goToPrevious();
            setTouchStart(null);
        }
    };

    if (loading) return <p>Loading gallery...</p>;
    if (error) return <p className="error-message">{error}</p>;
    if (images.length === 0) return <p>No images found. Generate some first!</p>;

    // Get the currently selected image's metadata
    const currentImage = selectedIndex !== null ? images[selectedIndex] : null;

    return (
        <div className="page-container">
            <h1>Past Images</h1>
            <div className="gallery-grid">
                {images.map((imgMeta, index) => (
                    <div key={index} className="gallery-item" onClick={() => openModal(index)}>
                        <img src={`${API_URL}${imgMeta.url}`} alt={imgMeta.prompt} />
                        <div className="gallery-info">
                            <p><strong>Seed:</strong> {imgMeta.seed}</p>
                            {/* --- NEW: Add a badge for Img2Img --- */}
                            {imgMeta.type === 'img2img' && (
                                <span className="img2img-badge">Img2Img</span>
                            )}
                        </div>
                    </div>
                ))}
            </div>

            {/* --- MODIFIED: Modal View --- */}
            {currentImage && (
                <div className="modal-overlay" onClick={closeModal}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()} onTouchStart={handleTouchStart} onTouchMove={handleTouchMove}>
                        <button className="modal-close" onClick={closeModal}>&times;</button>
                        <button className="modal-nav prev" onClick={goToPrevious}>&#10094;</button>
                        
                        {/* --- NEW: Image comparison container --- */}
                        <div className="modal-image-container">
                            {/* --- NEW: Show Input Image if it exists --- */}
                            {currentImage.input_image_url && (
                                <div className="modal-image-wrapper">
                                    <img 
                                        src={`${API_URL}${currentImage.input_image_url}`} 
                                        alt="Input" 
                                        className="modal-image"
                                    />
                                    <p className="image-label">Input</p>
                                </div>
                            )}

                            {/* Generated Image (Original) */}
                            <div className="modal-image-wrapper">
                                <img 
                                    src={`${API_URL}${currentImage.url}`} 
                                    alt={currentImage.prompt}
                                    className="modal-image"
                                />
                                <p className="image-label">Output</p>
                            </div>
                        </div>
                        
                        <button className="modal-nav next" onClick={goToNext}>&#10095;</button>
                        
                        {/* --- MODIFIED: Caption --- */}
                        <div className="modal-caption">
                           <p><strong>Prompt:</strong> {currentImage.prompt}</p>
                           <p>
                                <strong>Seed:</strong> {currentImage.seed} | 
                                <strong> Steps:</strong> {currentImage.steps} | 
                                <strong> CFG:</strong> {currentImage.cfg}
                                {/* --- NEW: Show Strength if it exists --- */}
                                {currentImage.strength && (
                                    <span> | <strong> Strength:</strong> {currentImage.strength}</span>
                                )}
                           </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default GalleryPage;