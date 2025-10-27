// src/GalleryPage.js
import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000'; // <-- REMEMBER TO SET YOUR IP:PORT

function GalleryPage() {
    const [images, setImages] = useState([]); // State name is fine
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const [selectedIndex, setSelectedIndex] = useState(null);
    const [touchStart, setTouchStart] = useState(null);

    useEffect(() => {
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
            if (e.key === 'ArrowRight') {
                goToNext();
            } else if (e.key === 'ArrowLeft') {
                goToPrevious();
            } else if (e.key === 'Escape') {
                closeModal();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [selectedIndex, goToNext, goToPrevious]);
    
    const handleTouchStart = (e) => {
        setTouchStart(e.targetTouches[0].clientX);
    };
    
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
    if (error) return <p style={{ color: 'red' }}>{error}</p>;
    if (images.length === 0) return <p>No videos found. Generate some first!</p>;

    return (
        <div>
            <h1>Past Videos</h1>
            {/* --- Thumbnail Grid --- */}
            <div className="gallery-grid">
                {images.map((videoMeta, index) => (
                    <div key={index} className="gallery-item" onClick={() => openModal(index)}>
                        {/* ## CHANGED from <img> to <video> ## */}
                        <video
                            src={`${API_URL}${videoMeta.url}`}
                            alt={videoMeta.prompt}
                            loop
                            autoPlay
                            muted
                        />
                        <div className="gallery-info">
                            <p><strong>Seed:</strong> {videoMeta.seed}</p>
                        </div>
                    </div>
                ))}
            </div>

            {/* --- Modal/Lightbox View --- */}
            {selectedIndex !== null && (
                <div className="modal-overlay" onClick={closeModal}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()} onTouchStart={handleTouchStart} onTouchMove={handleTouchMove}>
                        <button className="modal-close" onClick={closeModal}>&times;</button>
                        <button className="modal-nav prev" onClick={goToPrevious}>&#10094;</button>
                        
                        {/* ## CHANGED from <img> to <video> ## */}
                        <video
                            src={`${API_URL}${images[selectedIndex].url}`}
                            alt={images[selectedIndex].prompt}
                            controls
                            loop
                            autoPlay
                        />
                        
                        <button className="modal-nav next" onClick={goToNext}>&#10095;</button>
                        
                        <div className="modal-caption">
                           <p><strong>Prompt:</strong> {images[selectedIndex].prompt}</p>
                           {/* ## ADDED num_frames ## */}
                           <p>
                                <strong>Seed:</strong> {images[selectedIndex].seed} | 
                                <strong>Steps:</strong> {images[selectedIndex].steps} | 
                                <strong>CFG:</strong> {images[selectedIndex].cfg} |
                                <strong>Frames:</strong> {images[selectedIndex].num_frames}
                           </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default GalleryPage;