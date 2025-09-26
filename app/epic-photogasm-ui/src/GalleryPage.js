// src/GalleryPage.js
import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_URL = 'http://167.179.138.57:41106';

function GalleryPage() {
    const [images, setImages] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // ## NEW STATE: Track the index of the currently selected image ##
    // `null` means the modal is closed.
    const [selectedIndex, setSelectedIndex] = useState(null);
    const [touchStart, setTouchStart] = useState(null); // For swipe detection

    // Fetch gallery data when the component loads
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

    // ## NEW HANDLERS: For navigation ##
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

    // ## NEW EFFECT: Add keyboard navigation (left/right arrows) ##
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
    
    // ## NEW HANDLERS: For touch/swipe gestures ##
    const handleTouchStart = (e) => {
        setTouchStart(e.targetTouches[0].clientX);
    };
    
    const handleTouchMove = (e) => {
        if (touchStart === null) return;
        const touchEnd = e.targetTouches[0].clientX;
        const diff = touchStart - touchEnd;

        // Swipe left (next image)
        if (diff > 50) {
            goToNext();
            setTouchStart(null); // Reset after swipe
        }

        // Swipe right (previous image)
        if (diff < -50) {
            goToPrevious();
            setTouchStart(null); // Reset after swipe
        }
    };


    if (loading) return <p>Loading gallery...</p>;
    if (error) return <p style={{ color: 'red' }}>{error}</p>;
    if (images.length === 0) return <p>No images found. Generate some first!</p>;

    return (
        <div>
            <h1>Past Images</h1>
            {/* --- Thumbnail Grid --- */}
            <div className="gallery-grid">
                {images.map((imgMeta, index) => (
                    <div key={index} className="gallery-item" onClick={() => openModal(index)}>
                        <img src={`${API_URL}${imgMeta.url}`} alt={imgMeta.prompt} />
                        <div className="gallery-info">
                            <p><strong>Seed:</strong> {imgMeta.seed}</p>
                        </div>
                    </div>
                ))}
            </div>

            {/* --- ## NEW: Modal/Lightbox View ## --- */}
            {selectedIndex !== null && (
                <div className="modal-overlay" onClick={closeModal}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()} onTouchStart={handleTouchStart} onTouchMove={handleTouchMove}>
                        <button className="modal-close" onClick={closeModal}>&times;</button>
                        <button className="modal-nav prev" onClick={goToPrevious}>&#10094;</button>
                        
                        <img src={`${API_URL}${images[selectedIndex].url}`} alt={images[selectedIndex].prompt} />
                        
                        <button className="modal-nav next" onClick={goToNext}>&#10095;</button>
                        
                        <div className="modal-caption">
                           <p><strong>Prompt:</strong> {images[selectedIndex].prompt}</p>
                           <p><strong>Seed:</strong> {images[selectedIndex].seed} | <strong>Steps:</strong> {images[selectedIndex].steps} | <strong>CFG:</strong> {images[selectedIndex].cfg}</p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default GalleryPage;