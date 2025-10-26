// src/App.js
import React, { useState } from 'react';
import GenerationPage from './GenerationPage';
import Img2ImgPage from './Img2ImgPage'; // We will create this
import GalleryPage from './GalleryPage';

function App() {
  const [page, setPage] = useState('txt2img'); // 'txt2img', 'img2img', 'gallery'

  const renderPage = () => {
    switch (page) {
      case 'txt2img':
        return <GenerationPage />;
      case 'img2img':
        return <Img2ImgPage />;
      case 'gallery':
        return <GalleryPage />;
      default:
        return <GenerationPage />;
    }
  };

  return (
    <div className="app-container">
      <nav className="app-nav">
        <button 
          onClick={() => setPage('txt2img')} 
          className={page === 'txt2img' ? 'active' : ''}
        >
          Text-to-Image
        </button>
        <button 
          onClick={() => setPage('img2img')} 
          className={page === 'img2img' ? 'active' : ''}
        >
          Image-to-Image
        </button>
        <button 
          onClick={() => setPage('gallery')} 
          className={page === 'gallery' ? 'active' : ''}
        >
          Gallery
        </button>
      </nav>
      <main className="app-content">
        {renderPage()}
      </main>
    </div>
  );
}

export default App;