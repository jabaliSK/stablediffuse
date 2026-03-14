// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';

// ## ADD THIS LINE ##
import './App.css'; 

import GenerationPage from './GenerationPage';
import GalleryPage from './GalleryPage';


function App() {
  return (
    <Router>
      <div className="App">
        <nav className="app-nav">
          <Link to="/">Generator</Link>
          <Link to="/gallery">Past Images</Link>
        </nav>
        <main className="app-content">
          <Routes>
            <Route path="/" element={<GenerationPage />} />
            <Route path="/gallery" element={<GalleryPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;