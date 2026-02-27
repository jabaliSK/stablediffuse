import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Loader2 } from 'lucide-react';
import { fetchGallery, ImageMeta } from '../services/api';
import Carousel from '../components/Carousel';

export default function GalleryPage() {
  const [images, setImages] = useState<ImageMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  const loadGallery = async () => {
    try {
      const data = await fetchGallery();
      setImages(data);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to load gallery');
    } finally {
      setLoading(false);
    }
  };

  // Initial load & polling
  useEffect(() => {
    loadGallery();
    const interval = setInterval(loadGallery, 5000); // Poll every 5s
    return () => clearInterval(interval);
  }, []);

  if (loading && images.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
      </div>
    );
  }

  if (error && images.length === 0) {
    return (
      <div className="flex items-center justify-center h-full p-6 text-center">
        <p className="text-red-400 text-sm">{error}</p>
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="flex items-center justify-center h-full p-6 text-center">
        <p className="text-zinc-500 text-sm">No images yet. Go generate some!</p>
      </div>
    );
  }

  return (
    <>
      <div className="p-1 pb-24">
        {/* 3x3 Grid */}
        <div className="grid grid-cols-3 gap-1">
          {images.map((img, index) => (
            <motion.div
              key={img.id || index}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="aspect-square bg-zinc-900 relative cursor-pointer overflow-hidden"
              onClick={() => setSelectedIndex(index)}
            >
              <img 
                src={img.url} 
                alt={img.prompt} 
                className="w-full h-full object-cover hover:scale-105 transition-transform duration-500"
                loading="lazy"
              />
            </motion.div>
          ))}
        </div>
      </div>

      <AnimatePresence>
        {selectedIndex !== null && (
          <Carousel
            images={images}
            initialIndex={selectedIndex}
            onClose={() => setSelectedIndex(null)}
          />
        )}
      </AnimatePresence>
    </>
  );
}
