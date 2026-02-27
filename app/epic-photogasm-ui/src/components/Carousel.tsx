import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, ChevronLeft, ChevronRight, Info } from 'lucide-react';
import { ImageMeta } from '../services/api';

interface CarouselProps {
  images: ImageMeta[];
  initialIndex: number;
  onClose: () => void;
}

const swipeConfidenceThreshold = 10000;
const swipePower = (offset: number, velocity: number) => {
  return Math.abs(offset) * velocity;
};

export default function Carousel({ images, initialIndex, onClose }: CarouselProps) {
  const [[page, direction], setPage] = useState([initialIndex, 0]);
  const [showInfo, setShowInfo] = useState(false);
  const [isZoomed, setIsZoomed] = useState(false);
  const [lastTap, setLastTap] = useState(0);

  // Wrap around index
  const imageIndex = ((page % images.length) + images.length) % images.length;

  const paginate = useCallback((newDirection: number) => {
    setPage([page + newDirection, newDirection]);
    setIsZoomed(false);
  }, [page]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight') paginate(1);
      else if (e.key === 'ArrowLeft') paginate(-1);
      else if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [paginate, onClose]);

  const handleTap = () => {
    const now = Date.now();
    if (now - lastTap < 300) {
      setIsZoomed(!isZoomed);
    }
    setLastTap(now);
  };

  const currentImage = images[imageIndex];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 bg-black flex items-center justify-center"
    >
      {/* Top Bar */}
      <div className="absolute top-0 inset-x-0 p-4 flex items-center justify-between z-50 bg-gradient-to-b from-black/80 to-transparent pointer-events-none">
        <div className="text-zinc-400 text-sm font-medium pointer-events-auto">
          {imageIndex + 1} / {images.length}
        </div>
        <div className="flex items-center gap-4 pointer-events-auto">
          <button 
            onClick={() => setShowInfo(!showInfo)}
            className={`p-2 rounded-full transition-colors ${showInfo ? 'bg-white/20 text-white' : 'bg-white/10 text-zinc-300 hover:bg-white/20'}`}
          >
            <Info className="w-5 h-5" />
          </button>
          <button 
            onClick={onClose}
            className="p-2 bg-white/10 hover:bg-white/20 text-zinc-300 rounded-full transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Image Container */}
      <div 
        className="relative w-full h-full flex items-center justify-center overflow-hidden"
        onClick={handleTap}
      >
        <AnimatePresence initial={false} custom={direction}>
          <motion.img
            key={page}
            src={currentImage.url}
            custom={direction}
            variants={{
              enter: (direction: number) => ({
                x: direction > 0 ? 1000 : -1000,
                opacity: 0,
                scale: 1,
                y: 0
              }),
              center: {
                zIndex: 1,
                x: 0,
                y: 0,
                opacity: 1,
                scale: isZoomed ? 2.5 : 1
              },
              exit: (direction: number) => ({
                zIndex: 0,
                x: direction < 0 ? 1000 : -1000,
                opacity: 0,
                scale: 1,
                y: 0
              })
            }}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{
              x: { type: "spring", stiffness: 300, damping: 30 },
              y: { type: "spring", stiffness: 300, damping: 30 },
              opacity: { duration: 0.2 },
              scale: { type: "spring", stiffness: 300, damping: 30 }
            }}
            drag={isZoomed ? true : "x"}
            dragConstraints={isZoomed ? { left: -500, right: 500, top: -500, bottom: 500 } : { left: 0, right: 0 }}
            dragElastic={isZoomed ? 0.2 : 1}
            onDragEnd={(e, { offset, velocity }) => {
              if (isZoomed) return;
              const swipe = swipePower(offset.x, velocity.x);
              if (swipe < -swipeConfidenceThreshold) {
                paginate(1);
              } else if (swipe > swipeConfidenceThreshold) {
                paginate(-1);
              }
            }}
            className={`absolute max-w-full max-h-full object-contain ${isZoomed ? 'cursor-grab active:cursor-grabbing' : 'cursor-default'}`}
            alt={currentImage.prompt}
            onClick={(e) => e.stopPropagation()} // Prevent double tap on container when dragging
            onPointerDown={handleTap} // Use pointer down for better touch response on the image itself
          />
        </AnimatePresence>
      </div>

      {/* Navigation Buttons (Desktop) */}
      <div className="hidden md:flex absolute inset-y-0 inset-x-4 items-center justify-between pointer-events-none z-40">
        <button 
          onClick={() => paginate(-1)}
          className="p-3 bg-black/50 hover:bg-black/80 text-white rounded-full pointer-events-auto backdrop-blur-md transition-all"
        >
          <ChevronLeft className="w-6 h-6" />
        </button>
        <button 
          onClick={() => paginate(1)}
          className="p-3 bg-black/50 hover:bg-black/80 text-white rounded-full pointer-events-auto backdrop-blur-md transition-all"
        >
          <ChevronRight className="w-6 h-6" />
        </button>
      </div>

      {/* Info Overlay */}
      <AnimatePresence>
        {showInfo && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="absolute bottom-0 inset-x-0 p-6 bg-gradient-to-t from-black via-black/80 to-transparent z-40 pointer-events-none"
          >
            <div className="max-w-xl mx-auto space-y-3 pointer-events-auto">
              <p className="text-white text-sm leading-relaxed">
                {currentImage.prompt}
              </p>
              <div className="flex flex-wrap gap-3 text-xs font-mono text-zinc-400">
                <span className="bg-white/10 px-2 py-1 rounded">Seed: {currentImage.seed}</span>
                <span className="bg-white/10 px-2 py-1 rounded">Steps: {currentImage.steps}</span>
                <span className="bg-white/10 px-2 py-1 rounded">CFG: {currentImage.cfg}</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
