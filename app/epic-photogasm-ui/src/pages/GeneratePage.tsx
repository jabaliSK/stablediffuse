import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'motion/react';
import { Wand2, Loader2, Square } from 'lucide-react';
import { generateImages, ImageMeta } from '../services/api';

export default function GeneratePage() {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('blurry, low-res, text, watermark');
  const [batchSize, setBatchSize] = useState<number | ''>(1);
  const [seed, setSeed] = useState('');
  const [resolution, setResolution] = useState('512x896');
  const [continuous, setContinuous] = useState(false);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ImageMeta[]>([]);
  
  const stopRequested = useRef(false);
  const continuousRef = useRef(continuous);

  useEffect(() => {
    continuousRef.current = continuous;
  }, [continuous]);

  const handleGenerate = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!prompt.trim()) return;
    
    if (batchSize === '' || batchSize < 1) {
      setError('Please enter a valid number of images.');
      return;
    }
    
    setLoading(true);
    setError(null);
    stopRequested.current = false;
    
    try {
      const [widthStr, heightStr] = resolution.split('x');
      const width = parseInt(widthStr, 10);
      const height = parseInt(heightStr, 10);

      do {
        const images = await generateImages({
          prompt,
          negative_prompt: negativePrompt,
          batch_size: batchSize,
          steps: 28,
          guidance: 6.5,
          width,
          height,
          seed: seed ? parseInt(seed, 10) : undefined
        });
        setResults(prev => [...images, ...prev]);
        
        if (!continuousRef.current || stopRequested.current) break;
      } while (true);
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleStop = () => {
    stopRequested.current = true;
    setContinuous(false);
  };

  return (
    <div className="p-6 pb-24 max-w-md mx-auto">
      <form onSubmit={handleGenerate} className="space-y-6">
        {/* Prompt */}
        <div className="space-y-2">
          <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Prompt</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="A photorealistic portrait of..."
            className="w-full min-h-[44px] bg-zinc-900/50 border border-white/10 rounded-2xl p-4 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all resize-none h-32"
            required
          />
        </div>

        {/* Negative Prompt */}
        <div className="space-y-2">
          <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Negative Prompt</label>
          <textarea
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            className="w-full min-h-[44px] bg-zinc-900/50 border border-white/10 rounded-2xl p-4 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all resize-none h-20"
          />
        </div>

        {/* Controls Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Images</label>
            <input
              type="number"
              min="1"
              max="16"
              inputMode="numeric"
              pattern="[0-9]*"
              value={batchSize}
              onChange={(e) => {
                const val = e.target.value;
                if (val === '') setBatchSize('');
                else {
                  const num = parseInt(val, 10);
                  if (!isNaN(num)) setBatchSize(num);
                }
              }}
              className="w-full min-h-[44px] bg-zinc-900/50 border border-white/10 rounded-xl p-3 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
            />
          </div>
          <div className="space-y-2">
            <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Seed</label>
            <input
              type="text"
              inputMode="numeric"
              pattern="[0-9]*"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="Random"
              className="w-full min-h-[44px] bg-zinc-900/50 border border-white/10 rounded-xl p-3 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
            />
          </div>
          <div className="space-y-2 col-span-2">
            <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Resolution</label>
            <select
              value={resolution}
              onChange={(e) => setResolution(e.target.value)}
              className="w-full min-h-[44px] bg-zinc-900/50 border border-white/10 rounded-xl p-3 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all [&>option]:bg-zinc-900"
            >
              <option value="512x896">512 × 896 (9:16)</option>
              <option value="768x1344">768 × 1344 (9:16 HD)</option>
              <option value="768x1024">768 × 1024 (3:4)</option>
              <option value="512x768">512 × 768 (2:3)</option>
            </select>
          </div>
        </div>

        {/* Continuous Generation Checkbox */}
        <label className="flex items-center gap-3 p-3 bg-zinc-900/30 border border-white/5 rounded-xl cursor-pointer hover:bg-zinc-900/50 transition-colors">
          <div className="relative flex items-center justify-center">
            <input
              type="checkbox"
              checked={continuous}
              onChange={(e) => setContinuous(e.target.checked)}
              className="peer sr-only"
            />
            <div className="w-5 h-5 border-2 border-zinc-500 rounded flex items-center justify-center peer-checked:border-indigo-500 peer-checked:bg-indigo-500 transition-all">
              <motion.div
                initial={false}
                animate={{ scale: continuous ? 1 : 0 }}
                className="w-2.5 h-2.5 bg-white rounded-sm"
              />
            </div>
          </div>
          <span className="text-sm font-medium text-zinc-300">Continuous Generation</span>
        </label>

        {/* Submit Button */}
        {loading && continuous ? (
          <button
            type="button"
            onClick={handleStop}
            className="w-full min-h-[56px] bg-red-500/20 hover:bg-red-500/30 text-red-500 border border-red-500/50 rounded-2xl p-4 text-base font-medium transition-all flex items-center justify-center gap-2 active:scale-[0.98]"
          >
            <Square className="w-4 h-4 fill-current" />
            Stop Generation
          </button>
        ) : (
          <button
            type="submit"
            disabled={loading || !prompt.trim()}
            className="w-full min-h-[56px] bg-indigo-500 hover:bg-indigo-600 disabled:bg-zinc-800 disabled:text-zinc-500 text-white rounded-2xl p-4 text-base font-medium transition-all flex items-center justify-center gap-2 active:scale-[0.98]"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Wand2 className="w-5 h-5" />
                Generate
              </>
            )}
          </button>
        )}
      </form>

      {error && (
        <div className="mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-2xl text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="mt-8 space-y-4">
          <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wider">Results</h3>
          <div className="grid grid-cols-2 gap-4">
            {results.map((img, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.1 }}
                className="aspect-square rounded-2xl overflow-hidden bg-zinc-900 border border-white/5 relative group"
              >
                <img src={img.url} alt="Generated" className="w-full h-full object-cover" />
                <div className="absolute inset-x-0 bottom-0 p-3 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                  <p className="text-[10px] text-zinc-300 font-mono">Seed: {img.seed}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}