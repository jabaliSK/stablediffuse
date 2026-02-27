import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Wand2, Loader2 } from 'lucide-react';
import { generateImages, ImageMeta } from '../services/api';

export default function GeneratePage() {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('blurry, low-res, text, watermark');
  const [batchSize, setBatchSize] = useState(1);
  const [steps, setSteps] = useState(28);
  const [guidance, setGuidance] = useState(6.5);
  const [seed, setSeed] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ImageMeta[]>([]);

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    
    setLoading(true);
    setError(null);
    setResults([]);
    
    try {
      const images = await generateImages({
        prompt,
        negative_prompt: negativePrompt,
        batch_size: batchSize,
        steps,
        guidance,
        seed: seed ? parseInt(seed, 10) : undefined
      });
      setResults(images);
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
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
            className="w-full bg-zinc-900/50 border border-white/10 rounded-2xl p-4 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all resize-none h-32"
            required
          />
        </div>

        {/* Negative Prompt */}
        <div className="space-y-2">
          <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Negative Prompt</label>
          <textarea
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            className="w-full bg-zinc-900/50 border border-white/10 rounded-2xl p-4 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all resize-none h-20"
          />
        </div>

        {/* Controls Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Images</label>
            <input
              type="number"
              min="1"
              max="4"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
              className="w-full bg-zinc-900/50 border border-white/10 rounded-xl p-3 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
            />
          </div>
          <div className="space-y-2">
            <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Seed</label>
            <input
              type="text"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="Random"
              className="w-full bg-zinc-900/50 border border-white/10 rounded-xl p-3 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
            />
          </div>
          <div className="space-y-2">
            <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Steps</label>
            <input
              type="number"
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value) || 28)}
              className="w-full bg-zinc-900/50 border border-white/10 rounded-xl p-3 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
            />
          </div>
          <div className="space-y-2">
            <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Guidance</label>
            <input
              type="number"
              step="0.1"
              value={guidance}
              onChange={(e) => setGuidance(parseFloat(e.target.value) || 6.5)}
              className="w-full bg-zinc-900/50 border border-white/10 rounded-xl p-3 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
            />
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading || !prompt.trim()}
          className="w-full bg-indigo-500 hover:bg-indigo-600 disabled:bg-zinc-800 disabled:text-zinc-500 text-white rounded-2xl p-4 text-sm font-medium transition-all flex items-center justify-center gap-2 active:scale-[0.98]"
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
