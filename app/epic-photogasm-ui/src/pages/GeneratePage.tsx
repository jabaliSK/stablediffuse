import React, { useState, useRef } from 'react';
import { motion } from 'motion/react';
import { Wand2, Loader2, Play, Square, SlidersHorizontal } from 'lucide-react';
import { generateImages, ImageMeta } from '../services/api';

const ASPECT_RATIOS = [
  { label: 'Full Body (9:16)', width: 768, height: 1344, desc: 'Best for standing & action' },
  { label: 'Portrait (3:4)', width: 896, height: 1152, desc: 'Half body & fashion' },
  { label: 'Square (1:1)', width: 1024, height: 1024, desc: 'Standard 1024px' },
  { label: 'Landscape (16:9)', width: 1344, height: 768, desc: 'Wide cinema' },
];

export default function GeneratePage() {
  const [prompt, setPrompt] = useState('');
  const [batchSize, setBatchSize] = useState<number | string>(1);
  const [aspectIndex, setAspectIndex] = useState(0); // Default to Full Body 9:16
  const [steps, setSteps] = useState(25); // FLUX.1-dev standard is 20-30 steps
  const [guidance, setGuidance] = useState(3.5); // FLUX.1-dev guidance scale
  const [loraScale, setLoraScale] = useState(1.0); // Uncensored LoRA weight
  const [seed, setSeed] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  const [loading, setLoading] = useState(false);
  const [isContinuous, setIsContinuous] = useState(false);
  const stopRef = useRef(false);
  
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ImageMeta[]>([]);

  const selectedAspect = ASPECT_RATIOS[aspectIndex];

  const runGeneration = async () => {
    return await generateImages({
      prompt,
      width: selectedAspect.width,
      height: selectedAspect.height,
      batch_size: typeof batchSize === 'number' ? batchSize : (parseInt(batchSize as string) || 1),
      steps,
      guidance,
      lora_scale: loraScale,
      seed: seed ? parseInt(seed, 10) : undefined
    });
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    
    setLoading(true);
    setError(null);
    setResults([]);
    
    try {
      const images = await runGeneration();
      setResults(images);
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleContinuous = async () => {
    if (!prompt.trim()) return;
    
    setIsContinuous(true);
    stopRef.current = false;
    setLoading(true);
    setError(null);
    setResults([]);
    
    try {
      while (!stopRef.current) {
        const images = await runGeneration();
        if (stopRef.current) break;
        setResults(prev => [...images, ...prev].slice(0, 20));
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
      setIsContinuous(false);
    }
  };

  const handleStop = () => {
    stopRef.current = true;
  };

  return (
    <div className="p-6 pb-24 max-w-md mx-auto">
      <form onSubmit={handleGenerate} className="space-y-5">
        {/* Prompt */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Prompt</label>
            <span className="text-[10px] text-indigo-400 font-medium">FLUX.1 Dev + Uncensored</span>
          </div>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Full-length candid photo of an athletic woman in a dynamic pose, intricate fashion garments, realistic skin pores, natural ambient lighting, 8k..."
            className="w-full min-h-[44px] bg-zinc-900/50 border border-white/10 rounded-2xl p-4 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all resize-none h-32"
            required
          />
        </div>

        {/* Aspect Ratio Selector */}
        <div className="space-y-2">
          <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Aspect Ratio</label>
          <div className="grid grid-cols-2 gap-2">
            {ASPECT_RATIOS.map((item, idx) => (
              <button
                type="button"
                key={item.label}
                onClick={() => setAspectIndex(idx)}
                className={`p-2.5 rounded-xl border text-left transition-all ${
                  aspectIndex === idx
                    ? 'border-indigo-500 bg-indigo-500/10 text-white'
                    : 'border-white/5 bg-zinc-900/40 text-zinc-400 hover:border-white/10'
                }`}
              >
                <div className="text-xs font-medium text-zinc-200">{item.label}</div>
                <div className="text-[10px] text-zinc-500">{item.desc}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Basic Controls Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Images</label>
            <input
              type="number"
              inputMode="numeric"
              pattern="[0-9]*"
              value={batchSize}
              onChange={(e) => setBatchSize(e.target.value === '' ? '' : parseInt(e.target.value))}
              placeholder="1"
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
        </div>

        {/* Advanced Settings Toggle */}
        <div className="border border-white/5 rounded-2xl bg-zinc-900/30 overflow-hidden">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full px-4 py-3 flex items-center justify-between text-xs font-medium text-zinc-400 hover:text-zinc-200 transition-colors"
          >
            <span className="flex items-center gap-2">
              <SlidersHorizontal className="w-3.5 h-3.5" />
              Advanced Settings
            </span>
            <span className="text-[11px] text-zinc-500">{showAdvanced ? 'Hide' : 'Show'}</span>
          </button>
          
          {showAdvanced && (
            <div className="p-4 pt-2 space-y-4 border-t border-white/5 mt-2">
              {/* Uncensored LoRA Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center">
                  <label className="text-[11px] font-medium text-zinc-300">Uncensored LoRA Strength</label>
                  <span className="text-xs font-mono text-indigo-400 font-semibold">{loraScale.toFixed(2)}x</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.5"
                  step="0.05"
                  value={loraScale}
                  onChange={(e) => setLoraScale(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-zinc-800 rounded-lg"
                />
                <div className="flex justify-between text-[9px] text-zinc-500 font-mono">
                  <span>0.0 (Off / Base)</span>
                  <span>1.0 (Full Uncensored)</span>
                  <span>1.5 (High)</span>
                </div>
              </div>

              {/* Steps and Guidance Grid */}
              <div className="grid grid-cols-2 gap-3 pt-1">
                <div className="space-y-1.5">
                  <div className="flex justify-between items-center">
                    <label className="text-[11px] font-medium text-zinc-400">Steps</label>
                    <span className="text-[10px] text-zinc-500">20-30</span>
                  </div>
                  <input
                    type="number"
                    min="10"
                    max="50"
                    value={steps}
                    onChange={(e) => setSteps(parseInt(e.target.value) || 25)}
                    className="w-full bg-zinc-900/50 border border-white/10 rounded-xl p-2.5 text-xs text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                  />
                </div>
                <div className="space-y-1.5">
                  <div className="flex justify-between items-center">
                    <label className="text-[11px] font-medium text-zinc-400">Guidance (CFG)</label>
                    <span className="text-[10px] text-zinc-500">3.5</span>
                  </div>
                  <input
                    type="number"
                    step="0.5"
                    min="1.0"
                    max="10.0"
                    value={guidance}
                    onChange={(e) => setGuidance(parseFloat(e.target.value) || 3.5)}
                    className="w-full bg-zinc-900/50 border border-white/10 rounded-xl p-2.5 text-xs text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Submit Buttons */}
        <div className="space-y-3">
          {!isContinuous ? (
            <>
              <button
                type="submit"
                disabled={loading || !prompt.trim()}
                className="w-full min-h-[56px] bg-indigo-500 hover:bg-indigo-600 disabled:bg-zinc-800 disabled:text-zinc-500 text-white rounded-2xl p-4 text-base font-medium transition-all flex items-center justify-center gap-2 active:scale-[0.98]"
              >
                {loading ? (
                  <><Loader2 className="w-5 h-5 animate-spin" /> Generating...</>
                ) : (
                  <><Wand2 className="w-5 h-5" /> Generate Once</>
                )}
              </button>
              
              <button
                type="button"
                onClick={handleContinuous}
                disabled={loading || !prompt.trim()}
                className="w-full min-h-[56px] bg-zinc-800 hover:bg-zinc-700 disabled:bg-zinc-800/50 disabled:text-zinc-600 text-white rounded-2xl p-4 text-base font-medium transition-all flex items-center justify-center gap-2 active:scale-[0.98]"
              >
                <Play className="w-5 h-5" /> Continuous Generation
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={handleStop}
              className="w-full min-h-[56px] bg-red-500 hover:bg-red-600 text-white rounded-2xl p-4 text-base font-medium transition-all flex items-center justify-center gap-2 active:scale-[0.98]"
            >
              <Square className="w-5 h-5" /> Stop Generation
            </button>
          )}
        </div>
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
                transition={{ delay: 0.1 }}
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