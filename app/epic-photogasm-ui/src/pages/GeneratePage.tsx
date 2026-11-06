import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'motion/react';
import { Wand2, Loader2, Play, Square, SlidersHorizontal, CheckCircle2, Images } from 'lucide-react';
import { Link } from 'react-router-dom';
import { 
  generateImages, 
  ImageMeta, 
  startContinuous, 
  stopContinuous, 
  getContinuousStatus, 
  ContinuousStatus 
} from '../services/api';

const ASPECT_RATIOS = [
  { label: 'Full Body (9:16)', width: 768, height: 1344, desc: 'Best for standing & action' },
  { label: 'Portrait (3:4)', width: 896, height: 1152, desc: 'Half body & fashion' },
  { label: 'Square (1:1)', width: 1024, height: 1024, desc: 'Standard 1024px' },
  { label: 'Landscape (16:9)', width: 1344, height: 768, desc: 'Wide cinema' },
];

const BATCH_PRESETS = [1, 2, 4, 8, 12];

export default function GeneratePage() {
  const [prompt, setPrompt] = useState('');
  const [batchSize, setBatchSize] = useState<number | string>(1);
  const [aspectIndex, setAspectIndex] = useState(0); // Default to Full Body 9:16
  const [steps, setSteps] = useState(25); // FLUX.1-dev standard is 20-30 steps
  const [guidance, setGuidance] = useState(3.5); // FLUX.1-dev guidance scale
  const [seed, setSeed] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ImageMeta[]>([]);

  // Server-side continuous generation status (persists across phone app-switching)
  const [continuousStatus, setContinuousStatus] = useState<ContinuousStatus>({
    active: false,
    prompt: '',
    total_generated: 0
  });

  const selectedAspect = ASPECT_RATIOS[aspectIndex];

  // Poll server continuous status so phone lock/reload always recovers state
  useEffect(() => {
    let isMounted = true;
    const checkStatus = async () => {
      try {
        const status = await getContinuousStatus();
        if (isMounted) {
          setContinuousStatus(status);
          if (status.active && status.prompt && !prompt) {
            setPrompt(status.prompt);
          }
        }
      } catch (err) {
        // Server might still be booting or idle
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 3000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const getPayload = () => ({
    prompt,
    width: selectedAspect.width,
    height: selectedAspect.height,
    batch_size: typeof batchSize === 'number' ? batchSize : (parseInt(batchSize as string) || 1),
    steps,
    guidance,
    seed: seed ? parseInt(seed, 10) : undefined
  });

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    
    setLoading(true);
    setError(null);
    setResults([]);
    
    try {
      const images = await generateImages(getPayload());
      setResults(images);
    } catch (err: any) {
      setError(err.message || 'An error occurred during generation');
    } finally {
      setLoading(false);
    }
  };

  const handleStartContinuous = async () => {
    if (!prompt.trim()) return;
    setError(null);
    try {
      const res = await startContinuous(getPayload());
      setContinuousStatus(res.stats);
    } catch (err: any) {
      setError(err.message || 'Failed to start continuous generation on server');
    }
  };

  const handleStopContinuous = async () => {
    try {
      const res = await stopContinuous();
      setContinuousStatus(res.stats);
    } catch (err: any) {
      setError(err.message || 'Failed to stop continuous generation');
    }
  };

  return (
    <div className="p-4 sm:p-6 pb-28 max-w-md mx-auto">
      {/* Mobile Background Generation Alert Banner */}
      {continuousStatus.active && (
        <div className="mb-5 p-4 rounded-2xl bg-indigo-950/70 border border-indigo-500/40 text-indigo-100 shadow-lg backdrop-blur-md animate-pulse">
          <div className="flex items-center justify-between mb-2">
            <span className="flex items-center gap-2 text-xs font-semibold tracking-wide uppercase text-indigo-300">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
              Server Running Non-Stop
            </span>
            <span className="text-xs font-mono font-bold bg-indigo-500/30 px-2 py-0.5 rounded-full text-indigo-200">
              {continuousStatus.total_generated} generated
            </span>
          </div>
          <p className="text-xs text-indigo-200/80 leading-relaxed">
            Generating in background on the GPU. You can lock your phone screen or switch apps; images will keep saving to the Gallery.
          </p>
          <div className="mt-3 flex items-center gap-2">
            <button
              type="button"
              onClick={handleStopContinuous}
              className="flex-1 py-2 px-3 bg-red-500 hover:bg-red-600 active:scale-95 text-white font-medium text-xs rounded-xl transition-all flex items-center justify-center gap-1.5 shadow"
            >
              <Square className="w-3.5 h-3.5 fill-current" /> Stop Loop
            </button>
            <Link
              to="/gallery"
              className="py-2 px-3 bg-white/10 hover:bg-white/20 active:scale-95 text-zinc-200 font-medium text-xs rounded-xl transition-all flex items-center justify-center gap-1"
            >
              <Images className="w-3.5 h-3.5" /> Gallery
            </Link>
          </div>
        </div>
      )}

      <form onSubmit={handleGenerate} className="space-y-5">
        {/* Prompt */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Prompt</label>
            <span className="text-[10px] text-indigo-400 font-medium">FLUX.1 Dev (Parallel Batch)</span>
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

        {/* Batch Size & Seed Controls */}
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider">Batch Size (1 - 12)</label>
            <div className="flex gap-1">
              {BATCH_PRESETS.map((preset) => (
                <button
                  type="button"
                  key={preset}
                  onClick={() => setBatchSize(preset)}
                  className={`px-2 py-1 rounded-lg text-xs font-mono transition-all ${
                    Number(batchSize) === preset
                      ? 'bg-indigo-500 text-white font-bold'
                      : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-white/5'
                  }`}
                >
                  {preset}
                </button>
              ))}
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <input
                type="number"
                min="1"
                max="12"
                inputMode="numeric"
                pattern="[0-9]*"
                value={batchSize}
                onChange={(e) => {
                  const val = e.target.value;
                  if (val === '') setBatchSize('');
                  else setBatchSize(Math.max(1, Math.min(12, parseInt(val) || 1)));
                }}
                placeholder="1"
                className="w-full min-h-[44px] bg-zinc-900/50 border border-white/10 rounded-xl p-3 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all font-mono"
              />
            </div>
            <div className="space-y-1">
              <input
                type="text"
                inputMode="numeric"
                pattern="[0-9]*"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                placeholder="Seed: Random"
                className="w-full min-h-[44px] bg-zinc-900/50 border border-white/10 rounded-xl p-3 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
              />
            </div>
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
              Advanced Settings (Steps & Guidance)
            </span>
            <span className="text-[11px] text-zinc-500">{showAdvanced ? 'Hide' : 'Show'}</span>
          </button>
          
          {showAdvanced && (
            <div className="p-4 pt-2 border-t border-white/5 mt-2">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <div className="flex justify-between items-center">
                    <label className="text-[11px] font-medium text-zinc-400">Steps</label>
                    <span className="text-[10px] text-zinc-500">15-20 fast, 25 fine</span>
                  </div>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    value={steps}
                    onChange={(e) => setSteps(parseInt(e.target.value) || 20)}
                    className="w-full bg-zinc-900/50 border border-white/10 rounded-xl p-2.5 text-xs text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 font-mono"
                  />
                </div>
                <div className="space-y-1.5">
                  <div className="flex justify-between items-center">
                    <label className="text-[11px] font-medium text-zinc-400">Guidance (CFG)</label>
                    <span className="text-[10px] text-zinc-500">3.5 default</span>
                  </div>
                  <input
                    type="number"
                    step="0.5"
                    min="1.0"
                    max="10.0"
                    value={guidance}
                    onChange={(e) => setGuidance(parseFloat(e.target.value) || 3.5)}
                    className="w-full bg-zinc-900/50 border border-white/10 rounded-xl p-2.5 text-xs text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 font-mono"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Submit Buttons */}
        <div className="space-y-3 pt-1">
          <button
            type="submit"
            disabled={loading || continuousStatus.active || !prompt.trim()}
            className="w-full min-h-[56px] bg-indigo-500 hover:bg-indigo-600 disabled:bg-zinc-800 disabled:text-zinc-500 text-white rounded-2xl p-4 text-base font-medium transition-all flex items-center justify-center gap-2 active:scale-[0.98] shadow-md"
          >
            {loading ? (
              <><Loader2 className="w-5 h-5 animate-spin" /> Generating {batchSize} image(s)...</>
            ) : (
              <><Wand2 className="w-5 h-5" /> Generate {Number(batchSize) > 1 ? `Batch (${batchSize})` : 'Once'}</>
            )}
          </button>
          
          {!continuousStatus.active ? (
            <button
              type="button"
              onClick={handleStartContinuous}
              disabled={loading || !prompt.trim()}
              className="w-full min-h-[56px] bg-zinc-900 hover:bg-zinc-800 border border-white/10 disabled:bg-zinc-900/50 disabled:text-zinc-600 text-white rounded-2xl p-4 text-base font-medium transition-all flex items-center justify-center gap-2 active:scale-[0.98]"
            >
              <Play className="w-5 h-5 text-emerald-400 fill-emerald-400/20" /> Continuous Generation (Server-Side)
            </button>
          ) : (
            <button
              type="button"
              onClick={handleStopContinuous}
              className="w-full min-h-[56px] bg-red-500 hover:bg-red-600 active:scale-[0.98] text-white rounded-2xl p-4 text-base font-medium transition-all flex items-center justify-center gap-2 shadow-lg"
            >
              <Square className="w-5 h-5 fill-current" /> Stop Continuous Generation
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
          <div className="flex justify-between items-center">
            <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wider">Latest Batch ({results.length})</h3>
            <Link to="/gallery" className="text-xs text-indigo-400 hover:underline">View All in Gallery →</Link>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {results.map((img, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.05 }}
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