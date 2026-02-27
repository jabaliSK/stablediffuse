import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X } from 'lucide-react';
import { getApiUrl, setApiUrl } from '../services/api';

export default function SettingsModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [url, setUrl] = useState('');

  useEffect(() => {
    if (isOpen) {
      setUrl(getApiUrl());
    }
  }, [isOpen]);

  const handleSave = () => {
    setApiUrl(url);
    onClose();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed left-4 right-4 top-1/2 -translate-y-1/2 bg-zinc-900 border border-white/10 rounded-2xl p-6 z-50 shadow-2xl"
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-medium text-zinc-100">Settings</h2>
              <button onClick={onClose} className="text-zinc-400 hover:text-zinc-100">
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider mb-2">
                  Backend API URL
                </label>
                <input
                  type="text"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="http://192.168.1.100:5000"
                  className="w-full bg-zinc-950 border border-white/10 rounded-xl px-4 py-3 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
                />
                <p className="text-xs text-zinc-500 mt-2">
                  Leave empty or set to "mock" to use the built-in mock generator.
                </p>
              </div>
              
              <button
                onClick={handleSave}
                className="w-full bg-indigo-500 hover:bg-indigo-600 text-white rounded-xl px-4 py-3 text-sm font-medium transition-colors"
              >
                Save Changes
              </button>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
