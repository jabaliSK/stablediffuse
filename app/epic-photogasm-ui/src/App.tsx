/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Image as ImageIcon, Grid, Settings } from 'lucide-react';
import GeneratePage from './pages/GeneratePage';
import GalleryPage from './pages/GalleryPage';

export default function App() {
  const [activeTab, setActiveTab] = useState<'generate' | 'gallery'>('generate');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  return (
    <div className="flex flex-col h-[100dvh] bg-zinc-950 overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-white/5 bg-zinc-950/50 backdrop-blur-md z-10">
        <h1 className="text-xl font-medium tracking-tight text-zinc-100">
          {activeTab === 'generate' ? 'Create' : 'Gallery'}
        </h1>
        <button 
          onClick={() => setIsSettingsOpen(true)}
          className="p-2 -mr-2 text-zinc-400 hover:text-zinc-100 transition-colors"
        >
          <Settings className="w-5 h-5" />
        </button>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 overflow-hidden relative">
        <AnimatePresence mode="wait">
          {activeTab === 'generate' ? (
            <motion.div
              key="generate"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className="absolute inset-0 overflow-y-auto"
            >
              <GeneratePage />
            </motion.div>
          ) : (
            <motion.div
              key="gallery"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className="absolute inset-0 overflow-y-auto"
            >
              <GalleryPage />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Bottom Navigation */}
      <nav className="flex items-center justify-around px-6 py-4 border-t border-white/5 bg-zinc-950/80 backdrop-blur-xl pb-safe">
        <button
          onClick={() => setActiveTab('generate')}
          className={`flex flex-col items-center gap-1 transition-colors ${
            activeTab === 'generate' ? 'text-indigo-400' : 'text-zinc-500 hover:text-zinc-300'
          }`}
        >
          <ImageIcon className="w-6 h-6" />
          <span className="text-[10px] font-medium uppercase tracking-wider">Generate</span>
        </button>
        <button
          onClick={() => setActiveTab('gallery')}
          className={`flex flex-col items-center gap-1 transition-colors ${
            activeTab === 'gallery' ? 'text-indigo-400' : 'text-zinc-500 hover:text-zinc-300'
          }`}
        >
          <Grid className="w-6 h-6" />
          <span className="text-[10px] font-medium uppercase tracking-wider">Gallery</span>
        </button>
      </nav>
    </div>
  );
}
