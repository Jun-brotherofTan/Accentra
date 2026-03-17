import React, { useState } from 'react'
import AudioUploader from './components/AudioUploader'
import Results from './components/Results'

const LANGUAGES = [
  { value: 'spanish',    label: '🇪🇸 Spanish' },
  { value: 'french',     label: '🇫🇷 French' },
  { value: 'german',     label: '🇩🇪 German' },
  { value: 'chinese',    label: '🇨🇳 Chinese' },
  { value: 'korean',     label: '🇰🇷 Korean' },
  { value: 'japanese',   label: '🇯🇵 Japanese' },
  { value: 'portuguese', label: '🇧🇷 Portuguese' },
  { value: 'russian',    label: '🇷🇺 Russian' },
  { value: 'arabic',     label: '🇸🇦 Arabic' },
  { value: 'hindi',      label: '🇮🇳 Hindi' },
]

const API_BASE = import.meta.env.VITE_API_URL || '/api'

export default function App() {
  const [targetLanguage, setTargetLanguage] = useState('spanish')
  const [isLoading, setIsLoading]           = useState(false)
  const [results, setResults]               = useState(null)
  const [error, setError]                   = useState(null)

  const handleAudio = async (audioBlob, filename) => {
    setIsLoading(true)
    setError(null)
    setResults(null)

    try {
      const formData = new FormData()
      formData.append('file', audioBlob, filename || 'recording.wav')
      formData.append('target_language', targetLanguage)

      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || `Server error ${response.status}`)
      }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err.message || 'An unexpected error occurred.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-accent-50 to-white">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center gap-3">
          <span className="text-3xl">🎙️</span>
          <div>
            <h1 className="text-2xl font-bold text-accent-700 leading-tight">Accentra</h1>
            <p className="text-sm text-gray-500">AI-powered accent detection, transcription &amp; translation</p>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8 space-y-6">
        {/* Language selector */}
        <div className="bg-white rounded-2xl shadow-sm p-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Translate transcript to:
          </label>
          <select
            value={targetLanguage}
            onChange={(e) => setTargetLanguage(e.target.value)}
            className="w-full max-w-xs border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-400"
          >
            {LANGUAGES.map((lang) => (
              <option key={lang.value} value={lang.value}>
                {lang.label}
              </option>
            ))}
          </select>
        </div>

        {/* Audio uploader / recorder */}
        <AudioUploader onAudioReady={handleAudio} isLoading={isLoading} />

        {/* Loading spinner */}
        {isLoading && (
          <div className="flex items-center justify-center py-10">
            <div className="flex flex-col items-center gap-3 text-accent-600">
              <svg
                className="animate-spin h-10 w-10"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12" cy="12" r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v8H4z"
                />
              </svg>
              <span className="text-sm font-medium">Analysing audio…</span>
            </div>
          </div>
        )}

        {/* Error message */}
        {error && !isLoading && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded-2xl px-5 py-4 flex gap-3 items-start">
            <span className="text-xl">⚠️</span>
            <div>
              <p className="font-semibold">Something went wrong</p>
              <p className="text-sm mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {/* Results */}
        {results && !isLoading && (
          <Results results={results} targetLanguage={targetLanguage} />
        )}
      </main>

      <footer className="text-center text-xs text-gray-400 py-6">
        Accentra — Powered by Whisper, wav2vec2 &amp; MarianMT
      </footer>
    </div>
  )
}
