import React, { useState, useRef, useCallback } from 'react'

/**
 * AudioUploader
 * Allows the user to either:
 *   - Upload an audio file via drag-and-drop or file picker
 *   - Record audio directly in the browser
 *
 * Props
 * -----
 * onAudioReady(blob, filename) — called when audio is ready to analyse
 * isLoading : bool
 */
export default function AudioUploader({ onAudioReady, isLoading }) {
  const [mode, setMode]               = useState('upload')  // 'upload' | 'record'
  const [isDragging, setIsDragging]   = useState(false)
  const [audioFile, setAudioFile]     = useState(null)      // File object
  const [audioUrl, setAudioUrl]       = useState(null)      // Object URL for preview
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [recordError, setRecordError] = useState(null)

  const fileInputRef   = useRef(null)
  const mediaRecorder  = useRef(null)
  const audioChunks    = useRef([])
  const timerRef       = useRef(null)

  // -------------------------------------------------------------------------
  // File upload helpers
  // -------------------------------------------------------------------------

  const handleFile = useCallback((file) => {
    if (!file) return
    const allowed = ['audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/mp4', 'audio/webm']
    if (!allowed.includes(file.type) && !file.type.startsWith('audio/')) {
      alert(`Unsupported file type: ${file.type}. Please upload an audio file.`)
      return
    }
    setAudioFile(file)
    setAudioUrl(URL.createObjectURL(file))
  }, [])

  const onFileChange = (e) => handleFile(e.target.files[0])

  const onDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  // -------------------------------------------------------------------------
  // Recording helpers
  // -------------------------------------------------------------------------

  const startRecording = async () => {
    setRecordError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      audioChunks.current = []
      const recorder = new MediaRecorder(stream)
      mediaRecorder.current = recorder

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.current.push(e.data)
      }

      recorder.onstop = () => {
        const blob = new Blob(audioChunks.current, { type: 'audio/webm' })
        const url  = URL.createObjectURL(blob)
        setAudioFile(new File([blob], 'recording.webm', { type: 'audio/webm' }))
        setAudioUrl(url)
        stream.getTracks().forEach((t) => t.stop())
        clearInterval(timerRef.current)
        setRecordingTime(0)
      }

      recorder.start()
      setIsRecording(true)
      setRecordingTime(0)
      timerRef.current = setInterval(() => setRecordingTime((t) => t + 1), 1000)
    } catch (err) {
      setRecordError('Microphone access denied. Please allow microphone access and try again.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorder.current && mediaRecorder.current.state !== 'inactive') {
      mediaRecorder.current.stop()
    }
    setIsRecording(false)
  }

  const formatTime = (s) =>
    `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`

  // -------------------------------------------------------------------------
  // Submit
  // -------------------------------------------------------------------------

  const handleSubmit = () => {
    if (audioFile && onAudioReady) {
      onAudioReady(audioFile, audioFile.name)
    }
  }

  const handleClear = () => {
    setAudioFile(null)
    setAudioUrl(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="bg-white rounded-2xl shadow-sm p-6 space-y-4">
      {/* Mode tabs */}
      <div className="flex gap-2">
        {['upload', 'record'].map((m) => (
          <button
            key={m}
            onClick={() => { setMode(m); handleClear() }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              mode === m
                ? 'bg-accent-600 text-white shadow-sm'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {m === 'upload' ? '📁 Upload File' : '🎙️ Record Audio'}
          </button>
        ))}
      </div>

      {/* Upload mode */}
      {mode === 'upload' && (
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={onDrop}
          onClick={() => !audioFile && fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
            isDragging
              ? 'border-accent-400 bg-accent-50'
              : audioFile
              ? 'border-green-300 bg-green-50 cursor-default'
              : 'border-gray-300 hover:border-accent-400 hover:bg-accent-50'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={onFileChange}
            className="hidden"
          />
          {audioFile ? (
            <div className="space-y-1">
              <p className="text-green-600 font-medium">✅ {audioFile.name}</p>
              <p className="text-xs text-gray-500">
                {(audioFile.size / 1024).toFixed(1)} KB
              </p>
              <button
                onClick={(e) => { e.stopPropagation(); handleClear() }}
                className="text-xs text-red-500 hover:underline mt-1"
              >
                Remove
              </button>
            </div>
          ) : (
            <>
              <p className="text-4xl mb-2">🎵</p>
              <p className="text-gray-600 font-medium">Drag &amp; drop an audio file here</p>
              <p className="text-xs text-gray-400 mt-1">or click to browse (.wav, .mp3, .ogg, .flac…)</p>
            </>
          )}
        </div>
      )}

      {/* Record mode */}
      {mode === 'record' && (
        <div className="flex flex-col items-center gap-4 py-4">
          {recordError && (
            <p className="text-sm text-red-500">{recordError}</p>
          )}

          {!audioFile ? (
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`w-20 h-20 rounded-full flex items-center justify-center text-white text-3xl shadow-lg transition-transform active:scale-95 ${
                isRecording
                  ? 'bg-red-500 hover:bg-red-600 animate-pulse'
                  : 'bg-accent-600 hover:bg-accent-700'
              }`}
            >
              {isRecording ? '⏹' : '🎙️'}
            </button>
          ) : (
            <button
              onClick={handleClear}
              className="text-sm text-red-500 hover:underline"
            >
              🗑 Discard recording
            </button>
          )}

          {isRecording && (
            <p className="text-red-500 font-mono text-lg">{formatTime(recordingTime)}</p>
          )}

          {!isRecording && !audioFile && (
            <p className="text-sm text-gray-500">Press the button to start recording</p>
          )}
        </div>
      )}

      {/* Audio preview */}
      {audioUrl && (
        <audio controls src={audioUrl} className="w-full rounded-lg" />
      )}

      {/* Analyse button */}
      {audioFile && (
        <button
          onClick={handleSubmit}
          disabled={isLoading}
          className="w-full py-3 bg-accent-600 hover:bg-accent-700 disabled:opacity-50 text-white font-semibold rounded-xl transition-colors"
        >
          {isLoading ? 'Analysing…' : '🔍 Analyse Accent'}
        </button>
      )}
    </div>
  )
}
