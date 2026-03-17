import React from 'react'

/**
 * Results
 * Displays accent detection, confidence, transcript, and translation.
 *
 * Props
 * -----
 * results : {
 *   accent: string,
 *   confidence: number,
 *   transcript: string,
 *   translation: string,
 *   probabilities: { [accent]: number }
 * }
 * targetLanguage : string
 */
export default function Results({ results, targetLanguage }) {
  const { accent, confidence, transcript, translation, probabilities } = results

  const confidencePct = Math.round(confidence * 100)

  // Sort probabilities descending for display
  const sortedProbs = Object.entries(probabilities || {}).sort((a, b) => b[1] - a[1])

  const accentEmojis = {
    korean:  '🇰🇷',
    indian:  '🇮🇳',
    spanish: '🇪🇸',
    chinese: '🇨🇳',
  }

  const topEmoji = accentEmojis[accent?.toLowerCase()] || '🌍'

  const confidenceColor =
    confidencePct >= 75 ? 'text-green-600' :
    confidencePct >= 50 ? 'text-yellow-600' :
    'text-red-500'

  return (
    <div className="space-y-4">
      {/* Accent + confidence card */}
      <div className="bg-white rounded-2xl shadow-sm p-6 flex flex-col sm:flex-row items-center gap-6">
        <div className="text-6xl">{topEmoji}</div>
        <div className="flex-1 text-center sm:text-left">
          <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-0.5">
            Detected Accent
          </p>
          <h2 className="text-3xl font-bold text-accent-700">{accent}</h2>
          <p className={`text-lg font-semibold mt-1 ${confidenceColor}`}>
            {confidencePct}% confidence
          </p>
        </div>

        {/* Confidence bar */}
        <div className="w-full sm:w-40">
          <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full bg-accent-500 transition-all duration-700"
              style={{ width: `${confidencePct}%` }}
            />
          </div>
          <p className="text-xs text-gray-400 mt-1 text-right">{confidencePct}%</p>
        </div>
      </div>

      {/* Probability breakdown */}
      {sortedProbs.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm p-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Accent Probabilities</h3>
          <div className="space-y-2">
            {sortedProbs.map(([cls, prob]) => {
              const pct = Math.round(prob * 100)
              const barClass = `h-full rounded-full ${
                cls === accent?.toLowerCase() ? 'bg-accent-500' : 'bg-gray-300'
              }`
              return (
                <div key={cls} className="flex items-center gap-3">
                  <span className="text-base w-5">{accentEmojis[cls] || '🌍'}</span>
                  <span className="w-20 text-sm capitalize text-gray-600">{cls}</span>
                  <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                    <div className={barClass} style={{ width: `${pct}%` }} />
                  </div>
                  <span className="text-xs text-gray-500 w-10 text-right">{pct}%</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Transcript */}
      <div className="bg-white rounded-2xl shadow-sm p-6">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">📝 Transcript</h3>
        <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">
          {transcript || <span className="text-gray-400 italic">No transcript available.</span>}
        </p>
      </div>

      {/* Translation */}
      <div className="bg-white rounded-2xl shadow-sm p-6">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">
          🌐 Translation{' '}
          <span className="font-normal text-gray-400 capitalize">({targetLanguage})</span>
        </h3>
        <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">
          {translation || <span className="text-gray-400 italic">No translation available.</span>}
        </p>
      </div>
    </div>
  )
}
