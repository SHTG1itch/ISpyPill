import { useState, useRef } from 'react'
import UploadCard from './components/UploadCard'
import ResultsCard from './components/ResultsCard'
import ErrorBanner from './components/ErrorBanner'

export default function App() {
  const [refFile,   setRefFile]   = useState(null)
  const [groupFile, setGroupFile] = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState(null)
  const [results,   setResults]   = useState(null)

  const resultsRef = useRef(null)
  const canAnalyze = refFile && groupFile && !loading

  async function handleAnalyze() {
    if (!canAnalyze) return
    setLoading(true)
    setError(null)
    setResults(null)

    const body = new FormData()
    body.append('reference_pill', refFile)
    body.append('group_photo',    groupFile)

    try {
      const resp = await fetch('/analyze', { method: 'POST', body })
      const data = await resp.json()
      if (!resp.ok) {
        setError(data.error || 'Something went wrong. Please try again.')
        return
      }
      setResults(data)
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80)
    } catch {
      setError('Network error — please check your connection and try again.')
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setRefFile(null)
    setGroupFile(null)
    setResults(null)
    setError(null)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">💊</span>
            <span className="logo-text">ISpyPill</span>
          </div>
          <p className="tagline">Accurate pill counting from photos</p>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="main">

        {/* Upload cards */}
        <div className="upload-grid">
          <UploadCard
            step={1}
            title="Reference Pill"
            subtitle="Photo of one pill"
            tip="Place the pill on a plain contrasting background."
            file={refFile}
            onFile={setRefFile}
          />
          <UploadCard
            step={2}
            title="Group Photo"
            subtitle="Photo of all pills to count"
            tip="Spread pills flat on a surface for best accuracy."
            file={groupFile}
            onFile={setGroupFile}
          />
        </div>

        {/* Error */}
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

        {/* Analyse button */}
        <div className="action-row">
          <button
            className="count-btn"
            disabled={!canAnalyze}
            onClick={handleAnalyze}
          >
            {loading ? <><span className="spinner" /> Analyzing…</> : 'Count Pills'}
          </button>
        </div>

        {/* Results */}
        {results && (
          <div ref={resultsRef}>
            <ResultsCard data={results} onReset={handleReset} />
          </div>
        )}

      </main>

      {/* ── Footer ── */}
      <footer className="footer">
        <p>ISpyPill — For reference only. Always verify counts with a licensed pharmacist.</p>
      </footer>

    </div>
  )
}
