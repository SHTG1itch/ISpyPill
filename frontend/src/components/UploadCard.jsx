import { useEffect, useRef, useState } from 'react'

export default function UploadCard({ step, title, subtitle, tip, file, onFile }) {
  const [dragOver, setDragOver] = useState(false)
  const [preview,  setPreview]  = useState(null)

  const galleryRef = useRef(null)
  const cameraRef  = useRef(null)

  // Create/revoke object URL for preview
  useEffect(() => {
    if (!file) { setPreview(null); return }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  function accept(f) {
    if (!f) return
    if (!f.type.startsWith('image/')) return
    if (f.size > 16 * 1024 * 1024) return
    onFile(f)
  }

  function openGallery() { galleryRef.current?.click() }

  // Drop zone handlers
  function onDragOver(e)  { e.preventDefault(); setDragOver(true) }
  function onDragLeave(e) { if (!e.currentTarget.contains(e.relatedTarget)) setDragOver(false) }
  function onDrop(e)      { e.preventDefault(); setDragOver(false); accept(e.dataTransfer.files?.[0]) }
  function onZoneClick()  { if (!file) openGallery() }
  function onZoneKey(e)   { if (!file && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); openGallery() } }

  const zoneClass = ['drop-zone', dragOver && 'drag-over', file && 'has-image'].filter(Boolean).join(' ')

  return (
    <div className="upload-card">

      {/* Card header */}
      <div className="card-header">
        <span className="step-badge">{step}</span>
        <div>
          <h2 className="card-title">{title}</h2>
          <p className="card-subtitle">{subtitle}</p>
        </div>
      </div>

      {/* Drop zone */}
      <div
        className={zoneClass}
        onClick={onZoneClick}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        role={file ? undefined : 'button'}
        tabIndex={file ? undefined : 0}
        onKeyDown={onZoneKey}
        aria-label={file ? undefined : `Upload ${title}`}
      >
        {preview ? (
          <>
            <img src={preview} alt={`${title} preview`} className="preview-img" />
            <button
              className="clear-btn"
              type="button"
              onClick={e => { e.stopPropagation(); onFile(null) }}
              aria-label="Remove photo"
            >
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none"
                   stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <line x1="18" y1="6"  x2="6"  y2="18" />
                <line x1="6"  y1="6"  x2="18" y2="18" />
              </svg>
            </button>
          </>
        ) : (
          <div className="zone-placeholder">
            <svg className="zone-icon" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <p className="zone-text">Tap or drag to upload</p>
            <p className="zone-hint">PNG · JPG · HEIC</p>
          </div>
        )}
      </div>

      {/* Action buttons */}
      <div className="card-actions">
        <label className="btn btn-outline">
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none"
               stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21 15 16 10 5 21" />
          </svg>
          Gallery
          <input
            ref={galleryRef}
            type="file"
            accept="image/*"
            className="sr-only"
            onChange={e => { accept(e.target.files?.[0]); e.target.value = '' }}
          />
        </label>

        <label className="btn btn-outline">
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none"
               stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
            <circle cx="12" cy="13" r="4" />
          </svg>
          Camera
          <input
            ref={cameraRef}
            type="file"
            accept="image/*"
            capture="environment"
            className="sr-only"
            onChange={e => { accept(e.target.files?.[0]); e.target.value = '' }}
          />
        </label>
      </div>

      <p className="card-tip">{tip}</p>
    </div>
  )
}
