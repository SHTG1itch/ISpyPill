import { useEffect, useRef } from 'react'

export default function ResultsCard({ data, onReset }) {
  const countRef = useRef(null)

  // Animate count from 0 → data.count
  useEffect(() => {
    const target   = data.count || 0
    const duration = 500
    const start    = performance.now()

    function step(now) {
      const t     = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - t, 3)
      if (countRef.current) {
        countRef.current.textContent = Math.round(eased * target).toLocaleString()
      }
      if (t < 1) requestAnimationFrame(step)
      else if (countRef.current) countRef.current.textContent = target.toLocaleString()
    }
    requestAnimationFrame(step)
  }, [data.count])

  // Build info badges
  const badges = []
  if (typeof data.num_color_clusters === 'number') {
    badges.push(data.num_color_clusters === 1 ? '1 color' : `${data.num_color_clusters} colors`)
  }
  if (data.is_white_pill !== undefined) {
    badges.push(data.is_white_pill ? 'Brightness mode' : 'Color mode')
  }
  if (typeof data.ref_area_px === 'number') {
    badges.push(`${data.ref_area_px.toLocaleString()} px² ref`)
  }

  const imgSrc = data.annotated_image
    ? `data:image/jpeg;base64,${data.annotated_image}`
    : null

  return (
    <div className="results-card">

      {/* Count */}
      <p className="results-label">Pills counted</p>
      <p className="count-number" ref={countRef}>0</p>

      {/* Info badges */}
      {badges.length > 0 && (
        <div className="badges-row">
          {badges.map(b => (
            <span className="badge" key={b}>{b}</span>
          ))}
        </div>
      )}

      {/* Annotated image */}
      {imgSrc && (
        <div className="annotated-section">
          <p className="annotated-label">Detection map</p>
          <img src={imgSrc} alt="Annotated pill detection" className="annotated-img" />
        </div>
      )}

      {/* Actions */}
      <div className="results-actions">
        {imgSrc && (
          <a href={imgSrc} download="pill_count.jpg" className="btn btn-outline">
            <svg viewBox="0 0 24 24" width="15" height="15" fill="none"
                 stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Save Image
          </a>
        )}
        <button className="btn btn-primary" type="button" onClick={onReset}>
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none"
               stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <polyline points="1 4 1 10 7 10" />
            <path d="M3.51 15a9 9 0 1 0 .49-4.95" />
          </svg>
          Count Another
        </button>
      </div>

    </div>
  )
}
