import Constants from 'expo-constants'

// ── Where is the Flask backend? ────────────────────────────────────────────────
//
// There are two ways the phone connects to Metro, and they need different backend
// resolution:
//
//   • LAN  (`npm start` / `expo start`):  Metro's host IS your computer's LAN IP,
//       e.g. "192.168.x.x:8081", so the Flask backend is at that same IP on :5000.
//       We can auto-detect it — no configuration needed.
//
//   • Tunnel (`expo start --tunnel`):  Metro's host is a public tunnel address like
//       "u-anon-xxxx.exp.direct". The tunnel ONLY forwards Metro (the JS bundle),
//       NOT port 5000, so "<tunnel-host>:5000" can never reach your local Flask.
//       In this case you must tell the app where the backend is, by setting the
//       EXPO_PUBLIC_API_URL environment variable before starting Expo (see README).
//
// Resolution order:
//   1. EXPO_PUBLIC_API_URL          — explicit override (required for tunnel mode)
//   2. expoConfig.extra.apiUrl      — optional override baked into app.json
//   3. LAN auto-detect from Metro   — the default for same-Wi-Fi development
//   4. localhost                    — simulator / web fallback

// Tunnel hostnames can't host the backend on :5000, so auto-detect must skip them.
const TUNNEL_HOST_PATTERN = /\.(exp\.direct|exp\.host|ngrok\.io|ngrok-free\.app)$/i

function metroHost() {
  const sources = [
    Constants.expoGoConfig?.debuggerHost,
    Constants.expoConfig?.hostUri,
    Constants.manifest?.debuggerHost,
    Constants.manifest2?.extra?.expoClient?.hostUri,
  ]
  for (const src of sources) {
    if (src && typeof src === 'string') {
      const host = src.split(':')[0]
      if (host) return host
    }
  }
  return null
}

function resolveApiBaseUrl() {
  // 1 & 2 — explicit override (env var or app.json extra). Wins over everything.
  const explicit =
    process.env.EXPO_PUBLIC_API_URL ||
    Constants.expoConfig?.extra?.apiUrl
  if (explicit && typeof explicit === 'string') {
    // Accept either a full URL ("http://host:5000") or a bare host/IP.
    return /^https?:\/\//i.test(explicit) ? explicit : `http://${explicit}:5000`
  }

  // 3 — LAN auto-detect, but only when Metro's host is a real LAN address.
  const host = metroHost()
  if (host && !TUNNEL_HOST_PATTERN.test(host)) {
    return `http://${host}:5000`
  }

  // Tunnel mode with no explicit backend URL — warn loudly; the app will show
  // "Server unreachable" until EXPO_PUBLIC_API_URL is set.
  if (host && __DEV__) {
    console.warn(
      `[ISpyPill] Running over a tunnel (${host}) but EXPO_PUBLIC_API_URL is not set. ` +
      'The backend cannot be auto-detected in tunnel mode. ' +
      'Start Expo with EXPO_PUBLIC_API_URL pointing at your Flask server, e.g.\n' +
      '  EXPO_PUBLIC_API_URL=http://192.168.1.50:5000 npx expo start --tunnel',
    )
  }

  // 4 — simulator / web fallback.
  return 'http://localhost:5000'
}

export const API_BASE_URL = resolveApiBaseUrl()

if (__DEV__) {
  console.log(`[ISpyPill] Flask API → ${API_BASE_URL}`)
}
