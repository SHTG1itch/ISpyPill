import Constants from 'expo-constants'

// Resolve the LAN host of the machine running Metro + Flask.
// We try several Constants locations because the API changed across SDK versions:
//   SDK 49+ / Expo Go:  Constants.expoGoConfig.debuggerHost  = "192.168.x.x:8081"
//   Some SDK versions:  Constants.expoConfig.hostUri          = "192.168.x.x:8081"
//   Legacy (< SDK 46):  Constants.manifest.debuggerHost       = "192.168.x.x:8081"
// All three formats are "<host>:<port>", so we take the part before the first colon.
function resolveHost() {
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
  return 'localhost' // simulator / web only — won't work on a physical device
}

const HOST = resolveHost()

if (__DEV__) {
  console.log(`[ISpyPill] Flask API → http://${HOST}:5000`)
}

export const API_BASE_URL = `http://${HOST}:5000`
