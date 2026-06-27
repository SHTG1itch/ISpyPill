// Dynamic Expo config.
//
// Expo loads the static `app.json` first and passes it here as `config`. We keep
// everything from app.json untouched and only ADD `extra.apiUrl` — the address of
// the Flask backend — computed from this computer's LAN IP at start time.
//
// Why: the mobile app needs to reach Flask on port 5000, but the Expo dev server's
// reported host (`Constants.expoConfig.hostUri`) is unreliable — it reflects however
// the phone happened to connect (it can come back as `localhost`/`127.0.0.1`), which
// on the phone points at the phone itself, not your computer. By detecting the LAN
// IP here (in Node, on your machine) and baking it into `extra.apiUrl`, the app
// always targets the right backend. `config.js` reads `extra.apiUrl` with priority
// over the hostUri auto-detect, so this "just works" with plain `npx expo start`.
//
// An explicit EXPO_PUBLIC_API_URL still overrides everything (used for tunnel mode).

const os = require('os')

// Pick the most likely LAN IPv4 address of this machine.
function detectLanIp() {
  const ifaces = os.networkInterfaces()

  // Collect all external (non-internal) IPv4 addresses.
  const candidates = []
  for (const [name, addrs] of Object.entries(ifaces)) {
    for (const addr of addrs || []) {
      if (addr.family !== 'IPv4' || addr.internal) continue
      if (addr.address.startsWith('169.254.')) continue // link-local / APIPA
      candidates.push({ name, address: addr.address })
    }
  }

  if (candidates.length === 0) return null

  // Prefer real Wi-Fi / Ethernet adapters over virtual ones (VMware, VirtualBox,
  // Hyper-V, WSL, Docker, loopback bridges) which often carry private IPs too.
  const isVirtual = (name) =>
    /vmware|virtualbox|vethernet|hyper-v|wsl|docker|loopback|bluetooth|vpn|tailscale|zerotier/i.test(name)

  const preferred = candidates.find((c) => !isVirtual(c.name))
  return (preferred || candidates[0]).address
}

module.exports = ({ config }) => {
  // EXPO_PUBLIC_API_URL (if set) wins and is read directly by config.js, so we
  // only need to provide the LAN default when it isn't set.
  const lanIp = detectLanIp()
  const apiUrl =
    !process.env.EXPO_PUBLIC_API_URL && lanIp ? `http://${lanIp}:5000` : undefined

  return {
    ...config,
    extra: {
      ...config.extra,
      ...(apiUrl ? { apiUrl } : {}),
    },
  }
}
