const { getDefaultConfig } = require('expo/metro-config')
const path = require('path')

const config = getDefaultConfig(__dirname)

// Ensure Metro can always find `promise` even when npm deduplication
// places it only inside react-native's own nested node_modules.
config.resolver.extraNodeModules = {
  ...config.resolver.extraNodeModules,
  promise: path.resolve(
    __dirname,
    'node_modules/react-native/node_modules/promise',
  ),
}

module.exports = config
