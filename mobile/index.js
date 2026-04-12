import { registerRootComponent } from 'expo'
import App from './App'

// registerRootComponent calls AppRegistry.registerComponent('main', ...) AND
// bootstraps Expo.fx → winter runtime → WHATWG URL polyfill (which adds the
// missing `protocol` setter that React Native's built-in URL class omits).
registerRootComponent(App)
