import { useState, useRef, useEffect } from 'react'
import {
  View, Text, ScrollView, Pressable, ActivityIndicator,
  StyleSheet, SafeAreaView, Animated,
} from 'react-native'
import { StatusBar } from 'expo-status-bar'
import UploadCard  from './components/UploadCard'
import ResultsCard from './components/ResultsCard'
import ErrorBanner from './components/ErrorBanner'
import StepBar     from './components/StepBar'
import { API_BASE_URL } from './config'

export default function App() {
  const [refImage,     setRefImage]     = useState(null)
  const [groupImage,   setGroupImage]   = useState(null)
  const [loading,      setLoading]      = useState(false)
  const [error,        setError]        = useState(null)
  const [results,      setResults]      = useState(null)
  const [serverOnline, setServerOnline] = useState(null) // null=unknown, true/false

  // Check server reachability once on mount
  useEffect(() => {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 4000)

    fetch(`${API_BASE_URL}/ping`, { signal: controller.signal })
      .then(r => setServerOnline(r.ok))
      .catch(() => setServerOnline(false))
      .finally(() => clearTimeout(timeoutId))

    return () => {
      clearTimeout(timeoutId)
      controller.abort()
    }
  }, [])

  const scrollRef   = useRef(null)
  const btnScale    = useRef(new Animated.Value(1)).current
  const canAnalyze  = refImage && groupImage && !loading

  // Current step for the StepBar
  const step = results ? 4 : loading ? 3 : groupImage ? 3 : refImage ? 2 : 1

  function animatePress(toValue, cb) {
    Animated.spring(btnScale, { toValue, useNativeDriver: true, friction: 6 }).start(cb)
  }

  async function handleAnalyze() {
    if (!canAnalyze) return
    animatePress(0.96, async () => {
      animatePress(1)
      setLoading(true)
      setError(null)
      setResults(null)

      const body = new FormData()
      body.append('reference_pill', {
        uri:  refImage.uri,
        type: refImage.mimeType || 'image/jpeg',
        name: 'reference.jpg',
      })
      body.append('group_photo', {
        uri:  groupImage.uri,
        type: groupImage.mimeType || 'image/jpeg',
        name: 'group.jpg',
      })

      try {
        const resp = await fetch(`${API_BASE_URL}/analyze`, { method: 'POST', body })
        const data = await resp.json()
        if (!resp.ok) {
          setError(data.error || 'Something went wrong. Please try again.')
          return
        }
        setResults(data)
        setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 150)
      } catch {
        setError(
          `Network error — could not reach ${API_BASE_URL}.\n\n` +
          'Make sure:\n' +
          '1. The Flask server is running: python app.py\n' +
          '2. Your phone and computer are on the same Wi-Fi network.'
        )
      } finally {
        setLoading(false)
      }
    })
  }

  function handleReset() {
    setRefImage(null)
    setGroupImage(null)
    setResults(null)
    setError(null)
    scrollRef.current?.scrollTo({ y: 0, animated: true })
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="light" />

      {/* ── Header ── */}
      <View style={styles.header}>
        <Text style={styles.logoText}>💊 ISpyPill</Text>
        <Text style={styles.tagline}>Accurate pill counting from photos</Text>
        <View style={styles.serverRow}>
          <View style={[
            styles.serverDot,
            serverOnline === true  && styles.dotOnline,
            serverOnline === false && styles.dotOffline,
          ]} />
          <Text style={styles.serverText}>
            {serverOnline === null  ? 'Checking server…' :
             serverOnline === true  ? `Server connected` :
             `Server unreachable — run: python app.py`}
          </Text>
        </View>
      </View>

      {/* ── Step progress bar ── */}
      <StepBar current={step} />

      {/* ── Scrollable body ── */}
      <ScrollView
        ref={scrollRef}
        style={styles.scroll}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <UploadCard
          step={1}
          title="Reference Pill"
          subtitle="A single pill on a contrasting background"
          tip="Lay the pill flat. Fill most of the frame for best accuracy."
          image={refImage}
          onImage={setRefImage}
          done={Boolean(refImage)}
        />

        <UploadCard
          step={2}
          title="Group Photo"
          subtitle="All pills you want to count"
          tip="Spread pills in a single layer. Avoid shadows and overlaps."
          image={groupImage}
          onImage={setGroupImage}
          done={Boolean(groupImage)}
        />

        {error && (
          <ErrorBanner message={error} onDismiss={() => setError(null)} />
        )}

        {/* ── Analyze button ── */}
        <Animated.View style={{ transform: [{ scale: btnScale }] }}>
          <Pressable
            style={[styles.countBtn, !canAnalyze && styles.countBtnDisabled]}
            onPress={handleAnalyze}
            disabled={!canAnalyze}
            accessibilityRole="button"
            accessibilityLabel="Count pills"
            accessibilityState={{ disabled: !canAnalyze, busy: loading }}
          >
            {loading ? (
              <View style={styles.loadingRow}>
                <ActivityIndicator color="#fff" size="small" />
                <Text style={styles.countBtnText}>  Analyzing…</Text>
              </View>
            ) : (
              <Text style={styles.countBtnText}>
                {canAnalyze ? 'Count Pills' : refImage ? 'Add Group Photo' : 'Add Reference Pill'}
              </Text>
            )}
          </Pressable>
        </Animated.View>

        {results && (
          <ResultsCard data={results} onReset={handleReset} />
        )}

        <Text style={styles.disclaimer}>
          For reference only. Always verify counts with a licensed pharmacist.
        </Text>
      </ScrollView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: '#1e40af',
  },
  header: {
    backgroundColor: '#1e40af',
    paddingHorizontal: 20,
    paddingTop: 8,
    paddingBottom: 16,
    alignItems: 'center',
  },
  logoText: {
    fontSize: 28,
    fontWeight: '800',
    color: '#ffffff',
    letterSpacing: -0.5,
    marginBottom: 3,
  },
  tagline: {
    fontSize: 13,
    color: 'rgba(255,255,255,0.72)',
    fontWeight: '400',
  },
  serverRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    marginTop: 6,
  },
  serverDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: 'rgba(255,255,255,0.35)',
  },
  dotOnline: {
    backgroundColor: '#4ade80',
  },
  dotOffline: {
    backgroundColor: '#f87171',
  },
  serverText: {
    fontSize: 11,
    color: 'rgba(255,255,255,0.70)',
  },
  scroll: {
    flex: 1,
    backgroundColor: '#f1f5f9',
  },
  content: {
    padding: 16,
    paddingBottom: 52,
    gap: 14,
  },
  countBtn: {
    backgroundColor: '#2563eb',
    borderRadius: 16,
    paddingVertical: 17,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 58,
    shadowColor: '#1e40af',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.35,
    shadowRadius: 10,
    elevation: 6,
  },
  countBtnDisabled: {
    backgroundColor: '#93c5fd',
    shadowOpacity: 0,
    elevation: 0,
  },
  countBtnText: {
    color: '#ffffff',
    fontSize: 17,
    fontWeight: '700',
    letterSpacing: 0.2,
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  disclaimer: {
    textAlign: 'center',
    fontSize: 11,
    color: '#94a3b8',
    lineHeight: 17,
    marginTop: 4,
  },
})
