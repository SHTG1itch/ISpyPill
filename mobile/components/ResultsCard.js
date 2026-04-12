import { useState, useEffect, useRef } from 'react'
import {
  View, Text, Image, Pressable, StyleSheet, Share, Animated,
} from 'react-native'
import { Feather } from '@expo/vector-icons'

export default function ResultsCard({ data, onReset }) {
  const [displayCount, setDisplayCount] = useState(0)

  // Entrance animation
  const fade  = useRef(new Animated.Value(0)).current
  const slide = useRef(new Animated.Value(24)).current

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fade,  { toValue: 1, duration: 400, useNativeDriver: true }),
      Animated.spring(slide, { toValue: 0, friction: 9, tension: 60, useNativeDriver: true }),
    ]).start()
  }, [])

  // Count-up animation
  useEffect(() => {
    const target = data.count || 0
    if (target === 0) { setDisplayCount(0); return }

    const STEPS = 32
    let step = 0
    const timer = setInterval(() => {
      step++
      const t     = step / STEPS
      const eased = 1 - Math.pow(1 - t, 3)   // ease-out cubic
      setDisplayCount(Math.round(eased * target))
      if (step >= STEPS) { setDisplayCount(target); clearInterval(timer) }
    }, 16)
    return () => clearInterval(timer)
  }, [data.count])

  const badges = []
  if (typeof data.num_color_clusters === 'number')
    badges.push(data.num_color_clusters === 1 ? '1 color' : `${data.num_color_clusters} colors`)
  if (data.is_white_pill !== undefined)
    badges.push(data.is_white_pill ? 'Brightness mode' : 'Color mode')
  if (typeof data.ref_area_px === 'number')
    badges.push(`Ref: ${data.ref_area_px.toLocaleString()} px²`)

  const imgUri = data.annotated_image
    ? `data:image/jpeg;base64,${data.annotated_image}`
    : null

  async function handleShare() {
    try {
      await Share.share({
        message: `ISpyPill counted ${data.count} pill${data.count !== 1 ? 's' : ''}.`,
      })
    } catch { /* user cancelled */ }
  }

  return (
    <Animated.View style={[styles.card, { opacity: fade, transform: [{ translateY: slide }] }]}>

      {/* Count hero */}
      <View style={styles.countHero}>
        <View style={styles.countBubble}>
          <Text style={styles.countLabel}>PILLS COUNTED</Text>
          <Text style={styles.countNumber}>{displayCount.toLocaleString()}</Text>
        </View>
      </View>

      {/* Badges */}
      {badges.length > 0 && (
        <View style={styles.badgesRow}>
          {badges.map(b => (
            <View key={b} style={styles.badge}>
              <Text style={styles.badgeText}>{b}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Detection map */}
      {imgUri && (
        <View style={styles.mapSection}>
          <View style={styles.mapHeader}>
            <Feather name="grid" size={13} color="#64748b" />
            <Text style={styles.mapLabel}>Detection Map</Text>
          </View>
          <Image
            source={{ uri: imgUri }}
            style={styles.mapImg}
            resizeMode="contain"
          />
        </View>
      )}

      {/* Actions */}
      <View style={styles.actions}>
        <Pressable
          style={({ pressed }) => [styles.btnOutline, pressed && styles.btnOutlinePressed]}
          onPress={handleShare}
          accessibilityLabel="Share result"
        >
          <Feather name="share-2" size={15} color="#475569" />
          <Text style={styles.btnOutlineText}>Share</Text>
        </Pressable>

        <Pressable
          style={({ pressed }) => [styles.btnPrimary, pressed && styles.btnPrimaryPressed]}
          onPress={onReset}
          accessibilityLabel="Count another batch of pills"
        >
          <Feather name="refresh-cw" size={15} color="#fff" />
          <Text style={styles.btnPrimaryText}>Count Another</Text>
        </Pressable>
      </View>
    </Animated.View>
  )
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#ffffff',
    borderRadius: 20,
    padding: 22,
    shadowColor: '#1e3a5f',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.12,
    shadowRadius: 18,
    elevation: 6,
    gap: 18,
  },
  countHero: {
    alignItems: 'center',
  },
  countBubble: {
    alignItems: 'center',
    backgroundColor: '#eff6ff',
    borderRadius: 20,
    paddingHorizontal: 36,
    paddingVertical: 20,
    width: '100%',
  },
  countLabel: {
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 2,
    color: '#60a5fa',
    marginBottom: 4,
  },
  countNumber: {
    fontSize: 88,
    fontWeight: '800',
    color: '#1e40af',
    letterSpacing: -4,
    lineHeight: 96,
  },
  badgesRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 7,
    justifyContent: 'center',
  },
  badge: {
    backgroundColor: '#f1f5f9',
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  badgeText: {
    fontSize: 12,
    color: '#475569',
    fontWeight: '600',
  },
  mapSection: {
    gap: 10,
  },
  mapHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    justifyContent: 'center',
  },
  mapLabel: {
    fontSize: 12,
    fontWeight: '700',
    color: '#64748b',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  mapImg: {
    width: '100%',
    height: 240,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    backgroundColor: '#f8fafc',
  },
  actions: {
    flexDirection: 'row',
    gap: 10,
  },
  btnOutline: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 14,
    backgroundColor: '#f8fafc',
    borderWidth: 1.5,
    borderColor: '#e2e8f0',
    borderRadius: 13,
    minHeight: 50,
  },
  btnOutlinePressed: {
    backgroundColor: '#f1f5f9',
    borderColor: '#cbd5e1',
  },
  btnOutlineText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#475569',
  },
  btnPrimary: {
    flex: 2,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
    paddingVertical: 14,
    backgroundColor: '#2563eb',
    borderRadius: 13,
    minHeight: 50,
    shadowColor: '#1e40af',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    elevation: 4,
  },
  btnPrimaryPressed: {
    backgroundColor: '#1d4ed8',
    transform: [{ scale: 0.97 }],
  },
  btnPrimaryText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#ffffff',
  },
})
