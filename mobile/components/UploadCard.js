import { useRef } from 'react'
import {
  View, Text, Image, Pressable, StyleSheet, Alert, Animated,
} from 'react-native'
import * as ImagePicker from 'expo-image-picker'
import { Feather } from '@expo/vector-icons'

export default function UploadCard({ step, title, subtitle, tip, image, onImage, done }) {
  const scaleGallery = useRef(new Animated.Value(1)).current
  const scaleCamera  = useRef(new Animated.Value(1)).current

  function press(anim, cb) {
    Animated.sequence([
      Animated.spring(anim, { toValue: 0.94, useNativeDriver: true, friction: 8 }),
      Animated.spring(anim, { toValue: 1,    useNativeDriver: true, friction: 8 }),
    ]).start()
    cb()
  }

  async function pickFromGallery() {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync()
    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Please allow photo library access in Settings.')
      return
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 0.92,
      allowsEditing: false,
    })
    if (!result.canceled && result.assets?.length > 0) onImage(result.assets[0])
  }

  async function pickFromCamera() {
    const { status } = await ImagePicker.requestCameraPermissionsAsync()
    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Please allow camera access in Settings.')
      return
    }
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      quality: 0.92,
      allowsEditing: false,
    })
    if (!result.canceled && result.assets?.length > 0) onImage(result.assets[0])
  }

  return (
    <View style={[styles.card, done && styles.cardDone]}>
      {/* ── Header ── */}
      <View style={styles.header}>
        <View style={[styles.stepBadge, done && styles.stepBadgeDone]}>
          {done
            ? <Feather name="check" size={14} color="#fff" />
            : <Text style={styles.stepText}>{step}</Text>
          }
        </View>
        <View style={styles.headerText}>
          <Text style={styles.title}>{title}</Text>
          <Text style={styles.subtitle}>{subtitle}</Text>
        </View>
        {done && (
          <View style={styles.donePill}>
            <Text style={styles.donePillText}>Ready</Text>
          </View>
        )}
      </View>

      {/* ── Preview / Placeholder ── */}
      {image ? (
        <View style={styles.preview}>
          <Image source={{ uri: image.uri }} style={styles.previewImg} resizeMode="cover" />
          <Pressable
            style={({ pressed }) => [styles.clearBtn, pressed && styles.clearBtnPressed]}
            onPress={() => onImage(null)}
            hitSlop={12}
            accessibilityLabel="Remove photo"
          >
            <Feather name="x" size={14} color="#fff" />
          </Pressable>
          <View style={styles.changeBar}>
            <Feather name="refresh-cw" size={11} color="rgba(255,255,255,0.85)" />
            <Text style={styles.changeText}>Tap × to replace</Text>
          </View>
        </View>
      ) : (
        <Pressable
          style={({ pressed }) => [styles.placeholder, pressed && styles.placeholderPressed]}
          onPress={pickFromGallery}
          accessibilityLabel={`Add ${title} photo`}
        >
          <View style={styles.placeholderIcon}>
            <Feather name="image" size={26} color="#93c5fd" />
          </View>
          <Text style={styles.placeholderText}>Tap to add photo</Text>
          <Text style={styles.placeholderHint}>PNG · JPG · HEIC</Text>
        </Pressable>
      )}

      {/* ── Action buttons ── */}
      <View style={styles.actions}>
        <Animated.View style={[styles.btnWrap, { transform: [{ scale: scaleGallery }] }]}>
          <Pressable
            style={({ pressed }) => [styles.btn, pressed && styles.btnPressed]}
            onPress={() => press(scaleGallery, pickFromGallery)}
            accessibilityLabel={`Select ${title} from gallery`}
          >
            <Feather name="image" size={15} color="#2563eb" />
            <Text style={styles.btnText}>Gallery</Text>
          </Pressable>
        </Animated.View>

        <Animated.View style={[styles.btnWrap, { transform: [{ scale: scaleCamera }] }]}>
          <Pressable
            style={({ pressed }) => [styles.btn, pressed && styles.btnPressed]}
            onPress={() => press(scaleCamera, pickFromCamera)}
            accessibilityLabel={`Take ${title} photo with camera`}
          >
            <Feather name="camera" size={15} color="#2563eb" />
            <Text style={styles.btnText}>Camera</Text>
          </Pressable>
        </Animated.View>
      </View>

      <View style={styles.tipRow}>
        <Feather name="info" size={12} color="#94a3b8" style={{ marginTop: 1 }} />
        <Text style={styles.tip}>{tip}</Text>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#ffffff',
    borderRadius: 20,
    padding: 16,
    gap: 13,
    shadowColor: '#1e3a5f',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 10,
    elevation: 3,
    borderWidth: 1.5,
    borderColor: 'transparent',
  },
  cardDone: {
    borderColor: '#bfdbfe',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 11,
  },
  stepBadge: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: '#2563eb',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  stepBadgeDone: {
    backgroundColor: '#22c55e',
  },
  stepText: {
    color: '#ffffff',
    fontWeight: '800',
    fontSize: 15,
  },
  headerText: {
    flex: 1,
  },
  title: {
    fontSize: 16,
    fontWeight: '700',
    color: '#0f172a',
    lineHeight: 21,
  },
  subtitle: {
    fontSize: 12,
    color: '#64748b',
    marginTop: 1,
    lineHeight: 16,
  },
  donePill: {
    backgroundColor: '#dcfce7',
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 3,
  },
  donePillText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#16a34a',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  preview: {
    height: 200,
    borderRadius: 14,
    overflow: 'hidden',
    position: 'relative',
    backgroundColor: '#f1f5f9',
  },
  previewImg: {
    width: '100%',
    height: '100%',
  },
  clearBtn: {
    position: 'absolute',
    top: 8,
    right: 8,
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(0,0,0,0.58)',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1,
  },
  clearBtnPressed: {
    backgroundColor: 'rgba(0,0,0,0.80)',
    transform: [{ scale: 0.9 }],
  },
  changeBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0,0,0,0.32)',
    paddingVertical: 6,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 5,
  },
  changeText: {
    color: 'rgba(255,255,255,0.88)',
    fontSize: 11,
    fontWeight: '500',
  },
  placeholder: {
    height: 150,
    borderRadius: 14,
    borderWidth: 2,
    borderColor: '#e2e8f0',
    borderStyle: 'dashed',
    backgroundColor: '#f8fafc',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
  },
  placeholderPressed: {
    backgroundColor: '#eff6ff',
    borderColor: '#93c5fd',
  },
  placeholderIcon: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: '#eff6ff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholderText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#475569',
  },
  placeholderHint: {
    fontSize: 12,
    color: '#94a3b8',
  },
  actions: {
    flexDirection: 'row',
    gap: 9,
  },
  btnWrap: {
    flex: 1,
  },
  btn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
    paddingVertical: 12,
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    minHeight: 46,
  },
  btnPressed: {
    backgroundColor: '#dbeafe',
  },
  btnText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#2563eb',
  },
  tipRow: {
    flexDirection: 'row',
    gap: 5,
    alignItems: 'flex-start',
  },
  tip: {
    flex: 1,
    fontSize: 12,
    color: '#94a3b8',
    lineHeight: 17,
  },
})
