import { useRef, useEffect } from 'react'
import { View, Text, Pressable, StyleSheet, Animated } from 'react-native'
import { Feather } from '@expo/vector-icons'

export default function ErrorBanner({ message, onDismiss }) {
  const shake = useRef(new Animated.Value(0)).current

  useEffect(() => {
    Animated.sequence([
      Animated.timing(shake, { toValue:  6, duration: 60, useNativeDriver: true }),
      Animated.timing(shake, { toValue: -5, duration: 60, useNativeDriver: true }),
      Animated.timing(shake, { toValue:  4, duration: 60, useNativeDriver: true }),
      Animated.timing(shake, { toValue: -3, duration: 60, useNativeDriver: true }),
      Animated.timing(shake, { toValue:  0, duration: 60, useNativeDriver: true }),
    ]).start()
  }, [message])

  return (
    <Animated.View
      style={[styles.banner, { transform: [{ translateX: shake }] }]}
      accessibilityRole="alert"
    >
      <Feather name="alert-circle" size={18} color="#dc2626" style={styles.icon} />
      <Text style={styles.text}>{message}</Text>
      <Pressable
        onPress={onDismiss}
        hitSlop={12}
        style={({ pressed }) => [styles.dismiss, pressed && { opacity: 0.5 }]}
        accessibilityLabel="Dismiss error"
      >
        <Feather name="x" size={17} color="#b91c1c" />
      </Pressable>
    </Animated.View>
  )
}

const styles = StyleSheet.create({
  banner: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    backgroundColor: '#fef2f2',
    borderWidth: 1.5,
    borderColor: '#fca5a5',
    borderRadius: 14,
    padding: 14,
  },
  icon: {
    marginTop: 1,
    flexShrink: 0,
  },
  text: {
    flex: 1,
    fontSize: 14,
    color: '#b91c1c',
    lineHeight: 21,
    fontWeight: '500',
  },
  dismiss: {
    flexShrink: 0,
    marginTop: 1,
  },
})
