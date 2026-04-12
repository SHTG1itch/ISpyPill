import { View, Text, StyleSheet } from 'react-native'
import { Feather } from '@expo/vector-icons'

const STEPS = ['Reference', 'Group', 'Analyze', 'Results']

export default function StepBar({ current }) {
  return (
    <View style={styles.wrapper}>
      {STEPS.map((label, i) => {
        const num      = i + 1
        const done     = current > num
        const active   = current === num
        const upcoming = current < num

        return (
          <View key={label} style={styles.step}>
            {/* Connector line before (not for first step) */}
            {i > 0 && (
              <View style={[styles.line, done && styles.lineDone]} />
            )}

            {/* Circle */}
            <View style={[
              styles.circle,
              done     && styles.circleDone,
              active   && styles.circleActive,
              upcoming && styles.circleUpcoming,
            ]}>
              {done
                ? <Feather name="check" size={11} color="#fff" />
                : <Text style={[styles.num, upcoming && styles.numUpcoming]}>{num}</Text>
              }
            </View>

            {/* Label */}
            <Text style={[
              styles.label,
              active   && styles.labelActive,
              done     && styles.labelDone,
              upcoming && styles.labelUpcoming,
            ]}>
              {label}
            </Text>
          </View>
        )
      })}
    </View>
  )
}

const styles = StyleSheet.create({
  wrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1e40af',
    paddingHorizontal: 16,
    paddingBottom: 14,
    gap: 0,
  },
  step: {
    alignItems: 'center',
    flex: 1,
    position: 'relative',
    flexDirection: 'column',
    gap: 4,
  },
  line: {
    position: 'absolute',
    top: 11,
    right: '50%',
    left: '-50%',
    height: 2,
    backgroundColor: 'rgba(255,255,255,0.25)',
    zIndex: 0,
  },
  lineDone: {
    backgroundColor: '#60a5fa',
  },
  circle: {
    width: 24,
    height: 24,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1,
  },
  circleActive: {
    backgroundColor: '#ffffff',
  },
  circleDone: {
    backgroundColor: '#60a5fa',
  },
  circleUpcoming: {
    backgroundColor: 'rgba(255,255,255,0.18)',
    borderWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  num: {
    fontSize: 11,
    fontWeight: '700',
    color: '#1e40af',
  },
  numUpcoming: {
    color: 'rgba(255,255,255,0.6)',
  },
  label: {
    fontSize: 10,
    fontWeight: '600',
    color: '#ffffff',
  },
  labelActive: {
    color: '#ffffff',
    fontWeight: '700',
  },
  labelDone: {
    color: '#93c5fd',
  },
  labelUpcoming: {
    color: 'rgba(255,255,255,0.45)',
  },
})
