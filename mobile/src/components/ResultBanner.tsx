import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { InferenceResult } from '../utils/inference';

interface ResultBannerProps {
  result: InferenceResult | null;
}

export function ResultBanner({ result }: ResultBannerProps) {
  if (!result) return null;

  return (
    <View style={styles.resultContainer}>
      <Text style={styles.resultText}>
        {result.className} ({Math.round(result.confidence * 100)}%)
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  resultContainer: {
    marginTop: 20,
    padding: 15,
    backgroundColor: '#e6ffe6',
    borderRadius: 8,
    width: '100%',
    alignItems: 'center',
  },
  resultText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#006600',
  },
});
