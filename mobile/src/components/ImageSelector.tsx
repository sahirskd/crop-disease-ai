import React from 'react';
import { View, Text, Image, StyleSheet } from 'react-native';

interface ImageSelectorProps {
  imageUri: string | null;
  children?: React.ReactNode;
}

export function ImageSelector({ imageUri, children }: ImageSelectorProps) {
  if (imageUri) {
    return (
      <View style={styles.imageContainer}>
        <Image source={{ uri: imageUri }} style={styles.image} />
        {children}
      </View>
    );
  }

  return (
    <View style={styles.placeholder}>
      <Text style={styles.placeholderText}>No image selected</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  imageContainer: {
    width: 300,
    height: 300,
    borderRadius: 10,
    overflow: 'hidden',
    position: 'relative',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  placeholder: {
    width: 300,
    height: 300,
    backgroundColor: '#f0f0f0',
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderText: {
    color: '#888',
  },
});
